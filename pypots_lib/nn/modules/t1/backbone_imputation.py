import math
import torch
import torch.nn as nn
from omegaconf.listconfig import ListConfig

# --------------------------------------------------------------------------- #
# 1. Utilities
# --------------------------------------------------------------------------- #
def compute_out_len(L: int, k: int, s: int) -> int:
    """Compute output length with front padding (causal padding)"""
    return ceil_div(L, s)

def pick(val, idx: int) -> int:
    '''Pick Kernel size for each stage'''
    return int(val[idx]) if isinstance(val, (list, tuple, ListConfig)) else int(val)

def ceil_div(a: int, b: int) -> int:
    """Ceiling division: returns ⌈a/b⌉"""
    return (a + b - 1) // b

# --------------------------------------------------------------------------- #
# 2. Stem padding
# --------------------------------------------------------------------------- #
class FrontPadding(nn.Module):
    def __init__(self, patch_size: int, stride: int):
        super().__init__()
        self.k, self.s = patch_size, stride
    
    def forward(self, x):
        T = x.size(-1)
        # Calculate padding needed for ceil division
        out_len = ceil_div(T, self.s)
        total_len_needed = (out_len - 1) * self.s + self.k
        pad = max(0, total_len_needed - T)
        if pad == 0: 
            return x
        return torch.cat([x[..., -1:].repeat(*(1,)*(x.dim()-1), pad), x], dim=-1)

# --------------------------------------------------------------------------- #
# 3. Core Blocks: ConvMix, FFN
# --------------------------------------------------------------------------- #
class DepthwiseMix(nn.Module):
    def __init__(self, Cin: int, Cout: int, kL: int, kS: int, bias: bool):
        super().__init__()
        self.large = nn.Conv1d(Cin, Cout, kL, padding=kL//2, groups=Cin, bias=bias)
        self.small = nn.Conv1d(Cin, Cout, kS, padding=kS//2, groups=Cin, bias=False)
    
    def forward(self, x):
        return self.small(x) + self.large(x)

class FeedForward(nn.Module):
    def __init__(self, cfg, H: int):
        super().__init__()
        hidden = int(H * cfg.ffn_ratio)
        self.conv1 = nn.Conv1d(H, hidden, 1)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(getattr(cfg, 'drop_ffn', 0.0))
        self.conv2 = nn.Conv1d(hidden, H, 1)

    def forward(self, x):
        B, M, H, T = x.shape
        x = x.reshape(B * M, H, T)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x.reshape(B, M, H, T)

# --------------------------------------------------------------------------- #
# 4. Attention Module
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, cfg, T: int, kL: int, kS: int):
        super().__init__()
        self.cfg = cfg
        H = cfg.n_heads
        M = cfg.enc_in
        self.T = T
        self.M = M
        
        # Always use shared projections for Q, K, V
        self.q_proj = DepthwiseMix(H, H, kL, kS, cfg.qkv_bias)
        self.k_proj = DepthwiseMix(H, H, kL, kS, cfg.qkv_bias)
        self.v_proj = DepthwiseMix(H, H, kL, kS, cfg.qkv_bias)
        
        self.drop_attn = nn.Dropout(getattr(cfg, 'drop_attn', 0.1))
        self.proj = nn.Conv1d(H, H, 1)
        self.drop_proj = nn.Dropout(getattr(cfg, 'drop_proj', 0.0))
    
    def _compute_qkv(self, x):
        """Compute Q, K, V projections"""
        B, M, H, T = x.shape
        x_reshaped = x.reshape(B*M, H, T)
        q = self.q_proj(x_reshaped).reshape(B, M, H, T).permute(0, 2, 1, 3)  # [B, H, M, T]
        k = self.k_proj(x_reshaped).reshape(B, M, H, T).permute(0, 2, 1, 3)  # [B, H, M, T]
        v = self.v_proj(x_reshaped).reshape(B, M, H, T).permute(0, 2, 1, 3)  # [B, H, M, T]
        
        return q, k, v
    
    def forward(self, x):
        B, M, H, T = x.shape
        
        # Compute Q, K, V with shared projections
        q, k, v = self._compute_qkv(x)
        
        # Standard attention mechanism
        scale = 1.0 / math.sqrt(T)
        attn = (q @ k.transpose(-2,-1)) * scale  # [B, H, M, M]
        attn_scores = attn.softmax(-1)
        attn = self.drop_attn(attn_scores)
        out = attn @ v  # [B, H, M, T]
        
        # Final projection and dropout
        out = self.proj(out.transpose(1,2).reshape(B*M, H, T))
        out = self.drop_proj(out.reshape(B, M, H, T))
        
        return out

# --------------------------------------------------------------------------- #
# 5. Simple DropPath Implementation
# --------------------------------------------------------------------------- #
class DropPath(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor

# --------------------------------------------------------------------------- #
# 6. T1Block
# --------------------------------------------------------------------------- #
class T1Block(nn.Module):
    def __init__(self, cfg, T: int, kL: int, kS: int):
        super().__init__()
        M, H = cfg.enc_in, cfg.n_heads
        self.cfg = cfg

        self.attn = Attention(cfg, T, kL, kS)
        self.ffn = FeedForward(cfg, H)
        self.dp1 = DropPath(getattr(cfg, 'drop_path', 0.1))
        self.dp2 = DropPath(getattr(cfg, 'drop_path', 0.1))

        self.norm1 = nn.LayerNorm((H, T), eps=1e-5)
        self.norm2 = nn.LayerNorm((H, T), eps=1e-5)
        self.scale1 = nn.Parameter(torch.ones(1, 1, 1, 1) * 1e-6)
        self.scale2 = nn.Parameter(torch.ones(1, 1, 1, 1) * 1e-6)

    def forward(self, x):
        # First attention block
        attn_out = self.attn(x)
        y1 = self.norm1(attn_out)
        y1_scaled = self.scale1 * y1
        y1_dropped = self.dp1(y1_scaled)
        x = x + y1_dropped

        # Second FFN block
        y2 = self.norm2(self.ffn(x))
        y2_scaled = self.scale2 * y2
        y2_dropped = self.dp2(y2_scaled)
        x = x + y2_dropped

        return x

# --------------------------------------------------------------------------- #
# 7. Stage & Downsample
# --------------------------------------------------------------------------- #
class DownSample(nn.Module):
    def __init__(self, k: int, s: int, C: int):
        super().__init__()
        self.k, self.s = k, s
        self.dw = nn.Conv1d(C, C, k, s, groups=C)
    
    def forward(self, x):
        B, M, C, T = x.shape
        # Calculate padding needed for ceil division output
        out_len = ceil_div(T, self.s)
        total_len_needed = (out_len - 1) * self.s + self.k
        pad = max(0, total_len_needed - T)
        if pad:
            x = torch.cat([x[..., -1:].repeat(1,1,1,pad), x], -1)
        x = self.dw(x.reshape(B*M, C, -1))
        return x.reshape(B, M, C, x.size(-1))

class T1Stage(nn.Module):
    def __init__(self, cfg, n_blk: int, T: int, last: bool, kL: int, kS: int):
        super().__init__()
        self.blocks = nn.ModuleList()

        # Create blocks
        for i in range(n_blk):
            self.blocks.append(T1Block(cfg, T, kL, kS))

        # Downsampling layer
        self.down = None
        if not last:
            self.down = DownSample(cfg.downsample_ratio, cfg.downsample_ratio, cfg.n_heads)

    def forward(self, x):
        # Pass through all blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Apply downsampling if present
        if self.down:
            x = self.down(x)
            
        return x

# --------------------------------------------------------------------------- #
# 8. Heads
# --------------------------------------------------------------------------- #
class PixelShuffle1D(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = r
    
    def forward(self, x):
        B, C, L = x.shape
        assert C % self.r == 0
        out = x.reshape(B, C//self.r, self.r, L).permute(0,1,3,2)
        return out.reshape(B, C//self.r, L*self.r)

class ReconHead(nn.Module):
    def __init__(self, cfg, T_out: int):
        super().__init__()
        self.pred_len = cfg.seq_len  # For imputation, pred_len equals seq_len
        self.head_type = getattr(cfg, 'recon_head_type', 'pixelshuffle')
        
        if self.head_type == 'linear':
            # Simple linear projection: [H*T] -> [pred_len]
            self.linear = nn.Linear(cfg.n_heads * T_out, self.pred_len)
            self.dp = nn.Dropout(getattr(cfg, 'drop_head', 0.0))
        else:  # 'pixelshuffle'
            # Always use ceil_div for upsampling factor
            self.up = ceil_div(self.pred_len, T_out)
            
            # Find adjusted channels (smallest multiple of up >= n_heads)
            self.adjusted_channels = ((cfg.n_heads + self.up - 1) // self.up) * self.up
            
            # 1x1 conv to adjust channels before pixel shuffle
            self.channel_adjust = nn.Conv1d(cfg.n_heads, self.adjusted_channels, 1)
            
            # Pixel shuffle
            self.ps = PixelShuffle1D(self.up)
            
            # Actual length after pixel shuffle
            self.ps_output_len = T_out * self.up
            
            # Output channels after pixel shuffle
            outC = self.adjusted_channels // self.up
            
            # Final projection - always use Linear (head_params_shared = True)
            self.proj = nn.Linear(outC, 1)
            self.dp = nn.Dropout(getattr(cfg, 'drop_head', 0.0))
        
    def forward(self, x):
        B, M, H, T = x.shape
        
        if self.head_type == 'linear':
            # Direct reshape: [B, M, H, T] -> [B*M, H*T]
            x = x.reshape(B * M, H * T)
            # Linear projection: [B*M, H*T] -> [B*M, pred_len]
            y = self.linear(x)
            y = self.dp(y)
            return y.reshape(B, M, self.pred_len)
        
        else:  # 'pixelshuffle'
            # Adjust channels to be multiple of upsampling factor
            x = x.reshape(B * M, H, T)
            x = self.channel_adjust(x)  # H -> adjusted_channels
            
            # Pixel shuffle
            y = self.ps(x.reshape(B, M * self.adjusted_channels, T)).reshape(B * M, self.adjusted_channels // self.up, self.ps_output_len)
            
            # Center crop if needed
            if self.ps_output_len > self.pred_len:
                crop_start = (self.ps_output_len - self.pred_len) // 2
                y = y[..., crop_start:crop_start + self.pred_len]
            
            # Final projection
            y = y.transpose(1, 2)
            return self.dp(self.proj(y)).reshape(B, M, self.pred_len)

# --------------------------------------------------------------------------- #
# 9. Simplified Model with Mask-Aware Imputation
# --------------------------------------------------------------------------- #
class BackboneT1Imputation(nn.Module):
    """T1 backbone specialized for imputation task in PyPOTS"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.seq_len  # For imputation, pred_len equals seq_len
        
        self.stem_pad = FrontPadding(cfg.patch_size, cfg.patch_stride)
        
        # Determine input channels based on mask embedding setting
        if getattr(cfg, 'imputation_use_mask_embedding', False):
            input_channels = 2  # data + mask
        else:
            input_channels = 1  # data only
            
        self.stem = nn.Conv1d(input_channels, cfg.n_heads, cfg.patch_size, cfg.patch_stride)

        # Compute T after stem with front padding
        T_after_stem = compute_out_len(cfg.seq_len, cfg.patch_size, cfg.patch_stride)
        if cfg.positional_encoding:
            self.pos = nn.Parameter(torch.randn(1, cfg.enc_in, cfg.n_heads, T_after_stem) * .02)

        # Build stages
        stages, curT = [], T_after_stem
        for i, n in enumerate(cfg.n_blocks):
            kL = pick(cfg.kernel_size_large, i)
            kS = pick(cfg.kernel_size_small, i)
            is_last_stage = (i == len(cfg.n_blocks) - 1)
            
            stages.append(T1Stage(cfg, n, curT, is_last_stage, kL, kS))
            
            if not is_last_stage:
                # Downsample with ceil division
                curT = compute_out_len(curT, cfg.downsample_ratio, cfg.downsample_ratio)
                
        self.stages = nn.ModuleList(stages)
        
        # Always use ReconHead for imputation
        self.head = ReconHead(cfg, curT)

    def _embed(self, x, mask=None):
        """Enhanced embedding function with optional mask support for imputation
        
        Args:
            x: input tensor [B, T, M]
            mask: mask tensor [B, T, M] (1 for observed, 0 for missing) - only for imputation
        
        Returns:
            embedded tensor [B, M, n_heads, T_out]
        """
        B, T, M = x.shape
        
        # Apply padding to input
        x_padded = self.stem_pad(x.permute(0, 2, 1))  # [B, M, T_padded]
        
        # Check if we should use mask embedding
        use_mask = (self.cfg.imputation_use_mask_embedding and mask is not None)
        
        if use_mask:
            # Convert mask to float type matching x for memory efficiency
            mask = mask.to(x.dtype) / T
            
            # Apply same padding to mask
            mask_padded = self.stem_pad(mask.permute(0, 2, 1))  # [B, M, T_padded]
            
            # Concatenate x and mask along channel dimension
            x_padded = x_padded.reshape(B * M, 1, -1)  # [B*M, 1, T_padded]
            mask_padded = mask_padded.reshape(B * M, 1, -1)  # [B*M, 1, T_padded]
            x_input = torch.cat([x_padded, mask_padded], dim=1)  # [B*M, 2, T_padded]
        else:
            # Standard embedding (for other tasks or imputation without mask)
            x_input = x_padded.reshape(B * M, 1, -1)  # [B*M, 1, T_padded]
        
        # Apply stem convolution
        x_stemmed = self.stem(x_input)  # [B*M, n_heads, T_out]
        x_out = x_stemmed.reshape(B, M, self.cfg.n_heads, -1)  # [B, M, n_heads, T_out]
        
        # Add positional encoding if enabled
        if self.cfg.positional_encoding:
            x_out = x_out + self.pos
        
        return x_out

    def forward_features(self, x, mask=None):
        """Forward through feature extraction stages
        
        Args:
            x: input tensor [B, T, M]
            mask: mask tensor [B, T, M] - only used during embedding for imputation
        
        Returns:
            features tensor [B, M, n_heads, T_final]
        """
        x = self._embed(x, mask=mask)
        for stage in self.stages:
            x = stage(x)
        return x

    def _normalize_input(self, x, mask=None):
        """Normalize input for imputation and return normalization stats"""
        if mask is not None:
            n_observed = torch.sum(mask == 1, dim=1, keepdim=True)  # [B, 1, M]
            
            # Create masks for special cases
            is_zero_obs = (n_observed == 0)
            is_one_obs = (n_observed == 1)
            is_normal = ~(is_zero_obs | is_one_obs)
            
            # Initialize tensors
            mean_enc = torch.zeros_like(n_observed, dtype=x.dtype)
            std_enc = torch.ones_like(n_observed, dtype=x.dtype)
            
            # Case 1: n_observed == 0 -> mean = 0, std = 1
            # (already initialized as such)
            
            # Case 2: n_observed == 1 -> compute mean, std = 1
            if is_one_obs.any():
                mean_one = torch.sum(x * mask, dim=1, keepdim=True)  # Since n_observed=1, this is the single value
                mean_enc = torch.where(is_one_obs, mean_one, mean_enc)
            
            # Case 3: n_observed >= 2 -> normal computation
            if is_normal.any():
                # Safe division with epsilon to avoid numerical issues
                safe_n_observed = torch.where(is_normal, n_observed, torch.ones_like(n_observed))
                
                # Compute mean for normal cases
                mean_normal = torch.sum(x * mask, dim=1, keepdim=True) / safe_n_observed
                mean_enc = torch.where(is_normal, mean_normal, mean_enc)
                
                # Compute centered values
                x_centered = (x - mean_enc) * mask
                
                # Compute std for normal cases (unbiased estimate)
                variance = torch.sum(x_centered * x_centered, dim=1, keepdim=True) / torch.maximum(safe_n_observed - 1, torch.ones_like(safe_n_observed))
                std_normal = torch.sqrt(variance + 1e-5)
                std_enc = torch.where(is_normal, std_normal, std_enc)
            
            # Detach for stability
            mean_enc = mean_enc.detach()
            std_enc = std_enc.detach()
            
            # Normalize
            x_norm = (x - mean_enc) * mask / std_enc
            
        else:
            # Standard normalization
            mean_enc = x.mean(1, keepdim=True).detach()
            x_centered = x - mean_enc
            std_enc = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_norm = x_centered / std_enc
            
        return x_norm, mean_enc, std_enc

    def _forward(self, x_norm, mask=None):
        """Forward pass with optional mask
        
        Args:
            x_norm: normalized input [B, T, M]
            mask: optional mask tensor [B, T, M] for imputation
        
        Returns:
            features tensor
        """
        return self.forward_features(x_norm, mask=mask)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """Imputation task with mask-aware embedding
        
        Args:
            x_enc: input data with missing values [B, T, M]
            mask: binary mask (1 for observed, 0 for missing) [B, T, M]
        
        Returns:
            reconstructed data [B, T, M]
        """
        # Normalize input with mask
        x_norm, mean_enc, std_enc = self._normalize_input(x_enc, mask)
        
        # Forward pass with mask information passed to embedding
        features = self._forward(x_norm, mask=mask)
        
        # Generate reconstruction through head
        y = self.head(features).permute(0,2,1)
        
        # Always denormalize
        y = y * std_enc + mean_enc
        
        return y, features

    def forward(self, x_enc, mask):
        """Simplified forward method for imputation only
        
        Args:
            x_enc: input data with missing values [B, T, M]
            mask: binary mask (1 for observed, 0 for missing) [B, T, M]
        
        Returns:
            reconstructed data [B, T, M]
        """

        y, features = self.imputation(x_enc, None, None, None, mask)
        return y, features