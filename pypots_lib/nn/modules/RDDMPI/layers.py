"""Neural layers used by the released RDDMPI diffusion model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_BASELINES = {"T1", "ImputeFormer"}


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=64,
        activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class RDDMPI_DiffusionEmbedding(nn.Module):
    def __init__(self, n_diffusion_steps, d_embedding=128, d_projection=None):
        super().__init__()
        if d_projection is None:
            d_projection = d_embedding
        self.register_buffer(
            "embedding",
            self._build_embedding(n_diffusion_steps, d_embedding // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(d_embedding, d_projection)
        self.projection2 = nn.Linear(d_projection, d_projection)

    @staticmethod
    def _build_embedding(n_steps, d_embedding=64):
        steps = torch.arange(n_steps).unsqueeze(1)
        frequencies = 10.0 ** (
            torch.arange(d_embedding) / (d_embedding - 1) * 4.0
        ).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        return F.silu(self.projection2(x))


class RDDMPI_ResidualBlock(nn.Module):
    def __init__(self, d_side, n_channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, n_channels)
        self.cond_projection = conv1d_with_init(d_side, 2 * n_channels, 1)
        self.mid_projection = conv1d_with_init(n_channels, 2 * n_channels, 1)
        self.output_projection = conv1d_with_init(n_channels, 2 * n_channels, 1)
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=n_channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=n_channels)

    def forward_time(self, y, base_shape):
        batch_size, channels, n_features, n_steps = base_shape
        if n_steps == 1:
            return y
        y = (
            y.reshape(batch_size, channels, n_features, n_steps)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * n_features, channels, n_steps)
        )
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        return (
            y.reshape(batch_size, n_features, channels, n_steps)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, channels, n_features * n_steps)
        )

    def forward_feature(self, y, base_shape):
        batch_size, channels, n_features, n_steps = base_shape
        if n_features == 1:
            return y
        y = (
            y.reshape(batch_size, channels, n_features, n_steps)
            .permute(0, 3, 1, 2)
            .reshape(batch_size * n_steps, channels, n_features)
        )
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        return (
            y.reshape(batch_size, n_steps, channels, n_features)
            .permute(0, 2, 3, 1)
            .reshape(batch_size, channels, n_features * n_steps)
        )

    def forward(self, x, cond_info, diffusion_emb):
        batch_size, channels, n_features, n_steps = x.shape
        base_shape = x.shape
        x_flat = x.reshape(batch_size, channels, n_features * n_steps)

        diffusion_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x_flat + diffusion_proj
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)

        cond_dim = cond_info.shape[1]
        cond_info = cond_info.reshape(batch_size, cond_dim, n_features * n_steps)
        y = y + self.cond_projection(cond_info)

        gate, filter_ = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class RDDMPI_DiffusionModel(nn.Module):
    """Conditional residual diffusion network used by RDDMPI.

    The released model always uses the completed deterministic signal,
    reliability gating, and latent features from the frozen baseline. Ablation
    switches used during development are intentionally not part of this API.
    """

    def __init__(
        self,
        n_diffusion_steps,
        d_diffusion_embedding,
        d_input,
        d_side,
        n_channels,
        n_heads,
        n_layers,
        model,
        baseline_model="T1",
    ):
        super().__init__()

        if baseline_model not in SUPPORTED_BASELINES:
            raise ValueError(
                f"Unsupported baseline_model='{baseline_model}'. "
                f"Supported baselines are: {sorted(SUPPORTED_BASELINES)}."
            )
        if d_input != 2:
            raise ValueError(
                "RDDMPI expects exactly two diffusion-input channels: "
                "completed context and noisy residual."
            )

        self.diffusion_embedding = RDDMPI_DiffusionEmbedding(
            n_diffusion_steps=n_diffusion_steps,
            d_embedding=d_diffusion_embedding,
        )
        self.n_channels = n_channels
        self.baseline_model = baseline_model

        self.residual_input_projection = conv1d_with_init(1, n_channels, 1)
        self.context_input_projection = conv1d_with_init(1, n_channels, 1)
        self.reliability_gate = ReliabilityGate(
            in_channels=2,
            d_gate=16,
            k_large=31,
            k_small=5,
        )

        # Reliability alpha is appended to time/feature/mask side information.
        d_side = d_side + 1

        self.output_projection1 = conv1d_with_init(n_channels, n_channels, 1)
        self.output_projection2 = conv1d_with_init(n_channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                RDDMPI_ResidualBlock(
                    d_side=d_side,
                    n_channels=n_channels,
                    diffusion_embedding_dim=d_diffusion_embedding,
                    nheads=n_heads,
                )
                for _ in range(n_layers)
            ]
        )

        self.pretrained_model = model
        self.pretrained_model.eval()
        for parameter in self.pretrained_model.parameters():
            parameter.requires_grad = False

        if baseline_model == "T1":
            cfg = self.pretrained_model.cfg
            self.pred_len = int(cfg.seq_len)
            feature_channels = int(cfg.n_heads)

            head = getattr(self.pretrained_model, "head", None)
            if head is None or not hasattr(head, "ps_output_len") or not hasattr(head, "up"):
                raise ValueError(
                    "T1 backbone does not expose the reconstruction-head dimensions "
                    "required by RDDMPI."
                )

            feature_steps = int(head.ps_output_len // head.up)
            self.up = ceil_div(self.pred_len, feature_steps)
            self.adjusted_channels = (
                (feature_channels + self.up - 1) // self.up
            ) * self.up
            self.channel_adjust = nn.Conv1d(
                feature_channels, self.adjusted_channels, kernel_size=1
            )
            self.ps = PixelShuffle1D(self.up)
            self.outC = self.adjusted_channels // self.up
            self.pre2film = nn.Linear(self.outC, 2 * n_channels)
        else:
            # model.py replaces this with a fully initialized Linear using the
            # frozen ImputeFormer model_dim before PyPOTS counts parameters.
            self.pre2film = None

    def _input_stem(self, x, cond_mask):
        """Fuse residual and completed-context channels using reliability alpha."""
        batch_size, input_dim, n_features, n_steps = x.shape
        if input_dim != 2:
            raise ValueError(
                f"Expected two RDDMPI input channels, received {input_dim}."
            )

        x_context = x[:, 0:1]
        x_residual = x[:, 1:2]
        mask = cond_mask.unsqueeze(1).to(x.dtype)

        alpha = self.reliability_gate(torch.cat([x_context, mask], dim=1))

        h_context = self.context_input_projection(
            x_context.reshape(batch_size, 1, n_features * n_steps)
        )
        h_residual = self.residual_input_projection(
            x_residual.reshape(batch_size, 1, n_features * n_steps)
        )

        h_context = F.relu(h_context).reshape(
            batch_size, self.n_channels, n_features, n_steps
        )
        h_residual = F.relu(h_residual).reshape(
            batch_size, self.n_channels, n_features, n_steps
        )
        return h_residual + alpha * h_context, alpha

    def _prepare_latent_features(self, features, n_steps):
        """Convert standardized baseline features [B,K,H,T] to FiLM input."""
        batch_size, n_features, hidden_dim, feature_steps = features.shape

        if self.baseline_model == "T1":
            z = features.reshape(batch_size * n_features, hidden_dim, feature_steps)
            z = self.channel_adjust(z)
            z = self.ps(z)

            if z.shape[-1] < n_steps:
                raise ValueError(
                    "T1 latent upsampling produced fewer time steps than the "
                    "RDDMPI sequence length. Check the T1 configuration."
                )
            if z.shape[-1] > n_steps:
                crop_start = (z.shape[-1] - n_steps) // 2
                z = z[..., crop_start : crop_start + n_steps]

            return (
                z.reshape(batch_size, n_features, self.outC, n_steps)
                .permute(0, 1, 3, 2)
            )

        if feature_steps != n_steps:
            raise ValueError(
                "ImputeFormer latent time dimension does not match the RDDMPI "
                f"sequence length ({feature_steps} != {n_steps})."
            )
        return features.permute(0, 1, 3, 2)

    def forward(self, x, cond_info, diffusion_step, features, cond_mask):
        batch_size, _, n_features, n_steps = x.shape
        x, alpha = self._input_stem(x, cond_mask)

        z = self._prepare_latent_features(features, n_steps)
        if self.pre2film is None:
            raise RuntimeError("RDDMPI latent projection was not initialized.")
        film = self.pre2film(z)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        gamma = (1.0 + gamma).permute(0, 3, 1, 2)
        beta = beta.permute(0, 3, 1, 2)
        x = gamma * x + beta

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        cond_info = torch.cat([cond_info, alpha], dim=1)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(batch_size, self.n_channels, n_features * n_steps)
        x = F.relu(self.output_projection1(x))
        x = self.output_projection2(x)
        return x.reshape(batch_size, n_features, n_steps)


class PixelShuffle1D(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = r

    def forward(self, x):
        batch_size, channels, length = x.shape
        if channels % self.r != 0:
            raise ValueError(
                f"PixelShuffle1D requires channels divisible by {self.r}, "
                f"received {channels}."
            )
        out = x.reshape(
            batch_size, channels // self.r, self.r, length
        ).permute(0, 1, 3, 2)
        return out.reshape(batch_size, channels // self.r, length * self.r)


def ceil_div(a: int, b: int) -> int:
    """Return ceil(a / b) for positive integer dimensions."""
    return (a + b - 1) // b


class DepthwiseMix(nn.Module):
    """Local temporal mixer used by the reliability gate."""

    def __init__(
        self,
        channels: int,
        k_large: int = 7,
        k_small: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        if k_large % 2 != 1 or k_small % 2 != 1:
            raise ValueError("Reliability-gate kernel sizes must be odd.")
        self.large = nn.Conv1d(
            channels,
            channels,
            k_large,
            padding=k_large // 2,
            groups=channels,
            bias=bias,
        )
        self.small = nn.Conv1d(
            channels,
            channels,
            k_small,
            padding=k_small // 2,
            groups=channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.small(x) + self.large(x)
        return self.act(self.pointwise(y))


class ReliabilityGate(nn.Module):
    """Predict a spatiotemporal reliability map from context and mask."""

    def __init__(
        self,
        in_channels: int = 2,
        d_gate: int = 16,
        k_large: int = 31,
        k_small: int = 5,
    ):
        super().__init__()
        self.in_proj = conv1d_with_init(in_channels, d_gate, 1)
        self.mix = DepthwiseMix(
            channels=d_gate,
            k_large=k_large,
            k_small=k_small,
            bias=True,
        )
        self.out_proj = conv1d_with_init(d_gate, 1, 1)

    def forward(self, ctx):
        batch_size, channels, n_features, n_steps = ctx.shape
        z = (
            ctx.permute(0, 2, 1, 3)
            .reshape(batch_size * n_features, channels, n_steps)
        )
        z = F.gelu(self.in_proj(z))
        z = self.mix(z)
        z = torch.sigmoid(self.out_proj(z))
        return (
            z.reshape(batch_size, n_features, 1, n_steps)
            .permute(0, 2, 1, 3)
        )
