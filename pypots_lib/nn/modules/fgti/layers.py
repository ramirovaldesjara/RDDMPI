"""

"""


import torch.fft
from typing import Optional, Any
from torch import nn, Tensor
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import torch


def Attn_tem(heads=8, layers=1, channels=64):
    encoder_layer = TransformerEncoderLayer_QKV(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return TransformerEncoder_QKV(encoder_layer, num_layers=layers)


class Attn_spa(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, itp_x=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        v_len = n if itp_x is None else itp_x.shape[1]
        assert v_len == self.seq_len, f'the sequence length of the values must be {self.seq_len} - {v_len} given'

        q_input = x if itp_x is None else itp_x
        queries = self.to_q(q_input)
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        k_input = x if itp_x is None else itp_x
        v_input = x

        keys = self.to_k(k_input)
        values = self.to_v(v_input) if not self.share_kv else keys
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values
        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer_QKV(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_QKV, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_QKV, self).__setstate__(state)

    def forward(self, query, key, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(query, key, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder_QKV(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_QKV, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(query, key, output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class TemporalLearning(nn.Module):
    def __init__(self, channels, nheads, is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.time_layer = Attn_tem(heads=nheads, layers=1, channels=channels)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)

    def forward(self, y, base_shape, itp_y=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        v = y.permute(2, 0, 1)
        if self.is_cross:
            itp_y = itp_y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
            q = itp_y.permute(2, 0, 1)
            y = self.time_layer(q, q, v).permute(1, 2, 0)
        else:
            y = self.time_layer(v, v, v).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


class FeatureLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, proj_t=16, is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.attn = Attn_spa(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)
        self.norm1_attn = nn.GroupNorm(4, channels)

    def forward(self, y, base_shape, itp_y=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y_attn = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_cross:
            itp_y_attn = itp_y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
            y_attn = self.attn(y_attn.permute(0, 2, 1), itp_y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y_attn = self.attn(y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn = y_attn.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        y_attn = self.norm1_attn(y_attn)

        return y_attn


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,  is_causal: bool = False, **kwargs,) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, configs, dim_feedforward=128, dropout=0.1, pos_encoding='learnable', activation='gelu',
                 freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.configs = configs
        self.max_len = configs.seq_len
        self.d_model = configs.d_model
        self.n_heads = configs.nheads

        self.project_inp = nn.Linear(configs.enc_in * 2, self.d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(self.d_model, dropout=dropout * (1.0 - freeze),
                                                     max_len=self.max_len)

        encoder_layer = TransformerBatchNormEncoderLayer(self.d_model, self.n_heads, dim_feedforward,
                                                         dropout * (1.0 - freeze), activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, configs.e_layers)

        self.output_layer = nn.Linear(self.d_model, configs.enc_in)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = configs.enc_in

        self.criterion = nn.MSELoss(reduction="mean")

    def encoder(self, X):
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        # return shape: (batch_size, seq_length, d_model)
        return output

    def get_random_mask(self, data_batch, mask_batch, mask_ratio_ssl, avg_mask_len):
        mask_ratio_ssl = mask_ratio_ssl + self.configs.missing_rate

        mask_temp = torch.rand_like(mask_batch)
        mask_temp[mask_temp <= mask_ratio_ssl] = 0
        mask_temp[mask_temp > mask_ratio_ssl] = 1
        mask_temp = mask_temp.to(self.configs.device)

        mask_ssl = mask_batch * mask_temp
        return mask_ssl

    def get_random_block_mask(self, data_batch, mask_batch, mask_ratio_ssl, avg_mask_len):
        mask_ratio_ssl = mask_ratio_ssl + self.configs.missing_rate

        mask_batch = mask_batch.detach().to("cpu").numpy()
        mask_ssl = []
        for sample_id in range(mask_batch.shape[0]):
            mask = mask_batch[sample_id]
            # for each attribute generate independent mask
            for j in range(mask.shape[1]):
                choice_num = int(mask.shape[0] * mask_ratio_ssl / avg_mask_len)
                mask_indexs = np.random.choice(mask.shape[0], choice_num)
                mask_lens = np.random.geometric(1 / avg_mask_len, choice_num)
                for i in range(choice_num):
                    mask[mask_indexs[i]: mask_indexs[i] + mask_lens[i], j] = 0
            mask_ssl.append(mask)
        mask_ssl = np.array(mask_ssl)
        mask_ssl = torch.from_numpy(mask_ssl).float().to(self.configs.device)
        return mask_ssl

    def get_tst_mask(self, data_batch, mask_batch, mask_ratio_ssl, avg_mask_len):
        mask_ratio_ssl = mask_ratio_ssl + self.configs.missing_rate
        mask_batch = mask_batch.detach().to("cpu").numpy()
        mask_ssl = []
        for sample_id in range(mask_batch.shape[0]):
            mask = mask_batch[sample_id]

            tst_mask = []
            # for each attribute generate independent mask
            for j in range(mask.shape[1]):
                keep_mask = np.ones(mask.shape[0])
                p_m = 1 / avg_mask_len  # probability of each masking sequence stopping. parameter of geometric distribution.
                p_u = p_m * mask_ratio_ssl / (
                            1 - mask_ratio_ssl)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
                p = [p_m, p_u]

                # Start in state 0 with masking_ratio probability
                state = int(np.random.rand() > mask_ratio_ssl)  # state 0 means masking, 1 means not masking
                for i in range(mask.shape[0]):
                    keep_mask[
                        i] = state  # here it happens that state and masking value corresponding to state are identical
                    if np.random.rand() < p[state]:
                        state = 1 - state
                tst_mask.append(keep_mask)

            tst_mask = np.vstack(tst_mask).T
            tst_mask = mask * tst_mask
            mask_ssl.append(tst_mask)
        mask_ssl = np.array(mask_ssl)
        mask_ssl = torch.from_numpy(mask_ssl).float().to(self.configs.device)
        return mask_ssl

    def self_supervised_learning(self, X, mask):
        ssl_label = X * mask

        mask_ssl = self.get_tst_mask(ssl_label, mask, mask_ratio_ssl=self.configs.mask_ratio_ssl,
                                     avg_mask_len=self.configs.avg_mask_len_ssl)

        X_input = X * mask_ssl
        output = self.encoder(X_input)
        output = self.dropout1(output)
        output = self.output_layer(output)

        eval_ssl = mask - mask_ssl
        eval_p = torch.where(eval_ssl == 1)
        loss = self.criterion(output[eval_p], ssl_label[eval_p])

        return loss

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        output = self.encoder(X)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output



class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class diff_FGTI(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.channel = configs.channel
        self.residual_layers = self.configs.residual_layers

        self.encoder = TSTransformerEncoder(configs)

        # condition:
        self.condition_projection = Conv1d_with_init(self.configs.d_model, self.configs.enc_in * self.channel, 1)
        self.side_projection = Conv1d_with_init(self.configs.side_dim, self.channel, 1)
        self.condition_projection2 = Conv1d_with_init(self.channel, self.channel, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=configs.diffusion_step_num,
            embedding_dim=configs.d_model
        )

        self.input_projection = Conv1d_with_init(2, self.channel, 1)
        self.output_projection1 = Conv1d_with_init(self.channel, self.channel, 1)
        self.output_projection2 = Conv1d_with_init(self.channel, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=self.configs.side_dim,
                    channels=self.channel,
                    diffusion_embedding_dim=self.configs.d_model,
                    nheads=self.configs.nheads,
                    target_dim=self.configs.enc_in,
                    proj_t=self.configs.proj_t
                )
                for _ in range(self.residual_layers)
            ]
        )

    def forward(self, total_input, side_info, cond_obs, diffusion_step):
        B, inputdim, K, L = total_input.shape

        x = total_input.reshape(B, inputdim, K * L)

        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channel, K, L)

        # condition guide
        conditions = self.encoder.encoder(cond_obs.permute(0, 2, 1)).permute(0, 2, 1)  # [B, d, L]
        conditions = self.condition_projection(conditions)  # [B, d*K, L]
        sides = side_info.reshape(B, -1, K * L)
        sides = self.side_projection(sides)

        conditions = conditions.reshape(B, -1, K * L)
        conditions = conditions + sides
        conditions = self.condition_projection2(conditions)
        conditions = F.relu(conditions)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, side_info, diffusion_emb, conditions)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channel, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class NoiseProject(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, target_dim, proj_t, is_cross_t=True,
                 is_cross_s=True):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.forward_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)
        self.forward_feature = FeatureLearning(channels=channels, nheads=nheads,
                                               target_dim=target_dim, proj_t=proj_t, is_cross=is_cross_s)

    def forward(self, x, side_info, diffusion_emb, guide_info):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape, guide_info)
        y = self.forward_feature(y, base_shape, guide_info)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip
