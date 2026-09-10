"""

"""

import torch
import torch.nn as nn
import time
from ...nn.modules import ModelCore
from ...nn.modules.fgti import BackboneFGTI
from types import SimpleNamespace
import torchcde

class _FGTI(ModelCore):
    def __init__(
            self,
            n_steps: int,
            n_features: int,
            d_time_embedding: int,
            d_feature_embedding: int,
            d_model: int,
            n_heads: int,
            n_encoder_layers: int,
            n_channels: int,
            n_residual_layers: int,
            feature_projection_dim: int,
            n_diffusion_steps: int,
            schedule: str,
            beta_start: float,
            beta_end: float,
            frequency_threshold: float,
            n_top_frequencies: int,
            device: str = "cuda:0",
    ):
        super().__init__()

        if schedule not in {"linear", "quad"}:
            raise ValueError("schedule must be either 'linear' or 'quad'")
        if n_steps <= 0 or n_features <= 0:
            raise ValueError("n_steps and n_features must be positive")
        if n_diffusion_steps <= 0:
            raise ValueError("n_diffusion_steps must be positive")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("beta values must satisfy 0 < beta_start < beta_end < 1")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if d_time_embedding % 2 != 0:
            raise ValueError("d_time_embedding must be even")
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for the diffusion embedding")
        if n_channels % n_heads != 0:
            raise ValueError("n_channels must be divisible by n_heads")
        if n_channels % 4 != 0:
            raise ValueError("n_channels must be divisible by 4 for GroupNorm")
        if frequency_threshold < 0.0 or frequency_threshold > 0.5:
            raise ValueError("frequency_threshold must be in [0, 0.5]")
        if n_top_frequencies <= 0:
            raise ValueError("n_top_frequencies must be positive")

        self.n_steps = n_steps
        self.n_features = n_features
        self.d_time_embedding = d_time_embedding
        self.d_feature_embedding = d_feature_embedding
        self.frequency_threshold = frequency_threshold
        self.n_top_frequencies = n_top_frequencies

        # The original backbone accepts one config object. This namespace is
        # intentionally built only from explicit constructor hyperparameters.
        backbone_config = SimpleNamespace(
            seq_len=n_steps,
            enc_in=n_features,
            timeemb=d_time_embedding,
            featureemb=d_feature_embedding,
            d_model=d_model,
            nheads=n_heads,
            e_layers=n_encoder_layers,
            channel=n_channels,
            residual_layers=n_residual_layers,
            proj_t=feature_projection_dim,
            diffusion_step_num=n_diffusion_steps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
        )

        self.embed_layer = nn.Embedding(n_features, d_feature_embedding)
        self.backbone = BackboneFGTI(backbone_config)

    @staticmethod
    def time_embedding(pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(pos.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2, device=pos.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        device = observed_tp.device
        time_embed = self.time_embedding(observed_tp, self.d_time_embedding)  # (B,L,emb)
        time_embed = time_embed.to(device)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(self.n_features).to(device))  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,emb+d_feature_embedding)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)


        return side_info

    @staticmethod
    def fast_linear_interpolation(
            values: torch.Tensor,
            observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Linearly interpolate [B, L, K] using observed values only."""
        B, L, K = values.shape
        device = values.device

        observed_mask = observed_mask.bool()

        positions = torch.arange(
            L,
            device=device,
            dtype=torch.long,
        ).view(1, L, 1).expand(B, L, K)

        # Most recent observed position at or before every time point.
        previous_positions = torch.where(
            observed_mask,
            positions,
            torch.full_like(positions, -1),
        )
        previous_positions = torch.cummax(
            previous_positions,
            dim=1,
        ).values

        # First observed position at or after every time point.
        next_positions = torch.where(
            observed_mask,
            positions,
            torch.full_like(positions, L),
        )
        next_positions = torch.flip(
            torch.cummin(
                torch.flip(next_positions, dims=[1]),
                dim=1,
            ).values,
            dims=[1],
        )

        previous_indices = previous_positions.clamp(
            min=0,
            max=L - 1,
        )
        next_indices = next_positions.clamp(
            min=0,
            max=L - 1,
        )

        previous_values = values.gather(
            dim=1,
            index=previous_indices,
        )
        next_values = values.gather(
            dim=1,
            index=next_indices,
        )

        # Linear interpolation weight.
        denominator = (
                next_positions - previous_positions
        ).clamp(min=1).to(values.dtype)

        weight = (
                         positions - previous_positions
                 ).to(values.dtype) / denominator

        interpolated = previous_values + weight * (
                next_values - previous_values
        )

        # Keep genuinely observed values unchanged.
        interpolated = torch.where(
            observed_mask,
            values,
            interpolated,
        )

        # Before the first observation, use the first observed value.
        no_previous = previous_positions < 0
        interpolated = torch.where(
            no_previous,
            next_values,
            interpolated,
        )

        # After the final observation, use the final observed value.
        no_next = next_positions >= L
        interpolated = torch.where(
            no_next,
            previous_values,
            interpolated,
        )

        # If a complete feature has no observations, use zero.
        no_observations = ~observed_mask.any(
            dim=1,
            keepdim=True,
        )

        interpolated = torch.where(
            no_observations,
            torch.zeros_like(interpolated),
            interpolated,
        )

        return interpolated

    def build_frequency_condition(
            self,
            observed_data: torch.Tensor,
            cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Construct FGTI's high-pass and dominant-frequency guide.

        Both frequency branches are computed independently for every
        sample and feature, using only values allowed by ``cond_mask``.

        Args:
            observed_data: Tensor with shape [B, K, L].
            cond_mask: Tensor with shape [B, K, L].

        Returns:
            Frequency guide with shape [B, 2*K, L].
        """
        if observed_data.shape != cond_mask.shape:
            raise ValueError(
                "observed_data and cond_mask must have the same shape"
            )

        # Convert from [B,K,L] to [B,L,K].
        interpolated_input = observed_data.permute(
            0, 2, 1
        ).contiguous()

        interpolation_mask = cond_mask.permute(
            0, 2, 1
        ).bool().contiguous()

        # Fast batched linear interpolation.
        interpolated = self.fast_linear_interpolation(
            interpolated_input,
            interpolation_mask,
        )

        # Compute the spectrum once for both branches.
        spectrum = torch.fft.rfft(
            interpolated,
            dim=1,
        )

        # Branch 1: retain frequencies above the threshold.
        frequencies = torch.fft.rfftfreq(
            self.n_steps,
            d=1.0,
            device=interpolated.device,
        )

        high_pass_mask = (
                frequencies.abs() > self.frequency_threshold
        )

        high_pass = torch.fft.irfft(
            spectrum * high_pass_mask.view(1, -1, 1),
            n=self.n_steps,
            dim=1,
        )

        # Branch 2: retain the top-k Fourier coefficients.
        n_frequency_bins = spectrum.shape[1]

        top_k = min(
            self.n_top_frequencies,
            n_frequency_bins,
        )

        top_indices = spectrum.abs().topk(
            top_k,
            dim=1,
        ).indices

        dominant_spectrum = torch.zeros_like(spectrum)

        dominant_spectrum.scatter_(
            dim=1,
            index=top_indices,
            src=spectrum.gather(1, top_indices),
        )

        dominant = torch.fft.irfft(
            dominant_spectrum,
            n=self.n_steps,
            dim=1,
        )

        # Reproduce the original FGTI organization:
        # stack(..., dim=-1).reshape(B, L, 2*K).
        frequency_condition = torch.stack(
            [high_pass, dominant],
            dim=-1,
        )

        frequency_condition = frequency_condition.reshape(
            observed_data.shape[0],
            self.n_steps,
            self.n_features * 2,
        )

        return frequency_condition.permute(
            0, 2, 1
        ).contiguous()

    @staticmethod
    def _unpack_training_inputs(inputs: dict):
        observed_data = inputs["X_ori"]
        cond_mask = inputs["cond_mask"]
        indicating_mask = inputs["indicating_mask"]
        observed_tp = inputs["observed_tp"]
        observed_mask = (cond_mask + indicating_mask).clamp(max=1.0)
        return observed_data, cond_mask, observed_mask, observed_tp

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
        n_sampling_times: int = 1,
    ) -> dict:
        results = {}

        if calc_criterion:
            observed_data, cond_mask, observed_mask, observed_tp = (
                self._unpack_training_inputs(inputs)
            )
            observed_data = torch.nan_to_num(observed_data) * observed_mask
            observed_dataf = self.build_frequency_condition(observed_data, cond_mask)
            side_info = self.get_side_info(observed_tp, cond_mask)

            # The original FGTI implementation uses the same noise-prediction
            # objective for training and validation.
            loss = self.backbone.calc_loss(
                observed_data,
                observed_dataf,
                cond_mask,
                observed_mask,
                side_info,
            )

            results["loss" if self.training else "metric"] = loss
            return results

        observed_data = inputs["X"]
        cond_mask = inputs["cond_mask"]
        observed_tp = inputs["observed_tp"]
        observed_data = torch.nan_to_num(observed_data) * cond_mask

        observed_dataf = self.build_frequency_condition(observed_data, cond_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        samples = self.backbone.impute(
            observed_data,
            observed_dataf,
            cond_mask,
            side_info,
            n_samples=n_sampling_times,
        )

        repeated_observed = observed_data.unsqueeze(1).expand(
            -1, n_sampling_times, -1, -1
        )
        repeated_mask = cond_mask.unsqueeze(1).expand(
            -1, n_sampling_times, -1, -1
        )
        imputed_data = repeated_observed + samples * (1.0 - repeated_mask)

        results["imputation"] = imputed_data.permute(0, 1, 3, 2)
        results["reconstruction"] = samples.permute(0, 1, 3, 2)
        return results

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



