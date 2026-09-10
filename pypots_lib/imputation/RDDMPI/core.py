"""Core wrapper for the released RDDMPI imputation model."""

import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.RDDMPI import Backbone_RDDMPI


class _RDDMPI(ModelCore):
    def __init__(
        self,
        n_features,
        n_layers,
        n_heads,
        n_channels,
        d_time_embedding,
        d_feature_embedding,
        d_diffusion_embedding,
        n_diffusion_steps,
        schedule,
        beta_start,
        beta_end,
        model,
        baseline_model,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_time_embedding = d_time_embedding

        self.embed_layer = nn.Embedding(
            num_embeddings=n_features,
            embedding_dim=d_feature_embedding,
        )
        self.backbone = Backbone_RDDMPI(
            n_layers=n_layers,
            n_heads=n_heads,
            n_channels=n_channels,
            d_target=n_features,
            d_time_embedding=d_time_embedding,
            d_feature_embedding=d_feature_embedding,
            d_diffusion_embedding=d_diffusion_embedding,
            n_diffusion_steps=n_diffusion_steps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            model=model,
            baseline_model=baseline_model,
        )

    @staticmethod
    def time_embedding(pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0,
            torch.arange(0, d_model, 2, device=pos.device) / d_model,
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        """Build temporal, feature, and observation-mask conditioning."""
        batch_size, n_features, _ = cond_mask.shape
        device = observed_tp.device

        time_embed = self.time_embedding(observed_tp, self.d_time_embedding)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, n_features, -1)

        feature_embed = self.embed_layer(
            torch.arange(self.n_features, device=device)
        )
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(
            batch_size, observed_tp.shape[1], -1, -1
        )

        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)

        # The released RDDMPI is always conditional on the observation mask.
        side_mask = cond_mask.unsqueeze(1)
        return torch.cat([side_info, side_mask], dim=1)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
        n_sampling_times=1,
    ) -> dict:
        results = {}

        if calc_criterion:
            observed_data = inputs["X_ori"]
            indicating_mask = inputs["indicating_mask"]
            cond_mask = inputs["cond_mask"]
            observed_tp = inputs["observed_tp"]
            side_info = self.get_side_info(observed_tp, cond_mask)

            if self.training:
                results["loss"] = self.backbone.calc_loss(
                    observed_data,
                    cond_mask,
                    indicating_mask,
                    side_info,
                )
            else:
                results["metric"] = self.backbone.calc_loss_valid(
                    observed_data,
                    cond_mask,
                    indicating_mask,
                    side_info,
                )
            return results

        observed_data = inputs["X"]
        cond_mask = inputs["cond_mask"]
        observed_tp = inputs["observed_tp"]
        side_info = self.get_side_info(observed_tp, cond_mask)

        samples = self.backbone(
            observed_data,
            cond_mask,
            side_info,
            n_sampling_times,
        )
        repeated_obs = observed_data.unsqueeze(1).repeat(
            1, n_sampling_times, 1, 1
        )
        repeated_mask = cond_mask.unsqueeze(1).repeat(
            1, n_sampling_times, 1, 1
        )
        imputed_data = repeated_obs + samples * (1 - repeated_mask)

        results["imputation"] = imputed_data.permute(0, 1, 3, 2)
        results["reconstruction"] = samples.permute(0, 1, 3, 2)
        return results
