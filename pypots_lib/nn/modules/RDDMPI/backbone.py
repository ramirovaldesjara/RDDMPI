"""Residual diffusion backbone for RDDMPI."""

import numpy as np
import torch
import torch.nn as nn

from .layers import RDDMPI_DiffusionModel


SUPPORTED_BASELINES = {"T1", "ImputeFormer"}


class Backbone_RDDMPI(nn.Module):
    def __init__(
        self,
        n_layers,
        n_heads,
        n_channels,
        d_target,
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

        if baseline_model not in SUPPORTED_BASELINES:
            raise ValueError(
                f"Unsupported baseline_model='{baseline_model}'. "
                f"Supported baselines are: {sorted(SUPPORTED_BASELINES)}."
            )

        self.d_target = d_target
        self.d_time_embedding = d_time_embedding
        self.d_feature_embedding = d_feature_embedding
        self.n_channels = n_channels
        self.n_diffusion_steps = n_diffusion_steps
        self.baseline_model = baseline_model

        # RDDMPI is always conditional in the released implementation:
        # [completed context, noisy residual] + temporal/feature/mask side information.
        d_input = 2
        d_side = d_time_embedding + d_feature_embedding + 1

        self.pretrained_model = model
        self.pretrained_model.eval()
        for parameter in self.pretrained_model.parameters():
            parameter.requires_grad = False

        self.diff_model = RDDMPI_DiffusionModel(
            n_diffusion_steps=n_diffusion_steps,
            d_diffusion_embedding=d_diffusion_embedding,
            d_input=d_input,
            d_side=d_side,
            n_channels=n_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            model=model,
            baseline_model=baseline_model,
        )

        if schedule == "quad":
            self.beta = np.linspace(
                beta_start**0.5, beta_end**0.5, self.n_diffusion_steps
            ) ** 2
        elif schedule == "linear":
            self.beta = np.linspace(beta_start, beta_end, self.n_diffusion_steps)
        else:
            raise ValueError(
                f"schedule must be either 'quad' or 'linear', got '{schedule}'."
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer(
            "alpha_torch",
            torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1),
        )

    def _run_baseline(self, observed_data, cond_mask):
        """Run the frozen deterministic baseline and standardize feature layout.

        Returns
        -------
        pretrained_out : torch.Tensor
            Deterministic baseline reconstruction with shape [B, K, L].
        features : torch.Tensor
            Baseline latent features standardized to [B, K, H, T].
        """
        with torch.no_grad():
            if self.baseline_model == "T1":
                pretrained_out, features = self.pretrained_model(
                    observed_data.permute(0, 2, 1),
                    cond_mask.permute(0, 2, 1),
                )
                pretrained_out = pretrained_out.permute(0, 2, 1)

            elif self.baseline_model == "ImputeFormer":
                inputs = {
                    "X": observed_data.permute(0, 2, 1),
                    "missing_mask": cond_mask.permute(0, 2, 1),
                }
                pretrained_out, features = self.pretrained_model(inputs, None, True)
                pretrained_out = pretrained_out.permute(0, 2, 1)
                # ImputeFormer returns [B, L, K, H]; RDDMPI uses [B, K, H, L].
                features = features.permute(0, 2, 3, 1)

            else:  # guarded in __init__, retained for defensive programming
                raise RuntimeError(f"Unexpected baseline_model='{self.baseline_model}'.")

        return pretrained_out, features

    @staticmethod
    def build_residual_diff_input(
        noisy_residual,
        observed_data,
        pretrained_out,
        cond_mask,
    ):
        """Construct the two-channel RDDMPI diffusion input.

        Channel 0 is the completed signal: observed values where available and
        the deterministic baseline reconstruction at missing positions.
        Channel 1 is the current noisy residual restricted to missing positions.
        """
        noisy_residual_ch = ((1 - cond_mask) * noisy_residual).unsqueeze(1)
        context_channel = (
            cond_mask * observed_data + (1 - cond_mask) * pretrained_out
        ).unsqueeze(1)
        return torch.cat([context_channel, noisy_residual_ch], dim=1)

    def calc_loss_valid(self, observed_data, cond_mask, indicating_mask, side_info):
        loss_sum = 0
        for t in range(self.n_diffusion_steps):
            loss = self.calc_loss(
                observed_data,
                cond_mask,
                indicating_mask,
                side_info,
                set_t=t,
            )
            loss_sum += loss.detach()
        return loss_sum / self.n_diffusion_steps

    def calc_loss(
        self,
        observed_data,
        cond_mask,
        indicating_mask,
        side_info,
        set_t=-1,
    ):
        batch_size = observed_data.shape[0]
        device = observed_data.device

        if self.training:
            t = torch.randint(
                0, self.n_diffusion_steps, [batch_size], device=device
            )
        else:
            t = torch.full(
                (batch_size,), set_t, device=device, dtype=torch.long
            )

        pretrained_out, features = self._run_baseline(observed_data, cond_mask)

        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)

        # RDDMPI learns the uncertainty left after deterministic reconstruction.
        clean_residual = observed_data - pretrained_out
        noisy_residual = (
            current_alpha**0.5 * clean_residual
            + (1.0 - current_alpha) ** 0.5 * noise
        )

        diff_input = self.build_residual_diff_input(
            noisy_residual=noisy_residual,
            observed_data=observed_data,
            pretrained_out=pretrained_out,
            cond_mask=cond_mask,
        )

        predicted_noise = self.diff_model(
            diff_input, side_info, t, features, cond_mask
        )

        target_mask = indicating_mask
        residual = (noise - predicted_noise) * target_mask
        num_eval = target_mask.sum()
        return (residual**2).sum() / (num_eval if num_eval > 0 else 1)

    def _predict_eps(
        self,
        current_sample,
        observed_data,
        pretrained_out,
        cond_mask,
        side_info,
        t,
        features,
    ):
        device = current_sample.device
        batch_size = current_sample.shape[0]

        if not torch.is_tensor(t):
            t = torch.full((batch_size,), t, device=device, dtype=torch.long)
        elif t.dim() == 0:
            t = t.expand(batch_size).to(device)
        else:
            t = t.to(device)

        diff_input = self.build_residual_diff_input(
            noisy_residual=current_sample,
            observed_data=observed_data,
            pretrained_out=pretrained_out,
            cond_mask=cond_mask,
        )
        return self.diff_model(diff_input, side_info, t, features, cond_mask)

    def _ddpm_step(self, x_t, eps_hat, t):
        coeff1 = 1 / self.alpha_hat[t] ** 0.5
        coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
        x_prev = coeff1 * (x_t - coeff2 * eps_hat)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = (
                (1.0 - self.alpha[t - 1])
                / (1.0 - self.alpha[t])
                * self.beta[t]
            ) ** 0.5
            x_prev = x_prev + sigma * noise

        return x_prev

    def _ddim_step(self, x_t, eps_hat, t, s, eta=0.0):
        device = x_t.device
        abar_t = self.alpha_torch[t].to(device).view(1, 1, 1)
        abar_s = self.alpha_torch[s].to(device).view(1, 1, 1)

        x0_hat = (
            x_t - torch.sqrt(1.0 - abar_t) * eps_hat
        ) / torch.sqrt(abar_t)

        if eta == 0.0:
            return torch.sqrt(abar_s) * x0_hat + torch.sqrt(1.0 - abar_s) * eps_hat

        sigma_t = eta * torch.sqrt(
            ((1.0 - abar_s) / (1.0 - abar_t)) * (1.0 - abar_t / abar_s)
        )
        noise = torch.randn_like(x_t)
        return (
            torch.sqrt(abar_s) * x0_hat
            + torch.sqrt(
                torch.clamp(1.0 - abar_s - sigma_t**2, min=0.0)
            )
            * eps_hat
            + sigma_t * noise
        )

    def _make_ddim_schedule(self, ddim_steps, device, schedule_type="logsnr"):
        if ddim_steps > self.n_diffusion_steps:
            raise ValueError("ddim_steps cannot exceed n_diffusion_steps")

        schedule_type = schedule_type.lower()
        if schedule_type == "uniform":
            times = torch.linspace(
                self.n_diffusion_steps - 1,
                0,
                steps=ddim_steps,
                device=device,
            )
            times = torch.unique_consecutive(torch.round(times).long())

        elif schedule_type == "logsnr":
            alpha_bar = torch.tensor(
                self.alpha, device=device, dtype=torch.float32
            )
            alpha_bar = torch.clamp(alpha_bar, 1e-8, 1.0 - 1e-8)
            logsnr = torch.log(alpha_bar) - torch.log(1.0 - alpha_bar)
            target_logsnr = torch.linspace(
                logsnr[-1], logsnr[0], steps=ddim_steps, device=device
            )
            indices = [
                torch.argmin(torch.abs(logsnr - value))
                for value in target_logsnr
            ]
            times = torch.tensor(indices, device=device, dtype=torch.long)
            times = torch.unique_consecutive(
                torch.sort(times, descending=True).values
            )
        else:
            raise ValueError(
                f"Unknown ddim schedule_type='{schedule_type}'. "
                "Use 'uniform' or 'logsnr'."
            )

        if times[-1].item() != 0:
            times = torch.cat(
                [times, torch.zeros(1, device=device, dtype=torch.long)]
            )
        return times

    def forward(
        self,
        observed_data,
        cond_mask,
        side_info,
        n_sampling_times,
        sampler="ddpm",
        ddim_steps=10,
        eta=0.0,
        ddim_schedule_type="uniform",
    ):
        batch_size, n_features, n_steps = observed_data.shape
        device = observed_data.device
        imputed_samples = torch.zeros(
            batch_size,
            n_sampling_times,
            n_features,
            n_steps,
            device=device,
        )

        pretrained_out, features = self._run_baseline(observed_data, cond_mask)

        for i in range(n_sampling_times):
            current_sample = torch.randn_like(observed_data)

            if sampler.lower() == "ddpm":
                for t in range(self.n_diffusion_steps - 1, -1, -1):
                    eps_hat = self._predict_eps(
                        current_sample,
                        observed_data,
                        pretrained_out,
                        cond_mask,
                        side_info,
                        t,
                        features,
                    )
                    current_sample = self._ddpm_step(current_sample, eps_hat, t)

            elif sampler.lower() == "ddim":
                times = self._make_ddim_schedule(
                    ddim_steps, device, schedule_type=ddim_schedule_type
                )
                for j in range(len(times) - 1):
                    t = times[j].item()
                    s = times[j + 1].item()
                    eps_hat = self._predict_eps(
                        current_sample,
                        observed_data,
                        pretrained_out,
                        cond_mask,
                        side_info,
                        t,
                        features,
                    )
                    current_sample = self._ddim_step(
                        current_sample, eps_hat, t, s, eta=eta
                    )
            else:
                raise ValueError(
                    f"Unknown sampler='{sampler}'. Use 'ddpm' or 'ddim'."
                )

            # Convert sampled residuals back into imputations.
            imputed_samples[:, i] = (
                (1 - cond_mask) * (pretrained_out + current_sample)
            ).detach()

        return imputed_samples
