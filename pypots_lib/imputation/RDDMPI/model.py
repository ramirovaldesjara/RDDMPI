"""Public PyPOTS-style wrapper for RDDMPI time-series imputation."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .core import _RDDMPI
from .data import DatasetForCSDI, TestDatasetForCSDI
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...nn.functional import gather_listed_dicts
from ...nn.modules.loss import Criterion
from ...optim.adam import Adam
from ...optim.base import Optimizer


SUPPORTED_BASELINES = {"T1", "ImputeFormer"}


def _project_root() -> Path:
    """Return the repository root from this source file."""
    return Path(__file__).resolve().parents[3]


def _load_baseline_config(dataset: str, baseline_model: str, config_name: str):
    """Load a baseline experiment config, including simple local Hydra defaults.

    RDDMPI uses the same configuration that produced the frozen baseline
    checkpoint. Relative defaults such as ``default_temp`` are merged before
    the experiment-specific file. Global Hydra defaults are intentionally
    ignored here because the baseline constructors only need model/dataset
    fields already present in the baseline configuration hierarchy.
    """
    config_dir = (
        _project_root()
        / "lab"
        / "configs"
        / "imputation_pypots"
        / dataset
        / baseline_model
    )
    config_path = config_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{baseline_model} config not found at {config_path}. "
            "Set baseline_config_name to the configuration used to train the "
            "frozen baseline checkpoint."
        )

    cfg = OmegaConf.load(config_path)
    merged = OmegaConf.create({})

    for entry in cfg.get("defaults", []):
        if not isinstance(entry, str) or entry == "_self_" or entry.startswith("/"):
            continue
        default_name = entry[:-5] if entry.endswith(".yaml") else entry
        default_path = config_dir / f"{default_name}.yaml"
        if default_path.exists():
            merged = OmegaConf.merge(merged, OmegaConf.load(default_path))

    merged = OmegaConf.merge(merged, cfg)
    if "defaults" in merged:
        del merged["defaults"]
    return merged


def _checkpoint_state_dict(checkpoint):
    """Normalize common checkpoint containers and remove a leading model prefix."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected a checkpoint state dictionary, "
            f"received {type(checkpoint).__name__}."
        )

    return {
        (key[len("model.") :] if key.startswith("model.") else key): value
        for key, value in checkpoint.items()
    }


class RDDMPI(BaseNNImputer):
    """Residual Denoising Diffusion Model for probabilistic imputation.

    The released implementation is conditional and always uses a frozen
    deterministic baseline to provide (1) a completed context signal and (2)
    latent features for diffusion conditioning. Supported deterministic
    baselines are T1 and ImputeFormer.

    Parameters
    ----------
    n_steps : int
        Number of time steps per sample.
    n_features : int
        Number of variables/features per sample.
    n_layers : int
        Number of residual diffusion blocks.
    n_heads : int
        Number of attention heads in each diffusion residual block.
    n_channels : int
        Hidden channel width of the diffusion model.
    d_time_embedding : int
        Temporal side-information embedding dimension.
    d_feature_embedding : int
        Feature side-information embedding dimension.
    d_diffusion_embedding : int
        Diffusion-step embedding dimension.
    baseline_model : {"T1", "ImputeFormer"}
        Frozen deterministic baseline used by RDDMPI.
    baseline_config_name : str
        Configuration/checkpoint directory name of the frozen baseline.
    dataset : str
        Dataset configuration directory name.
    seed : int
        Seed suffix used by the saved baseline checkpoint.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        n_heads: int,
        n_channels: int,
        d_time_embedding: int,
        d_feature_embedding: int,
        d_diffusion_embedding: int,
        n_diffusion_steps: int = 50,
        target_strategy: str = "random",
        schedule: str = "quad",
        beta_start: float = 0.0001,
        beta_end: float = 0.5,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
        dataset: str = "ETTh1",
        baseline_config_name: str = "9062_0000",
        seed: int = 2,
        baseline_model: str = "T1",
    ):
        super().__init__(
            training_loss=Criterion,
            validation_metric=Criterion,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        if target_strategy not in {"mix", "random"}:
            raise ValueError(
                "target_strategy must be either 'mix' or 'random', "
                f"got '{target_strategy}'."
            )
        if schedule not in {"quad", "linear"}:
            raise ValueError(
                "schedule must be either 'quad' or 'linear', "
                f"got '{schedule}'."
            )
        if baseline_model not in SUPPORTED_BASELINES:
            raise ValueError(
                f"Unsupported baseline_model='{baseline_model}'. "
                f"Supported baselines are: {sorted(SUPPORTED_BASELINES)}."
            )

        self.n_steps = n_steps
        self.target_strategy = target_strategy
        self.baseline_model = baseline_model
        self.baseline_config_name = baseline_config_name

        baseline_cfg = _load_baseline_config(
            dataset=dataset,
            baseline_model=baseline_model,
            config_name=baseline_config_name,
        )
        baseline = self._build_baseline(baseline_model, baseline_cfg)

        checkpoint_path = (
            _project_root()
            / "lab"
            / "results"
            / "imputation_pypots"
            / dataset
            / baseline_model
            / baseline_config_name
            / f"model_used_for_testing_seed_{seed}.pth"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{baseline_model} checkpoint not found at {checkpoint_path}."
            )

        map_location = (
            device if isinstance(device, (str, torch.device)) else "cpu"
        )
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = _checkpoint_state_dict(checkpoint)

        # T1 checkpoints may include wrapper-only keys; ImputeFormer should
        # match its core architecture exactly.
        strict = baseline_model == "ImputeFormer"
        baseline.load_state_dict(state_dict, strict=strict)
        baseline.eval()
        for parameter in baseline.parameters():
            parameter.requires_grad = False

        self.model = _RDDMPI(
            n_features=n_features,
            n_layers=n_layers,
            n_heads=n_heads,
            n_channels=n_channels,
            d_time_embedding=d_time_embedding,
            d_feature_embedding=d_feature_embedding,
            d_diffusion_embedding=d_diffusion_embedding,
            n_diffusion_steps=n_diffusion_steps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            model=baseline,
            baseline_model=baseline_model,
        )

        # RDDMPI's ImputeFormer integration uses the latent width already
        # exposed by the frozen baseline. Materializing this projection here
        # avoids an uninitialized LazyLinear during PyPOTS parameter counting,
        # without changing ImputeFormer itself.
        if baseline_model == "ImputeFormer":
            latent_width = int(getattr(baseline, "model_dim", 0))
            if latent_width <= 0:
                raise ValueError(
                    "ImputeFormer baseline does not expose a valid model_dim "
                    "for RDDMPI latent conditioning."
                )
            self.model.backbone.diff_model.pre2film = torch.nn.Linear(
                latent_width, 2 * n_channels
            )

        self._print_model_size()
        self._send_model_to_given_device()

        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()
            if not isinstance(self.optimizer, Optimizer):
                raise TypeError("optimizer must be a PyPOTS Optimizer or optimizer class.")
        self.optimizer.init_optimizer(self.model.parameters())

    @staticmethod
    def _build_baseline(baseline_model, cfg):
        if baseline_model == "T1":
            from ...nn.modules.t1 import BackboneT1Imputation

            return BackboneT1Imputation(cfg)

        from ...imputation.imputeformer.core import _ImputeFormer

        return _ImputeFormer(
            n_steps=cfg.seq_len,
            n_features=cfg.enc_in,
            n_layers=cfg.model_params.n_layers,
            d_input_embed=cfg.model_params.d_input_embed,
            d_learnable_embed=cfg.model_params.d_learnable_embed,
            d_proj=cfg.model_params.d_proj,
            d_ffn=cfg.model_params.d_ffn,
            n_temporal_heads=cfg.model_params.n_temporal_heads,
            dropout=cfg.model_params.dropout,
            input_dim=cfg.model_params.input_dim,
            output_dim=cfg.model_params.output_dim,
            ORT_weight=cfg.ORT_weight,
            MIT_weight=cfg.MIT_weight,
            training_loss=None,
            validation_metric=None,
        )

    def _assemble_input_for_training(self, data: list) -> dict:
        (
            _,
            X_ori,
            indicating_mask,
            cond_mask,
            observed_tp,
        ) = self._send_data_to_given_device(data)
        return {
            "X_ori": X_ori.permute(0, 2, 1),
            "indicating_mask": indicating_mask.permute(0, 2, 1),
            "cond_mask": cond_mask.permute(0, 2, 1),
            "observed_tp": observed_tp,
        }

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        _, X, cond_mask, observed_tp = self._send_data_to_given_device(data)
        return {
            "X": X.permute(0, 2, 1),
            "cond_mask": cond_mask.permute(0, 2, 1),
            "observed_tp": observed_tp,
        }

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
        n_sampling_times: int = 1,
    ) -> None:
        train_dataset = DatasetForCSDI(
            train_set,
            self.target_strategy,
            return_X_ori=False,
            file_type=file_type,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        val_dataloader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_dataset = DatasetForCSDI(
                val_set,
                self.target_strategy,
                return_X_ori=True,
                file_type=file_type,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        self._train_model(train_dataloader, val_dataloader)
        self.model.load_state_dict(self.best_model_dict)
        self._auto_save_model_if_necessary(
            confirm_saving=self.model_saving_strategy == "best"
        )

    @torch.no_grad()
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        n_sampling_times: int = 1,
    ) -> dict:
        """Generate probabilistic imputations for a partially observed dataset."""
        if n_sampling_times <= 0:
            raise ValueError("n_sampling_times must be greater than 0.")

        self.model.eval()
        test_dataset = TestDatasetForCSDI(
            test_set, return_X_ori=False, file_type=file_type
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        result_collector = []
        for data in test_dataloader:
            inputs = self._assemble_input_for_testing(data)
            result_collector.append(
                self.model(inputs, n_sampling_times=n_sampling_times)
            )
        return gather_listed_dicts(result_collector)

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        n_sampling_times: int = 100,
    ) -> np.ndarray:
        """Return multiple RDDMPI samples for each input series."""
        if n_sampling_times <= 0:
            raise ValueError("n_sampling_times must be greater than 0.")
        return super().impute(
            test_set,
            file_type,
            n_sampling_times=n_sampling_times,
        )
