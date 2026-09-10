"""
The implementation of T1 for the partially-observed time-series imputation task.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import torch
from torch.utils.data import DataLoader

from .core import _T1
from ..saits.data import DatasetForSAITS
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...nn.modules.loss import Criterion, MAE, MSE, ORTMITLoss
from ...optim.adam import Adam
from ...optim.base import Optimizer
from .trainer import T1Trainer

# Import MetricsAggregator for result.json saving
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.training_utils import MetricsAggregator
    METRICS_AGGREGATOR_AVAILABLE = True
except ImportError:
    print("Warning: MetricsAggregator not available. Result.json will not be saved.")
    METRICS_AGGREGATOR_AVAILABLE = False


class T1(BaseNNImputer):
    """The PyTOTS implementation of the T1 model.
    T1 is a powerful transformer-based model for time-series imputation with mask-aware embedding support.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    ORT_weight :
        The weight for the ORT loss, the same as SAITS.

    MIT_weight :
        The weight for the MIT loss, the same as SAITS.

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    training_loss:
        The customized loss function designed by users for training the model.
        If not given, will use the default loss as claimed in the original paper.

    validation_metric:
        The customized metric function designed by users for validating the model.
        If not given, will use the default MSE metric.

    optimizer :
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers :
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device :
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path :
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy :
        The strategy to save model checkpoints. It has to be one of [None, "best", "better", "all"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.
        The "all" strategy will save every model after each epoch training.

    verbose :
        Whether to print out the training logs during the training process.
        
    **kwargs :
        Additional parameters for the T1 model. These will be passed directly to the T1 cfg.
        This allows complete flexibility in configuring the model through YAML files.
    """

    def __init__(
        self,
        # PyPOTS required parameters
        n_steps: int,
        n_features: int,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
        
        # Loss related
        ORT_weight: float = 1,
        MIT_weight: float = 1,
        mit_rate: float = 0.2,
        base_loss: Union[Criterion, type] = MSE,
        training_loss: Union[Criterion, type] = ORTMITLoss,
        validation_metric: Union[Criterion, type] = MSE,
        optimizer: Union[Optimizer, type] = Adam,
        
        # Receive all remaining parameters
        **kwargs
    ):
        # Create ORTMITLoss instance
        if training_loss == ORTMITLoss:
            loss_func = base_loss() if base_loss else MSE()
            self.training_loss = ORTMITLoss(MIT_weight=MIT_weight, ORT_weight=ORT_weight, loss_func=loss_func)
        else:
            self.training_loss = training_loss
        
        super().__init__(
            training_loss=self.training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            enable_amp=False,  # Will be replaced by precision in T1Trainer
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )
        
        # PyPOTS required parameters 저장
        self.n_steps = n_steps
        self.n_features = n_features
        self.mit_rate = mit_rate
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        
        # T1 trainer specific parameters
        self.lradj = kwargs.get('lradj', 'type3')
        self.gradient_clip_val = kwargs.get('gradient_clip_val', 1.0)
        self.precision = kwargs.get('precision', 32)  # 32, 16, or 'bf16'
        self.lr = kwargs.get('learning_rate', 0.001)
        self.seed = kwargs.get('seed', None)
        
        
        # Create full cfg for T1 (all additional parameters in kwargs)
        cfg_dict = {
            'seq_len': n_steps,
            'enc_in': n_features,
            'task_name': 'imputation',
            **kwargs  # All parameters from YAML
        }
        
        # Initialize model
        self.model = _T1(cfg_dict, self.training_loss, self.validation_metric)
        self._send_model_to_given_device()
        self._print_model_size()
        
        # Initialize optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer(lr=self.lr)
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())
    
    def _assemble_input_for_training(self, data: list) -> dict:
        (
            indices,
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        indices, X, missing_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }

        return inputs

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
        iteration: int = 0,  # Add iteration parameter
        setting: Optional[str] = None,  # Add setting parameter for result saving
    ) -> None:
        # Use T1-specific trainer for enhanced stability
        trainer = T1Trainer(
            model=self,
            optimizer=self.optimizer,
            device=self.device,
            epochs=self.epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            mit_rate=self.mit_rate,
            learning_rate=self.lr,
            lradj=self.lradj,
            precision=self.precision,
            gradient_clip_val=self.gradient_clip_val,
            verbose=self.verbose,
            saving_path=self.saving_path,
            model_saving_strategy=self.model_saving_strategy,
            seed=self.seed,
        )
        
        # Train the model using the custom trainer
        trainer.fit(train_set, val_set, file_type, iteration)
        
        # Store model profile for later use
        self._model_profile = trainer.get_model_profile()
    
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        """Make predictions with the trained model using the custom trainer.

        Parameters
        ----------
        test_set :
            The dataset for model testing, should be a dictionary including keys as 'X' and 'missing_mask',
            or a path string locating a data file.
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            missing_mask should also be array-like with the same shape as X,
            indicating the missing positions in X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'missing_mask'.

        file_type :
            The type of the given file if test_set is a path string.

        Returns
        -------
        results :
            A dictionary containing 'imputation' as the imputed data.
        """
        # Use T1-specific trainer for prediction
        trainer = T1Trainer(
            model=self,
            optimizer=self.optimizer,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            mit_rate=0.0,  # No masking for prediction
            precision=self.precision,
            verbose=False,  # Quiet during prediction
            seed= self.seed,
        )
        
        return trainer.predict(test_set, file_type)
    
    def get_model_profile(self) -> Optional[dict]:
        """Get the model profile from the last training session"""
        return getattr(self, '_model_profile', None)