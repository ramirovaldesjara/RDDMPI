"""
T1 model-specific trainer with enhanced stability features.
Inspired by exp_imputation.py for better mixed precision and gradient handling.
"""

import os
import time
from random import seed

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from typing import Dict, Optional, Tuple, Union

from ..saits.data import DatasetForSAITS
from ...data.checking import key_in_data_set
from ...nn.modules.loss import ORTMITLoss
from ...nn.functional import calc_mae

# Import ModelProfiler from utils
import sys
import os
# Add parent directories to path to import from utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.training_utils import ModelProfiler, PrecisionManager
    PROFILER_AVAILABLE = True
except ImportError:
    print("Warning: ModelProfiler/PrecisionManager not available. Some features will be skipped.")
    PROFILER_AVAILABLE = False
    # Dummy PrecisionManager for fallback
    class PrecisionManager:
        def __init__(self, args, device):
            self.use_amp = False
            self.scaler = None
        def autocast_context(self):
            from torch.cuda.amp import autocast
            return autocast(enabled=False)


class T1Trainer:
    """Custom trainer for T1 model with enhanced stability features"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int = 100,
        patience: int = 30,
        batch_size: int = 32,
        num_workers: int = 0,
        mit_rate: float = 0.7,
        learning_rate: float = 1e-3,
        lradj: str = "type3",
        precision: Union[int, str] = 32,  # 32, 16, or 'bf16'
        gradient_clip_val: float = 1.0,
        verbose: bool = True,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
        seed: Optional[int] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mit_rate = mit_rate
        self.learning_rate = learning_rate
        self.lradj = lradj
        self.gradient_clip_val = gradient_clip_val
        self.verbose = verbose
        self.saving_path = saving_path
        self.model_saving_strategy = model_saving_strategy
        self.seed = seed
        # Initialize loss functions
        self.training_loss = self.model.model.training_loss
        self.validation_loss = calc_mae  # Use MAE for validation
        
        # Initialize precision manager
        # Create a simple args object for PrecisionManager
        class Args:
            pass
        args = Args()
        args.precision = precision
        self.precision_manager = PrecisionManager(args, device)
        self.enable_amp = self.precision_manager.use_amp
        self.scaler = self.precision_manager.scaler
        
        # Important: Do NOT convert model to bfloat16/float16
        # Only use autocast for mixed precision (same as exp_imputation.py)
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
        # Model profiling
        self.model_profile = None
        
    def _adjust_learning_rate(self, optimizer, epoch, total_steps=None, current_step=None):
        """Adjust learning rate based on schedule"""
        if self.lradj == 'type3' or self.lradj == 'constant' or self.lradj == 'fixed':
            # Fixed learning rate - no adjustment needed
            pass
        elif self.lradj == 'type1':
            # Step decay
            lr_adjust = {epoch: self.learning_rate * (0.5 ** ((epoch - 1) // 1))}
            if epoch in lr_adjust.keys():
                lr = lr_adjust[epoch]
                for param_group in optimizer.torch_optimizer.param_groups:
                    param_group['lr'] = lr
        elif self.lradj == 'onecycle':
            # OneCycleLR-like schedule
            if current_step is not None and total_steps is not None:
                progress = current_step / total_steps
                if progress < 0.5:
                    # Warmup phase
                    lr = self.learning_rate * (2 * progress)
                else:
                    # Cosine annealing phase
                    lr = self.learning_rate * (1 + np.cos(np.pi * (progress - 0.5))) / 2
                    
                for param_group in optimizer.torch_optimizer.param_groups:
                    param_group['lr'] = lr
        else:
            # Default: Cosine annealing
            lr = self.learning_rate * (1 + np.cos(np.pi * epoch / self.epochs)) / 2
            for param_group in optimizer.torch_optimizer.param_groups:
                param_group['lr'] = lr
                
    def _check_nan(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check if tensor contains NaN values"""
        if torch.isnan(tensor).any():
            if self.verbose:
                print(f"Warning: NaN detected in {name}")
            return True
        return False
        
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int,
        total_steps: int,
    ) -> float:
        """Train for one epoch with enhanced stability"""
        self.model.model.train()
        train_losses = []
        
        for batch_idx, data in enumerate(train_loader):

            current_step = epoch * len(train_loader) + batch_idx
            
            # Adjust learning rate
            self._adjust_learning_rate(self.optimizer, epoch, total_steps, current_step)
            
            # Prepare inputs
            indices, X, missing_mask, X_ori, indicating_mask = data
            
            # Move to device
            X = X.to(self.device)
            missing_mask = missing_mask.to(self.device)
            X_ori = X_ori.to(self.device) if X_ori is not None else X
            indicating_mask = indicating_mask.to(self.device)
            
            # Check for NaN in inputs
            if self._check_nan(X, "input X"):
                print(f"Skipping batch {batch_idx} due to NaN in input")
                continue
                
            inputs = {
                "X": X,
                "missing_mask": missing_mask,
                "X_ori": X_ori,
                "indicating_mask": indicating_mask,
            }

            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.enable_amp and self.scaler is not None:
                with self.precision_manager.autocast_context():
                    outputs = self.model.model(inputs, calc_criterion=True)
                    loss = outputs["loss"]
                    
                    # Check for NaN in loss
                    if self._check_nan(loss, "loss"):
                        print(f"Skipping batch {batch_idx} due to NaN in loss")
                        continue
                        
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping and optimizer step
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(), 
                        self.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                outputs = self.model.model(inputs, calc_criterion=True)
                loss = outputs["loss"]
                
                # Check for NaN in loss
                if self._check_nan(loss, "loss"):
                    print(f"Skipping batch {batch_idx} due to NaN in loss")
                    continue
                    
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.gradient_clip_val
                    )
                    
                self.optimizer.step()
                
            train_losses.append(loss.item())
                
        mean_train_loss = np.mean(train_losses) if train_losses else float('inf')
        return mean_train_loss
        
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model with enhanced stability"""
        self.model.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                # Prepare inputs
                indices, X, missing_mask, X_ori, indicating_mask = data
                
                # Move to device
                X = X.to(self.device)
                missing_mask = missing_mask.to(self.device)
                X_ori = X_ori.to(self.device) if X_ori is not None else X
                indicating_mask = indicating_mask.to(self.device)
                
                inputs = {
                    "X": X,
                    "missing_mask": missing_mask,
                    "X_ori": X_ori,
                    "indicating_mask": indicating_mask,
                }
                
                # Forward pass
                if self.enable_amp:
                    with self.precision_manager.autocast_context():
                        outputs = self.model.model(inputs, calc_criterion=True)
                        if "metric" in outputs:
                            val_loss = outputs["metric"]
                        else:
                            # Fallback to MAE calculation
                            reconstruction = outputs["reconstruction"]
                            val_loss = self.validation_loss(
                                reconstruction, X_ori, indicating_mask
                            )
                else:
                    outputs = self.model.model(inputs, calc_criterion=True)
                    if "metric" in outputs:
                        val_loss = outputs["metric"]
                    else:
                        # Fallback to MAE calculation
                        reconstruction = outputs["reconstruction"]
                        val_loss = self.validation_loss(
                            reconstruction, X_ori, indicating_mask
                        )
                        
                if not self._check_nan(val_loss, "validation loss"):
                    val_losses.append(val_loss.item())
                    
        mean_val_loss = np.mean(val_losses) if val_losses else float('inf')
        return mean_val_loss
        
    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
        iteration: int = 0,  # Add iteration parameter for profiling
    ) -> None:
        """Train the model with enhanced stability features"""
        # Create data loaders
        train_dataset = DatasetForSAITS(
            train_set,
            return_X_ori=False,
            return_y=False,
            file_type=file_type,
            rate=self.mit_rate
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,  # Drop last incomplete batch
        )
        
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_dataset = DatasetForSAITS(
                val_set,
                return_X_ori=True,
                return_y=False,
                file_type=file_type,
                rate=self.mit_rate
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            
        # Profile model on first iteration
        if iteration == 0 and self.model_profile is None and PROFILER_AVAILABLE:
            try:
                # Get input shape from dataset
                if isinstance(train_set, dict) and 'X' in train_set:
                    input_shape = train_set['X'].shape  # (n_samples, n_steps, n_features)
                else:
                    # Get from first batch
                    sample_data = next(iter(train_loader))
                    _, X, _, _, _ = sample_data
                    input_shape = (len(train_dataset), X.shape[1], X.shape[2])
                
                # Create profiler and profile model
                profiler = ModelProfiler(self.model.model, self.device)
                # Silently count parameters (suppress output)
                import sys
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                profiler.count_parameters()
                sys.stdout = old_stdout
                self.model_profile = profiler.get_results()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not profile model: {e}")
                self.model_profile = None
        
        # Calculate total training steps
        total_steps = len(train_loader) * self.epochs
        
        # Clear GPU memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch, total_steps)
            
            # Validate
            val_loss = float('inf')
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                
            epoch_time = time.time() - epoch_start_time
            
            if self.verbose:
                print(f"Epoch [{epoch}/{self.epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Time: {epoch_time:.2f}s", end="")
                
            # Early stopping and best model tracking
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_state = self.model.model.state_dict().copy()
                self.patience_counter = 0
                
                # Save best model if path provided
                if self.saving_path and self.model_saving_strategy == "best":
                    self._save_model()
                    
                if self.verbose:
                    print(f" *Best*")
            else:
                if self.verbose:
                    print()  # New line for non-best epochs
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break
                    
        # Load best model
        if self.best_model_state is not None:
            self.model.model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"Loaded best model with validation loss: {self.best_loss:.4f}")
                
        # Measure actual training peak memory on first iteration
        if iteration == 0 and self.model_profile is not None and torch.cuda.is_available():
            try:
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.model_profile['training_peak_memory_mb'] = round(peak_memory_mb, 2)
                if self.verbose:
                    print(f"Actual Training Peak Memory: {peak_memory_mb:.2f} MB")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not get training peak memory: {e}")
                
    def test(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> Dict[str, float]:
        """Test the model and return metrics"""
        if not key_in_data_set("X_ori", test_set):
            raise ValueError("test_set must contain 'X_ori' for testing.")
            
        test_dataset = DatasetForSAITS(
            test_set,
            return_X_ori=True,
            return_y=False,
            file_type=file_type,
            rate=0.0  # No masking for test
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        test_loss = self.validate(test_loader)
        
        return {"mae": test_loss}
    
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> Dict[str, np.ndarray]:
        """Generate predictions for the given test set"""
        test_dataset = DatasetForSAITS(
            test_set,
            return_X_ori=False,
            return_y=False,
            file_type=file_type,
            rate=0.0,  # No additional masking for prediction
            seed= self.seed if hasattr(self, "seed") else None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        self.model.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                # Prepare inputs
                indices, X, missing_mask = data[:3]  # Only need first 3 elements
                
                # Move to device
                X = X.to(self.device)
                missing_mask = missing_mask.to(self.device)
                
                inputs = {
                    "X": X,
                    "missing_mask": missing_mask,
                }

                # Forward pass
                if self.enable_amp:
                    with self.precision_manager.autocast_context():
                        outputs = self.model.model(inputs, calc_criterion=False)
                else:
                    outputs = self.model.model(inputs, calc_criterion=False)
                    
                imputation = outputs["imputation"]
                predictions.append(imputation.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        
        return {"imputation": predictions}
    
    def get_model_profile(self) -> Optional[Dict]:
        """Get the model profile results"""
        return self.model_profile
        
    def _save_model(self):
        """Save model checkpoint"""
        if self.saving_path:
            os.makedirs(self.saving_path, exist_ok=True)
            checkpoint_path = os.path.join(self.saving_path, "best_model.pth")
            torch.save({
                'model_state_dict': self.best_model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'epoch': self.patience_counter,
            }, checkpoint_path)
            if self.verbose:
                print(f"Model saved to {checkpoint_path}")