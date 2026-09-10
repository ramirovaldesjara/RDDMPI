"""
Training utilities for mixed precision, gradient accumulation, and other common functionalities
"""
import torch
from torch.cuda.amp import GradScaler, autocast
from torch import optim
import torch.nn as nn
import os
import json
import numpy as np
import math
import time
import psutil
from omegaconf import OmegaConf


class TimeProfiler:
    """Class to manage execution time measurement"""
    
    def __init__(self):
        self.iter_times = []
        self.epoch_times = []
        self.current_epoch_start = None
        self.current_iter_start = None
        
    def start_epoch(self):
        """Record epoch start time"""
        self.current_epoch_start = time.time()
        
    def end_epoch(self):
        """Record epoch end time and save duration"""
        if self.current_epoch_start is not None:
            epoch_time = time.time() - self.current_epoch_start
            self.epoch_times.append(epoch_time)
            self.current_epoch_start = None
            return epoch_time
        return None
        
    def start_iter(self):
        """Record iteration start time"""
        self.current_iter_start = time.time()
        
    def end_iter(self):
        """Record iteration end time and save duration"""
        if self.current_iter_start is not None:
            iter_time = time.time() - self.current_iter_start
            self.iter_times.append(iter_time)
            self.current_iter_start = None
            return iter_time
        return None
        
    def get_statistics(self):
        """Return average time statistics"""
        stats = {}
        
        if self.iter_times:
            # Exclude first few iterations as warm-up (10% of total or at least 5)
            warmup_count = max(5, len(self.iter_times) // 10)
            stable_iter_times = self.iter_times[warmup_count:]
            
            if stable_iter_times:
                avg_iter_time_s = np.mean(stable_iter_times)
                stats['ms_per_iter'] = round(avg_iter_time_s * 1000, 2)  # Convert to milliseconds
            else:
                # If no data after excluding warm-up, use overall average
                avg_iter_time_s = np.mean(self.iter_times)
                stats['ms_per_iter'] = round(avg_iter_time_s * 1000, 2)
        else:
            stats['ms_per_iter'] = None
            
        if self.epoch_times:
            avg_epoch_time_s = np.mean(self.epoch_times)
            stats['s_per_epoch'] = round(avg_epoch_time_s, 2)  # In seconds
        else:
            stats['s_per_epoch'] = None
            
        return stats
        
    def reset(self):
        """Reset statistics"""
        self.iter_times = []
        self.epoch_times = []
        self.current_epoch_start = None
        self.current_iter_start = None


class PrecisionManager:
    """Class to manage Mixed Precision settings"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.precision = getattr(args, 'precision', 32)
        
        # Initialize precision settings
        self._setup_precision()

    def _setup_precision(self):
        """Initialize precision settings"""
        if self.precision == 32:
            self.use_amp = False
            self.amp_dtype = None
            self.scaler = None
            print("Using FP32 precision (no mixed precision)")
        elif self.precision == 16:
            if self.device.type != 'cuda':
                print("Warning: FP16 requires CUDA. Falling back to FP32.")
                self.use_amp = False
                self.amp_dtype = None
                self.scaler = None
            else:
                self.use_amp = True
                self.amp_dtype = torch.float16
                self.scaler = GradScaler()
                print("Using FP16 mixed precision")
        elif self.precision == 'bf16' or self.precision == 'bfloat16':
            if self.device.type != 'cuda':
                print("Warning: BF16 requires CUDA. Falling back to FP32.")
                self.use_amp = False
                self.amp_dtype = None
                self.scaler = None
            elif not torch.cuda.is_bf16_supported():
                print("Warning: BF16 not supported on this hardware. Falling back to FP32.")
                self.use_amp = False
                self.amp_dtype = None
                self.scaler = None
            else:
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # BF16 doesn't need gradient scaling
                print("Using BF16 mixed precision")
        else:
            print(f"Warning: Unknown precision '{self.precision}'. Using FP32.")
            self.use_amp = False
            self.amp_dtype = None
            self.scaler = None

    def autocast_context(self):
        """Return appropriate autocast context"""
        if self.use_amp:
            return autocast(dtype=self.amp_dtype)
        else:
            return autocast(enabled=False)

    def scale_loss_and_backward(self, loss, accumulate_grad_batches):
        """Loss scaling and backward pass"""
        scaled_loss = loss / accumulate_grad_batches
        
        if self.use_amp and self.scaler is not None:
            # FP16: gradient scaling needed
            self.scaler.scale(scaled_loss).backward()
        else:
            # FP32 or BF16: gradient scaling not needed
            scaled_loss.backward()

    def optimizer_step(self, optimizer):
        """Optimizer step with proper scaling"""
        if self.use_amp and self.scaler is not None:
            # FP16: unscale, step, update
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=4.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # FP32 or BF16: clip, step
            # Get parameters from all param_groups of optimizer
            all_params = []
            for group in optimizer.param_groups:
                all_params.extend(group['params'])
            nn.utils.clip_grad_norm_(all_params, max_norm=4.0)
            optimizer.step()


class OptimizerFactory:
    """Factory class responsible for creating optimizers"""
    
    @staticmethod
    def create_optimizer(model, args):
        optimizer_dict = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adamax': optim.Adamax,
            'adadelta': optim.Adadelta,
            'adagrad': optim.Adagrad,
            'sparseadam': optim.SparseAdam,
            'asgd': optim.ASGD,
            'rprop': optim.Rprop,
            'lbfgs': optim.LBFGS
        }
        
        optimizer_name = getattr(args, 'optimizer', 'adam').lower()
        
        if optimizer_name not in optimizer_dict:
            print(f"Warning: Optimizer '{optimizer_name}' not found. Using Adam as default.")
            optimizer_name = 'adam'
        
        optimizer_class = optimizer_dict[optimizer_name]
        
        # Different optimizers may require different parameters
        if optimizer_name == 'sgd':
            model_optim = optimizer_class(model.parameters(), 
                                        lr=args.learning_rate,
                                        momentum=getattr(args, 'momentum', 0.9),
                                        weight_decay=getattr(args, 'weight_decay', 0))
        elif optimizer_name == 'adamw':
            model_optim = optimizer_class(model.parameters(),
                                        lr=args.learning_rate,
                                        weight_decay=getattr(args, 'weight_decay', 0.01))
        elif optimizer_name == 'rmsprop':
            model_optim = optimizer_class(model.parameters(),
                                        lr=args.learning_rate,
                                        alpha=getattr(args, 'alpha', 0.99),
                                        eps=getattr(args, 'eps', 1e-8))
        elif optimizer_name == 'lbfgs':
            model_optim = optimizer_class(model.parameters(),
                                        lr=getattr(args, 'learning_rate', 1),
                                        max_iter=getattr(args, 'max_iter', 20))
        else:
            # Default parameters for Adam, Adamax, Adadelta, Adagrad, etc.
            model_optim = optimizer_class(model.parameters(), lr=args.learning_rate)
        
        return model_optim


class GradientAccumulator:
    """Class to manage Gradient Accumulation"""
    
    def __init__(self, args):
        self.accumulate_grad_batches = getattr(args, 'accumulate_grad_batches', 1)
        
    def should_step(self, batch_idx, total_batches):
        """Decide whether to perform optimizer step"""
        return ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (batch_idx + 1 == total_batches)


class IterationManager:
    """Class to manage iteration-related logic"""
    
    @staticmethod
    def should_save_artifacts(iteration):
        """Save artifacts only on first iteration"""
        return iteration == 0
    
    @staticmethod
    def is_last_iteration(iteration, total_iterations):
        """Check if this is the last iteration"""
        return iteration + 1 == total_iterations
    
    @staticmethod
    def create_result_directory(setting, iteration=0):
        """Create result save directory"""
        folder_path = os.path.join('./lab/results', setting)
        os.makedirs(folder_path, exist_ok=True)
        
        # Create samples folder only on first iteration
        if iteration == 0:
            samples_folder = os.path.join(folder_path, 'samples')
            os.makedirs(samples_folder, exist_ok=True)
            return folder_path, samples_folder
        
        return folder_path, None


class MetricsAggregator:
    """Class to aggregate metrics from multiple iterations"""
    
    @staticmethod
    def calculate_statistics(metrics_list, metric_names):
        """Calculate mean and confidence interval of metrics"""
        n = len(metrics_list)
        if n == 0:
            return {}
        
        result = {}
        
        # Calculate mean
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list]
            result[metric_name] = sum(values) / n
            
            # Calculate confidence interval (only when n > 1)
            if n > 1:
                std = np.std(values, ddof=1)
                ci_95 = 1.96 * std / math.sqrt(n)
                ci_99 = 2.58 * std / math.sqrt(n)
                result[f"{metric_name}_ci_95"] = ci_95
                result[f"{metric_name}_ci_99"] = ci_99
            else:
                result[f"{metric_name}_ci_95"] = 0.0
                result[f"{metric_name}_ci_99"] = 0.0
        
        return result
    
    @staticmethod
    def save_final_results(setting, metrics_stats, args, model_profile=None):
        """Save final results to JSON file (including model profiling info)"""
        folder_path = os.path.join('./lab/results', setting)
        os.makedirs(folder_path, exist_ok=True)
        
        cfg_dict = OmegaConf.to_container(args, resolve=True)
        result_data = {
            "setting": setting,
            "metrics": metrics_stats,
            "config": cfg_dict
        }
        
        # Add model profiling information
        if model_profile is not None:
            result_data["model_profile"] = model_profile
        
        result_file = os.path.join(folder_path, 'result.json')
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        
        print(f"Final results saved in: {result_file}")
        if model_profile:
            print("Model profiling information included in results")


class CheckpointManager:
    """Class to manage checkpoint save/load"""
    
    @staticmethod
    def get_checkpoint_path(args, setting):
        """Return checkpoint path"""
        return os.path.join(args.checkpoints, setting)
    
    @staticmethod
    def get_checkpoint_file_path(args, setting):
        """Return checkpoint file path"""
        return os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    
    @staticmethod
    def load_best_model(model, args, setting):
        """Load best model"""
        checkpoint_path = CheckpointManager.get_checkpoint_file_path(args, setting)
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded best model from: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        return model


class ModelProfiler:
    """Class to measure model parameters, operations, and memory usage"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.profile_results = {}
        
    def count_parameters(self):
        """Calculate model parameter count"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.profile_results.update({
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'total_parameters_M': round(total_params / 1e6, 2),
            'trainable_parameters_M': round(trainable_params / 1e6, 2)
        })
        
        print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        return total_params, trainable_params
    
    def measure_flops(self, input_shape, num_samples=1):
        """Measure model FLOPs (estimation)"""
        try:
            # Create dummy input based on the model type
            if len(input_shape) == 3:  # (batch, seq, features)
                dummy_input = torch.randn(num_samples, *input_shape[1:]).to(self.device)
                dummy_marks = torch.randn(num_samples, *input_shape[1:]).to(self.device)
                dummy_dec_inp = torch.randn(num_samples, *input_shape[1:]).to(self.device)
            else:
                dummy_input = torch.randn(num_samples, *input_shape).to(self.device)
                dummy_marks = None
                dummy_dec_inp = None
            
            # Count operations using hooks
            self.flops_count = 0
            hooks = []
            
            def flop_count_hook(module, input, output):
                if isinstance(module, nn.Linear):
                    # Linear layer: input_size * output_size * batch_size
                    input_size = input[0].size(-1)
                    output_size = output.size(-1)
                    batch_size = output.numel() // output_size
                    self.flops_count += input_size * output_size * batch_size
                elif isinstance(module, nn.Conv1d):
                    # Conv1d: kernel_size * in_channels * out_channels * output_length * batch_size
                    kernel_size = module.kernel_size[0]
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    output_length = output.size(-1)
                    batch_size = output.size(0)
                    self.flops_count += kernel_size * in_channels * out_channels * output_length * batch_size
                elif isinstance(module, nn.MultiheadAttention):
                    # Attention: rough estimation
                    seq_len = input[0].size(1) if len(input[0].shape) >= 2 else 1
                    embed_dim = input[0].size(-1)
                    batch_size = input[0].size(0)
                    # Q*K^T + softmax + attention*V (simplified)
                    self.flops_count += 2 * batch_size * seq_len * seq_len * embed_dim
            
            # Register hooks
            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.MultiheadAttention)):
                    hooks.append(module.register_forward_hook(flop_count_hook))
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                # Check if model needs mask for imputation task
                needs_mask = (hasattr(self.model, 'task') and 
                              self.model.task == 'imputation' and
                              hasattr(self.model, 'cfg') and
                              hasattr(self.model.cfg, 'imputation_use_mask_embedding') and
                              self.model.cfg.imputation_use_mask_embedding)
                
                if dummy_marks is not None and dummy_dec_inp is not None:
                    # For time series models
                    try:
                        if needs_mask:
                            dummy_mask = torch.ones(num_samples, *input_shape[1:]).to(self.device)
                            _ = self.model(dummy_input, dummy_marks, dummy_dec_inp, dummy_marks, dummy_mask)
                        else:
                            _ = self.model(dummy_input, dummy_marks, dummy_dec_inp, dummy_marks)
                    except:
                        try:
                            if needs_mask:
                                dummy_mask = torch.ones(num_samples, *input_shape[1:]).to(self.device)
                                _ = self.model(dummy_input, None, dummy_dec_inp, None, dummy_mask)
                            else:
                                _ = self.model(dummy_input, None, dummy_dec_inp, None)
                        except:
                            _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            flops_per_sample = self.flops_count / num_samples
            gflops_per_sample = flops_per_sample / 1e9
            
            self.profile_results.update({
                'flops': int(flops_per_sample),
                'gflops': round(gflops_per_sample, 3)
            })
            
            print(f"FLOPs: {flops_per_sample:,} ({gflops_per_sample:.3f} GFLOPs)")
            
            return flops_per_sample
            
        except Exception as e:
            print(f"Warning: Could not measure FLOPs: {e}")
            self.profile_results.update({
                'flops': None,
                'gflops': None
            })
            return None
    
    def measure_memory_footprint(self, input_shape, num_samples=1):
        """Measure model memory usage"""
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Measure initial memory
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated(self.device)
                initial_memory_mb = initial_memory / 1024 / 1024
            else:
                process = psutil.Process()
                initial_memory = process.memory_info().rss
                initial_memory_mb = initial_memory / 1024 / 1024
            
            # Create dummy input
            if len(input_shape) == 3:  # (batch, seq, features)
                dummy_input = torch.randn(num_samples, *input_shape[1:]).to(self.device)
                dummy_marks = torch.randn(num_samples, *input_shape[1:]).to(self.device)
                dummy_dec_inp = torch.randn(num_samples, *input_shape[1:]).to(self.device)
            else:
                dummy_input = torch.randn(num_samples, *input_shape).to(self.device)
                dummy_marks = None
                dummy_dec_inp = None
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                # Check if model needs mask for imputation task
                needs_mask = (hasattr(self.model, 'task') and 
                              self.model.task == 'imputation' and
                              hasattr(self.model, 'cfg') and
                              hasattr(self.model.cfg, 'imputation_use_mask_embedding') and
                              self.model.cfg.imputation_use_mask_embedding)
                
                if dummy_marks is not None and dummy_dec_inp is not None:
                    try:
                        if needs_mask:
                            dummy_mask = torch.ones(num_samples, *input_shape[1:]).to(self.device)
                            output = self.model(dummy_input, dummy_marks, dummy_dec_inp, dummy_marks, dummy_mask)
                        else:
                            output = self.model(dummy_input, dummy_marks, dummy_dec_inp, dummy_marks)
                    except:
                        try:
                            if needs_mask:
                                dummy_mask = torch.ones(num_samples, *input_shape[1:]).to(self.device)
                                output = self.model(dummy_input, None, dummy_dec_inp, None, dummy_mask)
                            else:
                                output = self.model(dummy_input, None, dummy_dec_inp, None)
                        except:
                            output = self.model(dummy_input)
                else:
                    output = self.model(dummy_input)
            
            # Measure peak memory
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated(self.device)
                peak_memory_mb = peak_memory / 1024 / 1024
                model_memory_mb = (peak_memory - initial_memory) / 1024 / 1024
                torch.cuda.reset_peak_memory_stats(self.device)
            else:
                process = psutil.Process()
                peak_memory = process.memory_info().rss
                peak_memory_mb = peak_memory / 1024 / 1024
                model_memory_mb = peak_memory_mb - initial_memory_mb
            
            self.profile_results.update({
                'peak_memory_mb': round(peak_memory_mb, 2),
                'model_memory_mb': round(max(0, model_memory_mb), 2),
                'initial_memory_mb': round(initial_memory_mb, 2)
            })
            
            print(f"Peak Memory: {peak_memory_mb:.2f} MB")
            print(f"Model Memory: {max(0, model_memory_mb):.2f} MB")
            
            return peak_memory_mb, max(0, model_memory_mb)
            
        except Exception as e:
            print(f"Warning: Could not measure memory: {e}")
            self.profile_results.update({
                'peak_memory_mb': None,
                'model_memory_mb': None,
                'initial_memory_mb': None
            })
            return None, None
    
    def profile_model(self, input_shape, num_samples=1):
        """Perform all model profiling"""
        print("=" * 50)
        print("MODEL PROFILING")
        print("=" * 50)
        
        # Count parameters
        self.count_parameters()
        
        # Measure FLOPs
        self.measure_flops(input_shape, num_samples)
        
        # Measure memory
        self.measure_memory_footprint(input_shape, num_samples)
        
        print("=" * 50)
        
        return self.profile_results
    
    def get_results(self):
        """Return profiling results"""
        return self.profile_results.copy()


class DataLoaderManager:
    """DataLoader related utilities"""
    
    @staticmethod
    def adjust_batch_size_for_accumulation(args, data_provider_func, flag):
        """Adjust batch_size for gradient accumulation"""
        accumulate_grad_batches = getattr(args, 'accumulate_grad_batches', 1)
        
        # Temporarily divide batch_size by gradient accumulation count
        original_batch_size = args.batch_size
        args.batch_size = original_batch_size // accumulate_grad_batches
        
        # Create DataLoader
        data_set, data_loader = data_provider_func(args, flag)
        
        # Restore original batch_size after use
        args.batch_size = original_batch_size
        
        return data_set, data_loader