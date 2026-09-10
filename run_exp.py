import os
import sys
import torch
import torch.backends
import random
import numpy as np
import hydra
import traceback
from omegaconf import DictConfig, OmegaConf
from omegaconf import open_dict
from hydra.core.hydra_config import HydraConfig
from datetime import datetime

# Import experiment definitions and utils
from exp.exp_imputation_pypots import Exp_Imputation_PyPOTS
from utils.print_args import print_args

# Disable output buffering
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def print_separator(char="=", length=80):
    """Print separator line"""
    print(char * length)

def print_section_header(title, iteration=None, total_iterations=None):
    """Print section header"""
    print_separator("=", 80)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if iteration is not None and total_iterations is not None:
        print(f"[{timestamp}] {title} - Iteration {iteration+1}/{total_iterations}")
    else:
        print(f"[{timestamp}] {title}")
    print_separator("=", 80)

def print_subsection_header(title):
    """Print subsection header"""
    print_separator("-", 60)
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {title}")
    print_separator("-", 60)

def print_error_section(error_msg, iteration=None):
    """Print error section"""
    print_separator("!", 80)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if iteration is not None:
        print(f"[{timestamp}] ERROR in Iteration {iteration+1}: {error_msg}")
    else:
        print(f"[{timestamp}] ERROR: {error_msg}")
    print_separator("!", 80)

def print_completion_status(success, total_iterations, completed_iterations):
    """Print completion status"""
    print_separator("*", 80)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SUCCESS" if success else "FAILED"
    print(f"[{timestamp}] EXPERIMENT {status}")
    print(f"Completed iterations: {completed_iterations}/{total_iterations}")
    print_separator("*", 80)

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU support
    
    # Ensure deterministic behavior (may cause slight performance degradation)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_experiment_class(task_name):
    """Return experiment class based on task name"""
    exp_classes = {
        'imputation_pypots': Exp_Imputation_PyPOTS
    }
    
    exp_class = exp_classes.get(task_name, Exp_Imputation_PyPOTS)
    print(f"✓ Using experiment class for task: {task_name}")
    return exp_class


@hydra.main(config_path="lab/configs", config_name="default_temp")
def main(cfg: DictConfig):
    success = True  # track overall success
    completed_iterations = 0
    
    try:
        # Set working directory to original root
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        os.chdir(ROOT_DIR)
        
        # Set base seed (from config or default value)
        base_seed = cfg.get('seed', 2021)

        print_section_header("EXPERIMENT INITIALIZATION")
        print("Configuration:")
        print_args(cfg)
        
        # Update cfg.root_path by joining cfg.root_base and cfg.root_path
        with open_dict(cfg):
            if "root_base" not in cfg:
                cfg.root_base = str(ROOT_DIR / "dataset")
            cfg.root_path = os.path.join(cfg.root_base, cfg.root_path)
            
            # Also update csdi_root if it exists or if we're using PM25 data
            if cfg.data == 'pm25' or 'csdi_root' in cfg:
                cfg.csdi_root = os.path.join(cfg.root_base, 'pm25')

        # Get current config from hydra's config path
        current_path = HydraConfig.get().runtime.config_sources[2].path
        
        # Extract the path after 'configs/'
        configs_index = current_path.find('configs/')
        current_config = current_path[configs_index + len('configs/'):] + '/' + HydraConfig.get().job.config_name
        
        print(f"\nExperiment Config: {current_config}")
        print(f"Base seed: {base_seed}")
        print(f"Total iterations planned: {cfg.itr}")
        
        # Print seeds for each iteration (for reproducibility verification)
        if cfg.itr > 1:
            print("\nPlanned seeds for each iteration:")
            for i in range(cfg.itr):
                iter_seed = base_seed + i * 100
                print(f"  Iteration {i+1}: seed={iter_seed}")
        
        print_separator("=", 80)
            
        # Validate gradient accumulation settings
        if cfg.get("batch_size") is not None and cfg.get("accumulate_gradient_batches") is not None:
            if cfg.batch_size % cfg.accumulate_gradient_batches != 0:
                raise ValueError("Error: batch_size must be divisible by accumulate_gradient_batches")

        # Set device (with open_dict to allow modifications)
        with open_dict(cfg):
            if torch.cuda.is_available() and cfg.get("use_gpu", False):
                cuda_count = torch.cuda.device_count()
                if cfg.gpu >= cuda_count:
                    print(f'GPU index {cfg.gpu} is out of range. Using GPU index 0 instead. Available GPUs: {cuda_count}')
                    cfg.gpu = 0
                cfg.device = f'cuda:{cfg.gpu}'
                print(f'Using GPU: {cfg.device}')
            else:
                cfg.device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
                print(f'Using device: {cfg.device}')

        # Multi-GPU support
        if cfg.get("use_gpu", False) and cfg.get("use_multi_gpu", False):
            cfg.devices = cfg.devices.replace(' ', '')
            cfg.device_ids = [int(id_) for id_ in cfg.devices.split(',')]
            cfg.gpu = cfg.device_ids[0]

        # Select experiment class based on task
        Exp = get_experiment_class(cfg.task_name)

        # Training or testing mode
        if cfg.get("is_training", False):
            for ii in range(cfg.itr):
                # Set different (but reproducible) seed for each iteration
                iteration_seed = base_seed + ii * 100
                set_seed(iteration_seed)
                
                print_section_header(
                    f"TRAINING - {current_config} (seed={iteration_seed})", 
                    ii, 
                    cfg.itr
                )
                cfg.seed = iteration_seed

                exp = Exp(cfg)
                try:
                    # Pass iteration for imputation_pypots task
                    if cfg.task_name == 'imputation_pypots':
                        exp.train(current_config, iteration=ii)
                    else:
                        exp.train(current_config)
                    print_subsection_header(f"Training completed successfully for iteration {ii+1}")
                except Exception as e:
                    print_error_section(f"Train failed: {e}", ii)
                    traceback.print_exc()  # Print full traceback
                    success = False
                    break  # Skip testing and further iterations on error

                # Only continue with testing if training succeeded
                if success:
                    print_section_header(
                        f"TESTING - {current_config} (seed={iteration_seed})", 
                        ii, 
                        cfg.itr
                    )
                    try:
                        exp.test(current_config, iteration=ii)
                        print_subsection_header(f"Testing completed successfully for iteration {ii+1}")
                        completed_iterations += 1
                    except Exception as e:
                        print_error_section(f"Test failed: {e}", ii)
                        traceback.print_exc()
                        success = False
                        break  # Skip further iterations
                    
                    # Clear GPU cache
                    if cfg.get("gpu_type") == 'mps':
                        torch.backends.mps.empty_cache()
                    elif cfg.get("gpu_type") == 'cuda':
                        torch.cuda.empty_cache()
                        
                    print_subsection_header(f"Iteration {ii+1} completed - GPU cache cleared")
        else:
            # Test-only mode: use single seed
            set_seed(base_seed)

            cfg.seed = base_seed
            exp = Exp(cfg)
            print_section_header(f"TESTING ONLY - {current_config} (seed={base_seed})")
            try:
                # In test-only mode, we still run with iteration=0 for consistency
                exp.test(current_config, test=True, iteration=0)
                print_subsection_header("Testing completed successfully")
                completed_iterations = 1
            except Exception as e:
                print_error_section(f"Test failed: {e}")
                traceback.print_exc()
                success = False
            
            # Clear GPU cache
            if cfg.get("gpu_type") == 'mps':
                torch.backends.mps.empty_cache()
            elif cfg.get("gpu_type") == 'cuda':
                torch.cuda.empty_cache()
                
        # Print final completion status
        if cfg.get("is_training", False):
            print_completion_status(success, cfg.itr, completed_iterations)
        else:
            print_completion_status(success, 1, completed_iterations)
            
    except Exception as e:
        # This will now only catch exceptions not caught by inner handlers
        print_error_section(f"Unhandled exception: {e}")
        traceback.print_exc()
        success = False
    finally:
        if not success:
            # Force a non-zero exit code even if no uncaught exception
            sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print_error_section(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)  # Ensure non-zero exit on error