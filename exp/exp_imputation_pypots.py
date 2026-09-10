"""
PyPOTS native integration - using BenchPOTS/TSDB for data loading
Simplified version using direct parameter passing from config
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import time
import json
from omegaconf import OmegaConf
import pickle


# Enable AMP support in PyPOTS
os.environ['ENABLE_AMP'] = 'True'

# PyPOTS ecosystem imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pypots_lib.imputation import (
    SAITS, BRITS, CSDI, ImputeFormer, ModernTCN,
    iTransformer, TimesNet, PatchTST, DLinear, TimeMixerPP, T1, RDDMPI, GPVAE, FGTI
)
from pypots_lib.nn.functional import calc_mse, calc_mae, calc_rmse
from pypots_lib.nn.modules.loss import MAE, MSE
from pypots_lib.utils.missing_patterns import MissingPatternGenerator
# from benchpots.datasets import preprocess_physionet2012, preprocess_ett, preprocess_electricity_load_diagrams, preprocess_pems_traffic
from pygrinder import mcar

# Import custom dataset loaders - now including ETT and electricity
from data_provider.pypots_custom_datasets import (
    preprocess_weather, preprocess_exchange, preprocess_illness,
    preprocess_pems03, preprocess_pems04, preprocess_ett, preprocess_electricity
)


from utils.metrics import MAPE, MSPE


from utils.training_utils import (
    IterationManager, 
    MetricsAggregator, 
    ModelProfiler,
    TimeProfiler
)

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS_noscale(target, forecast, eval_points):
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for q in quantiles:
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j:j+1], q, dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, q, eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def convert_to_serializable(obj):
    """Convert non-serializable objects to JSON-serializable format"""
    import numpy as np
    from omegaconf import DictConfig, ListConfig
    
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class Exp_Imputation_PyPOTS:
    """PyPOTS experiment using native data loading with simplified parameter handling"""
    
    # Class variables for aggregating results across iterations
    _all_metrics = []
    
    # Model class mapping
    MODEL_CLASSES = {
        'saits': SAITS,
        'brits': BRITS,
        'csdi': CSDI,
        'imputeformer': ImputeFormer,
        'moderntcn': ModernTCN,
        'itransformer': iTransformer,
        'timesnet': TimesNet,
        'patchtst': PatchTST,
        'dlinear': DLinear,
        'timemixer++': TimeMixerPP,
        'timemixerpp': TimeMixerPP,
        't1': T1,
        'rddmpi': RDDMPI,
        'gpvae': GPVAE,
        'fgti': FGTI,


    }
    
    def __init__(self, args):
        self.args = args
        self.device = args.device if hasattr(args, 'device') else 'cuda:0'
        self.model = None
        self.model_profile = None
        self.pattern_generator = MissingPatternGenerator(args.seed)
        # self.pattern_generator = MissingPatternGenerator()
        self.time_profiler = TimeProfiler()
        
        # Auto-detect dimensions for BenchPOTS and CSDI datasets
        data_source = getattr(args, 'data_source', 'native')
        
        if data_source == 'benchpots':
            benchpots_dims = {
                'physionet2012': {'enc_in': 36, 'seq_len': 48},
                'air_quality': {'enc_in': 12, 'seq_len': 24}
            }
            
            if args.data in benchpots_dims:
                # Set dimensions if not already specified
                if not hasattr(args, 'enc_in') or args.enc_in == 0:
                    args.enc_in = benchpots_dims[args.data]['enc_in']
                    args.dec_in = args.enc_in
                    args.c_out = args.enc_in
                if not hasattr(args, 'seq_len') or args.seq_len == 0:
                    args.seq_len = benchpots_dims[args.data]['seq_len']
                    
        elif data_source == 'csdi':
            csdi_dims = {
                'pm25': {'enc_in': 36, 'seq_len': 36}
            }
            
            if args.data in csdi_dims:
                # Set dimensions if not already specified
                if not hasattr(args, 'enc_in') or args.enc_in == 0:
                    args.enc_in = csdi_dims[args.data]['enc_in']
                    args.dec_in = args.enc_in
                    args.c_out = args.enc_in
                if not hasattr(args, 'seq_len') or args.seq_len == 0:
                    args.seq_len = csdi_dims[args.data]['seq_len']
        
    def _load_data_native(self):
        """Load data using PyPOTS ecosystem or BenchPOTS"""
        print(f"Loading {self.args.data} dataset...")
        
        # Check data source
        data_source = getattr(self.args, 'data_source', 'native')
        
        if data_source == 'benchpots':
            # Use BenchPOTS data loader
            self._load_benchpots_data()
            # Return the loaded data
            return self.train_data['X_ori'], self.val_data['X_ori'], self.test_data['X_ori']
            
        elif data_source == 'csdi':
            # Use CSDI data loader
            self._load_csdi_data()
            # Return the loaded data
            return self.train_data['X_ori'], self.val_data['X_ori'], self.test_data['X_ori']
        
        # Always use traditional split style
        split_style = 'traditional'
        print(f"Using data split style: {split_style}")
        

        n_steps = self.args.seq_len  # Use seq_len from config
        dataset_map = {
            'ETTh1': lambda: preprocess_ett(subset='ETTh1', rate=0.0, n_steps=n_steps,
                                           root_path=self.args.root_path,
                                           data_path=self.args.data_path,
                                           split_style=split_style,  test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            'ETTh2': lambda: preprocess_ett(subset='ETTh2', rate=0.0, n_steps=n_steps,
                                           root_path=self.args.root_path,
                                           data_path=self.args.data_path,
                                           split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            'ETTm1': lambda: preprocess_ett(subset='ETTm1', rate=0.0, n_steps=n_steps,
                                           root_path=self.args.root_path,
                                           data_path=self.args.data_path,
                                           split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            'ETTm2': lambda: preprocess_ett(subset='ETTm2', rate=0.0, n_steps=n_steps,
                                           root_path=self.args.root_path,
                                           data_path=self.args.data_path,
                                           split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            # 'physionet': lambda: preprocess_physionet2012(subset='set-a', rate=0.0),
            'electricity': lambda: preprocess_electricity(rate=0.0, n_steps=n_steps,
                                                        root_path=self.args.root_path,
                                                        data_path=self.args.data_path,
                                                        split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            'PEMS03': lambda: preprocess_pems03(rate=0.0, n_steps=n_steps,
                                               root_path=self.args.root_path,
                                               data_path=self.args.data_path,
                                               split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            'PEMS04': lambda: preprocess_pems04(rate=0.0, n_steps=n_steps,
                                               root_path=self.args.root_path,
                                               data_path=self.args.data_path,
                                               split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            # Custom datasets
            'Weather': lambda: preprocess_weather(rate=0.0, n_steps=n_steps, 
                                                root_path=self.args.root_path, 
                                                data_path=self.args.data_path,
                                                split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            'weather': lambda: preprocess_weather(rate=0.0, n_steps=n_steps,
                                                root_path=self.args.root_path,
                                                data_path=self.args.data_path,
                                                split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            'Exchange': lambda: preprocess_exchange(rate=0.0, n_steps=n_steps,
                                                  root_path=self.args.root_path,
                                                  data_path=self.args.data_path,
                                                  split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            'exchange': lambda: preprocess_exchange(rate=0.0, n_steps=n_steps,
                                                  root_path=self.args.root_path,
                                                  data_path=self.args.data_path,
                                                  split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            'Illness': lambda: preprocess_illness(rate=0.0, n_steps=n_steps,
                                                root_path=self.args.root_path,
                                                data_path=self.args.data_path,
                                                split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
            'illness': lambda: preprocess_illness(rate=0.0, n_steps=n_steps,
                                                root_path=self.args.root_path,
                                                data_path=self.args.data_path,
                                                split_style=split_style, test_non_overlapping=getattr(self.args, 'test_non_overlapping', False)),
        }
        
        if self.args.data not in dataset_map:
            raise ValueError(f"Dataset {self.args.data} not supported in BenchPOTS")
        
        # Load data
        data = dataset_map[self.args.data]()
        
        # Extract arrays
        train_X = data["train_X"]
        val_X = data["val_X"]
        test_X = data["test_X"]
        
        print(f"Data loaded - Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")
        
        return train_X, val_X, test_X
    
    def _load_benchpots_data(self):
        """Load data using BenchPOTS wrapper"""
        from data_provider.data_loader_benchpots import BenchPOTSWrapper
        
        print(f"Loading {self.args.data} from BenchPOTS...")
        
        # BenchPOTS configuration
        benchpots_config = {
            'dataset_name': self.args.data,
            'root_path': getattr(self.args, 'benchpots_root', '../dataset/benchpots'),
            'benchpots_missing_rate': getattr(self.args, 'benchpots_missing_rate', 0.1),
            'mit_rate': 0.0,  # MIT will be applied dynamically later
            'cache_mit_masks': False,
            'return_x_ori': True
        }
        
        # Create datasets for train/val/test
        train_dataset = BenchPOTSWrapper(subset='train', **benchpots_config)
        val_dataset = BenchPOTSWrapper(subset='val', **benchpots_config)
        test_dataset = BenchPOTSWrapper(subset='test', **benchpots_config)
        
        # Convert to PyPOTS format
        self.train_data = self._convert_benchpots_to_pypots(train_dataset)
        self.val_data = self._convert_benchpots_to_pypots(val_dataset) 
        self.test_data = self._convert_benchpots_to_pypots(test_dataset)
        
        print(f"BenchPOTS data loaded - Train: {self.train_data['X'].shape}, Val: {self.val_data['X'].shape}, Test: {self.test_data['X'].shape}")
    
    def _convert_benchpots_to_pypots(self, dataset):
        """Convert BenchPOTS dataset to PyPOTS dictionary format"""
        import numpy as np
        
        # Load all data into memory
        X_list = []
        X_ori_list = []
        
        for i in range(len(dataset)):
            X, X_ori, mask, _, _, _ = dataset[i]
            X_list.append(X.numpy())
            X_ori_list.append(X_ori.numpy())
        
        return {
            'X': np.array(X_list),
            'X_ori': np.array(X_ori_list)
        }
    
    def _load_csdi_data(self):
        """Load data using CSDI PM25 wrapper"""
        from data_provider.data_loader_pm25 import PM25Wrapper
        
        print(f"Loading {self.args.data} from CSDI...")
        
        # CSDI configuration
        csdi_config = {
            'root_path': getattr(self.args, 'csdi_root', os.path.join(getattr(self.args, 'root_base', '/ssd/datasets/TimeSeries'), 'pm25')),
            'eval_length': getattr(self.args, 'seq_len', 36),
            'target_dim': getattr(self.args, 'enc_in', 36),
            'return_x_ori': True,
            'validindex': getattr(self.args, 'validindex', 0)
        }
        
        # Create datasets for train/val/test
        train_dataset = PM25Wrapper(subset='train', **csdi_config)
        val_dataset = PM25Wrapper(subset='val', **csdi_config)
        test_dataset = PM25Wrapper(subset='test', **csdi_config)
        
        # Convert to PyPOTS format (including indicating_mask for CSDI)
        self.train_data = self._convert_csdi_to_pypots(train_dataset)
        self.val_data = self._convert_csdi_to_pypots(val_dataset)
        self.test_data = self._convert_csdi_to_pypots(test_dataset)
        
        print(f"CSDI data loaded - Train: {self.train_data['X'].shape}, Val: {self.val_data['X'].shape}, Test: {self.test_data['X'].shape}")
    
    def _convert_csdi_to_pypots(self, dataset):
        """Convert CSDI dataset to PyPOTS dictionary format with indicating mask"""
        import numpy as np
        
        # Load all data into memory
        X_list = []
        X_ori_list = []
        mask_list = []
        indicating_list = []
        
        for i in range(len(dataset)):
            X, X_ori, mask, indicating, _, _ = dataset[i]
            X_list.append(X.numpy())
            X_ori_list.append(X_ori.numpy())
            mask_list.append(mask.numpy())
            indicating_list.append(indicating.numpy())
        
        return {
            'X': np.array(X_list),
            'X_ori': np.array(X_ori_list),
            'missing_mask': np.array(mask_list),
            'indicating_mask': np.array(indicating_list)
        }
    
    def _apply_missing_patterns(self, X, patterns):
        """Apply missing patterns to data using MissingPatternGenerator"""
        mask = self.pattern_generator.generate_combined_mask(X.shape, patterns)
        X_masked = X.copy()
        X_masked[mask] = np.nan
        return X_masked
    
    def _create_model(self, n_steps, n_features, train_batch_size=None):
        """Create model using model_params from config"""
        # Create optimizer with learning rate
        from pypots_lib.optim.adam import Adam
        optimizer = Adam(lr=self.args.learning_rate)
        
        # Get model name first
        model_name = self.args.model.lower()
        
        # Common parameters
        common_params = {
            'n_steps': n_steps,
            'n_features': n_features,
            'epochs': self.args.train_epochs,
            'patience': self.args.patience,
            'batch_size': train_batch_size or self.args.batch_size,
            'optimizer': optimizer,
            'device': self.device,
            'num_workers': getattr(self.args, 'num_workers', 5),  # Use config value, default 5
            'mit_rate': getattr(self.args, 'mit_rate', 0.2),
            'MIT_weight': getattr(self.args, 'MIT_weight', 1.0),
            'ORT_weight': getattr(self.args, 'ORT_weight', 1.0),
        }
        
        # Handle base_loss for models that use ORTMITLoss
        if model_name in ['moderntcn', 'timesnet', 'timemixerpp', 'brits', 'itransformer', 'patchtst', 'dlinear']:
            base_loss_name = getattr(self.args, 'base_loss', 'MSE')
            if base_loss_name == 'MAE':
                common_params['base_loss'] = MAE
            else:
                common_params['base_loss'] = MSE
        
        # Check if AMP should be enabled from config
        enable_amp = getattr(self.args, 'enable_amp', False)
        if enable_amp:
            print(f"Enabling AMP for {model_name}")
            common_params['enable_amp'] = enable_amp
        
        # Get model class
        if model_name not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = self.MODEL_CLASSES[model_name]
        
        # Get model-specific parameters from config
        if hasattr(self.args, 'model_params') and self.args.model_params:
            # Handle OmegaConf
            if OmegaConf.is_config(self.args.model_params):
                model_params = OmegaConf.to_container(self.args.model_params, resolve=True)
            else:
                model_params = dict(self.args.model_params)
        else:
            # No model_params found
            model_params = {}
            print(f"Warning: No model_params found for {model_name}")

        if model_name == 'rddmpi':
            model_params['dataset'] = self.args.data
            model_params.setdefault('seed', self.args.seed)


        # Remove unnecessary parameters for specific models
        if model_name in ['csdi', 'rddmpi', 'gpvae', 'fgti']:
            # CSDI doesn't use mit_rate, MIT_weight, or ORT_weight
            common_params.pop('mit_rate', None)
            common_params.pop('MIT_weight', None)
            common_params.pop('ORT_weight', None)
            common_params.pop('base_loss', None)
        elif model_name == 'brits':
            # BRITS should use MAE loss, not ORTMITLoss
            common_params.pop('mit_rate', None)
            common_params['MIT_weight'] = 0  # Disable MIT for BRITS
            common_params['ORT_weight'] = 0  # Disable ORT for BRITS 
            common_params['training_loss'] = MAE  # Use MAE directly
        
        # For T1, pass all parameters from args
        if model_name == 't1':
            # Convert args to dict
            if OmegaConf.is_config(self.args):
                args_dict = OmegaConf.to_container(self.args, resolve=True)
            else:
                args_dict = vars(self.args) if hasattr(self.args, '__dict__') else {}
            
            # Remove PyPOTS reserved parameters (stored as strings)
            pypots_reserved_params = ['optimizer', 'training_loss', 'validation_metric']
            for param in pypots_reserved_params:
                if param in args_dict:
                    del args_dict[param]
            
            # Insert common_params first, then override with args
            model_params = {**common_params, **args_dict}
            
            # Handle base_loss
            base_loss_name = getattr(self.args, 'base_loss', 'MSE')
            if base_loss_name == 'MAE':
                model_params['base_loss'] = MAE
            else:
                model_params['base_loss'] = MSE
            

            if enable_amp:
                model_params['enable_amp'] = enable_amp
            
            # Handle separately
            try:
                model = model_class(**model_params)
            except Exception as e:
                print(f"Error creating model {model_name} with params: {model_params}")
                raise e
            
            return model
        
        # Create model for other models
        try:
            model = model_class(**common_params, **model_params)
        except Exception as e:
            print(f"Error creating model {model_name} with params: {model_params}")
            raise e
        
        return model
    
    def _run_test_scenario(self, model, test_X_clean, scenario_name, patterns, current_config):
        """Run a single test scenario with given patterns"""
        results = {}
        
        # Check if this is a multi-rate scenario (has 'rates' instead of 'rate')
        if patterns and 'rates' in patterns[0]:
            # Multi-rate scenario
            rates = patterns[0]['rates']
            for rate in rates:
                # Create pattern with single rate
                test_patterns = []
                for p in patterns:
                    pattern_copy = p.copy()
                    if 'rates' in pattern_copy:
                        pattern_copy['rate'] = rate
                        del pattern_copy['rates']
                    test_patterns.append(pattern_copy)
                
                # Apply patterns
                test_X = self._apply_missing_patterns(test_X_clean, test_patterns)
                test_set = {"X": test_X}
                
                # Impute
                imputation = model.impute(test_set)
                if isinstance(imputation, dict):
                    imputed_data = imputation["imputation"]
                else:
                    imputed_data = imputation

                # CRPS on full predictive samples if available
                crps = None
                mask = np.isnan(test_X) & ~np.isnan(test_X_clean)

                if len(imputed_data.shape) == 4 and imputed_data.shape[1] > 1 and np.sum(mask) > 0:
                    eval_points_t = torch.as_tensor(mask.astype(np.float32))
                    target_t = torch.as_tensor(test_X_clean.astype(np.float32))
                    forecast_t = torch.as_tensor(imputed_data.astype(np.float32))
                    crps = calc_quantile_CRPS_noscale(target_t, forecast_t, eval_points_t)

                results_dir = f'./lab/results/{current_config}/'
                os.makedirs(results_dir, exist_ok=True)

                save_path = os.path.join(results_dir, f"{scenario_name}_rate_{rate}_outputs.pk")

                with open(save_path, "wb") as f:
                    pickle.dump(
                        {
                            "imputed_data": imputed_data,
                            "test_X": test_X,
                            "test_X_clean": test_X_clean,
                            "scenario_name": scenario_name,
                            "crps": crps,
                        },
                        f,
                    )

                print(f"Saved test outputs to: {os.path.abspath(save_path)}")

                # Handle CSDI output shape
                if len(imputed_data.shape) == 4:
                    if imputed_data.shape[1] == 1:
                        imputed_data = imputed_data.squeeze(1)
                    else:
                        if isinstance(imputed_data, torch.Tensor):
                            imputed_data = torch.median(imputed_data, dim=1).values
                        else:
                            imputed_data = np.median(imputed_data, axis=1)
                
                # Check for NaN in predictions
                if np.isnan(imputed_data).any():
                    print(f"Warning: Model produced NaN values for {scenario_name}_rate_{rate}, skipping...")
                    continue
                
                # Calculate metrics on artificially missing values
                if np.sum(mask) > 0:
                    mae = float(calc_mae(imputed_data[mask], test_X_clean[mask]))
                    mse = float(calc_mse(imputed_data[mask], test_X_clean[mask]))
                    rmse = float(calc_rmse(imputed_data[mask], test_X_clean[mask]))
                    mape = float(MAPE(imputed_data[mask], test_X_clean[mask]))
                    mspe = float(MSPE(imputed_data[mask], test_X_clean[mask]))
                    
                    actual_rate = mask.sum() / mask.size
                    
                    results[f"{scenario_name}_rate_{rate}"] = {
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'mape': mape,
                        'mspe': mspe,
                        'crps': crps,
                        'actual_missing_rate': actual_rate,
                        'patterns': test_patterns
                    }
        else:
            # Single pattern scenario
            test_X = self._apply_missing_patterns(test_X_clean, patterns)
            test_set = {"X": test_X}

            # Impute
            imputation = model.impute(test_set)
            if isinstance(imputation, dict):
                imputed_data = imputation["imputation"]
            else:
                imputed_data = imputation

            # CRPS on full predictive samples if available
            crps = None
            mask = np.isnan(test_X) & ~np.isnan(test_X_clean)

            if len(imputed_data.shape) == 4 and imputed_data.shape[1] > 1 and np.sum(mask) > 0:
                eval_points_t = torch.as_tensor(mask.astype(np.float32))
                target_t = torch.as_tensor(test_X_clean.astype(np.float32))
                forecast_t = torch.as_tensor(imputed_data.astype(np.float32))
                crps = calc_quantile_CRPS_noscale(target_t, forecast_t, eval_points_t)

            results_dir = f'./lab/results/{current_config}/'
            os.makedirs(results_dir, exist_ok=True)

            save_path = os.path.join(results_dir, f"{scenario_name}__outputs.pk")

            with open(save_path, "wb") as f:
                pickle.dump(
                    {
                        "imputed_data": imputed_data,
                        "test_X": test_X,
                        "test_X_clean": test_X_clean,
                        "scenario_name": scenario_name,
                        "crps": crps,
                    },
                    f,
                )

            print(f"Saved test outputs to: {os.path.abspath(save_path)}")
            
            # Handle CSDI output shape
            if len(imputed_data.shape) == 4:
                if imputed_data.shape[1] == 1:
                    imputed_data = imputed_data.squeeze(1)
                else:
                    if isinstance(imputed_data, torch.Tensor):
                        imputed_data = torch.median(imputed_data, dim=1).values
                    else:
                        imputed_data = np.median(imputed_data, axis=1)
            
            # Check for NaN in predictions
            if np.isnan(imputed_data).any():
                print(f"Warning: Model produced NaN values for {scenario_name}, skipping...")
                return results
            
            # Calculate metrics
            mask = np.isnan(test_X) & ~np.isnan(test_X_clean)
            if np.sum(mask) > 0:
                mae = float(calc_mae(imputed_data[mask], test_X_clean[mask]))
                mse = float(calc_mse(imputed_data[mask], test_X_clean[mask]))
                rmse = float(calc_rmse(imputed_data[mask], test_X_clean[mask]))
                mape = float(MAPE(imputed_data[mask], test_X_clean[mask]))
                mspe = float(MSPE(imputed_data[mask], test_X_clean[mask]))
                
                actual_rate = mask.sum() / mask.size
                
                results[scenario_name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape,
                    'mspe': mspe,
                    'crps': crps,
                    'actual_missing_rate': actual_rate,
                    'patterns': patterns
                }
        
        return results
    
    def train(self, current_config, iteration=0):
        """Train with flexible missing patterns"""
        print("="*60)
        print(f"PyPOTS Training - Iteration {iteration+1}/{self.args.itr}")
        print(f"Model: {self.args.model}")
        print("="*60)
        
        # Clear metrics at the start of first iteration
        if iteration == 0:
            Exp_Imputation_PyPOTS._all_metrics = []
        
        # Load data - _load_data_native handles all data sources
        train_X_clean, val_X_clean, test_X_clean = self._load_data_native()
        
        # Get data source for special handling
        data_source = getattr(self.args, 'data_source', 'native')
        
        # Check if using CSDI data with pre-defined missing patterns
        if data_source == 'csdi':
            # CSDI uses pre-defined missing patterns
            print("Using CSDI pre-defined missing patterns")
            return self._train_with_csdi_patterns(train_X_clean, val_X_clean, test_X_clean, current_config, iteration)
        
        # Check if using mit_rate directly or mit_patterns
        if hasattr(self.args, 'mit_rate') and self.args.mit_rate is not None:
            # Direct mit_rate configuration
            print(f"Using direct MIT rate: {self.args.mit_rate}")
        elif hasattr(self.args, 'mit_patterns') and self.args.mit_patterns is not None:
            # Pattern-based configuration (will extract rate from patterns)
            print("Using pattern-based configuration")
        else:
            raise ValueError("Either mit_rate or mit_patterns must be specified in the config file.")
        
        # Use pattern system
        return self._train_with_patterns(train_X_clean, val_X_clean, test_X_clean, current_config, iteration)
    
    def _train_with_csdi_patterns(self, train_X_clean, val_X_clean, test_X_clean, current_config, iteration):
        """Train with CSDI pre-defined missing patterns"""
        
        # Get pre-defined missing data and masks
        train_X = self.train_data['X']  # Data with missing values
        train_mask = self.train_data['missing_mask']
        train_indicating_mask = self.train_data['indicating_mask']
        
        val_X = self.val_data['X']
        val_mask = self.val_data['missing_mask']
        val_indicating_mask = self.val_data['indicating_mask']
        
        test_X = self.test_data['X']
        test_mask = self.test_data['missing_mask']
        test_indicating_mask = self.test_data['indicating_mask']
        
        # For CSDI, we pass all data to avoid additional MIT masking
        # This makes DatasetForSAITS use the data as-is when return_X_ori=True
        
        # Prepare PyPOTS format with all required fields
        if self.args.data == 'pm25':
            # PM25 uses natural missing, let PyPOTS handle NaN values
            # DatasetForSAITS will apply MIT to observed values in training
            train_set = {
                "X": train_X,  # Contains NaN
                "dataset_name": "pm25"  # Mark as PM25 dataset
            }
            val_set = {
                "X": val_X,  # Contains NaN
                "X_ori": val_X_clean,  # Ground truth with NaN
                "dataset_name": "pm25"
            }
        else:
            # Other datasets with pre-defined masks
            train_set = {
                "X": train_X,
                "X_ori": train_X_clean,
                "missing_mask": train_mask,
                "indicating_mask": train_indicating_mask
            }
            val_set = {
                "X": val_X,
                "X_ori": val_X_clean,
                "missing_mask": val_mask,
                "indicating_mask": val_indicating_mask
            }
        
        # Create model
        model = self._create_model(
            n_steps=train_X.shape[1],
            n_features=train_X.shape[2]
        )
        
        # Profile model on first iteration
        if iteration == 0 and self.model_profile is None:
            try:
                self.model_profile = self._profile_pypots_model(model, train_X.shape)
            except Exception as e:
                print(f"Warning: Could not profile model: {e}")
                self.model_profile = {}  # Initialize as empty dict instead of None
        
        # Clear GPU memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Train with time tracking
        print(f"Training {self.args.model}...")
        
        # Calculate total iterations for time estimation
        train_size = train_X.shape[0]
        batch_size = self.args.batch_size
        iterations_per_epoch = (train_size + batch_size - 1) // batch_size
        total_iterations = iterations_per_epoch * self.args.train_epochs
        
        # Start training with timing
        start_time = time.time()
        self.time_profiler.reset()
        
        # For overall epoch timing
        epoch_start_time = start_time
        
        model.fit(train_set, val_set)
        
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Calculate time statistics
        avg_epoch_time = train_time / self.args.train_epochs
        avg_iter_time = train_time / total_iterations
        
        # Store time statistics in model profile
        if self.model_profile is None:
            self.model_profile = {}
        
        self.model_profile['s_per_epoch'] = round(avg_epoch_time, 2)
        self.model_profile['ms_per_iter'] = round(avg_iter_time * 1000, 2)
        self.model_profile['total_train_time_s'] = round(train_time, 2)
        print(f"Average time - Epoch: {avg_epoch_time:.2f}s, Iteration: {avg_iter_time*1000:.2f}ms")
        
        # Measure training peak memory
        if iteration == 0 and torch.cuda.is_available():
            try:
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                if self.model_profile is None:
                    self.model_profile = {}
                self.model_profile['training_peak_memory_mb'] = round(peak_memory_mb, 2)
                print(f"Actual Training Peak Memory: {peak_memory_mb:.2f} MB")
            except Exception as e:
                print(f"Warning: Could not get training peak memory: {e}")
        
        # Test with CSDI pre-defined test data
        print(f"\nTesting on CSDI pre-defined test set...")
        test_set = {"X": test_X}
        
        # Impute
        imputation = model.impute(test_set)
        if isinstance(imputation, dict):
            imputed_data = imputation["imputation"]
        else:
            imputed_data = imputation
        
        # Handle CSDI output shape
        if len(imputed_data.shape) == 4 and imputed_data.shape[1] == 1:
            imputed_data = imputed_data.squeeze(1)
        
        # Calculate metrics based on dataset
        if self.args.data == 'pm25':
            # PM25 with natural missing: handle NaN in ground truth
            # 1. Create mask for valid ground truth positions
            valid_gt_mask = ~np.isnan(test_X_clean)
            
            # 2. Fill NaN with 0 for metric calculation (won't affect result due to mask)
            test_X_clean_filled = np.nan_to_num(test_X_clean, nan=0.0)
            
            # 3. Evaluate only on natural missing positions that have valid ground truth
            natural_missing_mask = np.isnan(self.test_data['X'])
            eval_mask = natural_missing_mask & valid_gt_mask
            
            mae = calc_mae(imputed_data, test_X_clean_filled, eval_mask)
            mse = calc_mse(imputed_data, test_X_clean_filled, eval_mask)
            rmse = calc_rmse(imputed_data, test_X_clean_filled, eval_mask)
            
            print(f"Evaluating on {eval_mask.sum()} natural missing positions with valid ground truth")
        else:
            # Original: evaluate only on artificially masked values
            mae = calc_mae(imputed_data, test_X_clean, test_indicating_mask)
            mse = calc_mse(imputed_data, test_X_clean, test_indicating_mask)
            rmse = calc_rmse(imputed_data, test_X_clean, test_indicating_mask)
        
        results = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse)
        }
        
        print(f"CSDI Test Results - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        
        # Store results
        all_results = {'csdi_predefined': results}
        
        # Save model
        setting = f'{current_config}_csdi'
        checkpoint_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save(os.path.join(checkpoint_path, "model.pypots"), overwrite=True)
        
        # Aggregate metrics for this iteration
        iteration_metrics = self._aggregate_pattern_metrics(all_results)
        Exp_Imputation_PyPOTS._all_metrics.append(iteration_metrics)
        
        # Save results table
        self._save_pattern_results_table(all_results, current_config, iteration)
        
        # On last iteration, save final results
        if IterationManager.is_last_iteration(iteration, self.args.itr):
            self._save_final_results(current_config)
        
        return all_results
    
    def _train_with_patterns(self, train_X_clean, val_X_clean, test_X_clean, current_config, iteration):
        """Train with flexible missing patterns"""
        
        # Check if mit_rate is directly provided
        if hasattr(self.args, 'mit_rate') and self.args.mit_rate is not None:
            mit_rate = self.args.mit_rate
            print(f"Using direct MIT rate: {mit_rate}")
        elif hasattr(self.args, 'mit_patterns') and self.args.mit_patterns is not None:
            # Extract MIT rate from patterns for training (only use MCAR)
            print("Using pattern-based configuration")
            mit_patterns = OmegaConf.to_container(self.args.mit_patterns)
            mit_rate = 0.2  # default
            
            # Find the first MCAR pattern and use its rate
            for pattern in mit_patterns:
                if pattern.get('type') == 'mcar':
                    mit_rate = pattern.get('rate', 0.2)
                    print(f"Extracted MIT rate {mit_rate} from MCAR pattern for training")
                    break
        else:
            mit_rate = 0.2  # fallback default
            print(f"Using default MIT rate: {mit_rate}")
        
        # For training, we'll let the model's DatasetForSAITS handle the masking with the rate
        # So we don't apply patterns here, just pass clean data
        train_X = train_X_clean
        
        # For validation, use clean data just like training
        # DatasetForSAITS will handle artificial masking for MIT
        val_X = val_X_clean
        
        # Prepare PyPOTS format
        train_set = {"X": train_X}
        
        # For validation, provide both X and X_ori like other PyPOTS models
        # This matches the standard PyPOTS validation approach
        # Apply artificial missing values to X while keeping X_ori clean
        from pygrinder import mcar
        val_X_with_missing = mcar(val_X_clean, p=mit_rate)
        val_set = {"X": val_X_with_missing, "X_ori": val_X_clean}
        
        # Create model with the mit_rate
        # Only update args.mit_rate if it wasn't already in config
        if not hasattr(self.args, 'mit_rate'):
            self.args.mit_rate = mit_rate
        else:
            # Use the already set mit_rate
            self.args.mit_rate = mit_rate
        

        model = self._create_model(
            n_steps=train_X.shape[1],
            n_features=train_X.shape[2],
        )
        
        # Profile model on first iteration (before training)
        if iteration == 0 and self.model_profile is None:
            try:
                # Full profiling (parameters, FLOPs, inference memory)
                self.model_profile = self._profile_pypots_model(model, train_X.shape)
            except Exception as e:
                print(f"Warning: Could not profile model: {e}")
                self.model_profile = {}  # Initialize as empty dict instead of None
        
        # Clear GPU memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Train with time tracking
        print(f"Training {self.args.model}...")
        
        # Calculate total iterations for time estimation
        train_size = train_X.shape[0]
        batch_size = self.args.batch_size
        iterations_per_epoch = (train_size + batch_size - 1) // batch_size
        total_iterations = iterations_per_epoch * self.args.train_epochs
        
        # Start training with timing
        start_time = time.time()
        self.time_profiler.reset()
        
        # For overall epoch timing
        epoch_start_time = start_time

        # results_dir = f'./lab/results/{current_config}/'
        # os.makedirs(results_dir, exist_ok=True)
        # seed_suffix = f"_seed_{self.args.seed}" if hasattr(self.args, "seed") else ""
        # # seed_suffix = ""
        # pth_path = os.path.join(results_dir, f"model_used_for_testing{seed_suffix}.pth")
        # print(f"Loading existing PTH model from: {os.path.abspath(pth_path)}")
        # state_dict = torch.load(pth_path, map_location=self.device)
        # model.model.load_state_dict(state_dict, strict=False)

        model.fit(train_set, val_set)
        
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Calculate time statistics
        avg_epoch_time = train_time / self.args.train_epochs
        avg_iter_time = train_time / total_iterations
        
        # Store time statistics in model profile
        if self.model_profile is None:
            self.model_profile = {}
        
        self.model_profile['s_per_epoch'] = round(avg_epoch_time, 2)
        self.model_profile['ms_per_iter'] = round(avg_iter_time * 1000, 2)
        self.model_profile['total_train_time_s'] = round(train_time, 2)
        print(f"Average time - Epoch: {avg_epoch_time:.2f}s, Iteration: {avg_iter_time*1000:.2f}ms")
        
        # Measure actual training peak memory after training
        if iteration == 0 and self.model_profile is not None and torch.cuda.is_available():
            try:
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.model_profile['training_peak_memory_mb'] = round(peak_memory_mb, 2)
                print(f"Actual Training Peak Memory: {peak_memory_mb:.2f} MB")
            except Exception as e:
                print(f"Warning: Could not get training peak memory: {e}")
        
        # Test with scenarios
        all_results = {}
        test_scenarios = getattr(self.args, 'test_scenarios', [])
        
        if test_scenarios:
            # Convert OmegaConf to container
            test_scenarios = OmegaConf.to_container(test_scenarios)
            
            for scenario in test_scenarios:
                scenario_name = scenario['name']
                patterns = scenario['patterns']
                
                print(f"\nTesting scenario: {scenario_name}")
                scenario_results = self._run_test_scenario(
                    model, test_X_clean, scenario_name, patterns, current_config
                )
                all_results.update(scenario_results)
        else:
            # Fallback to simple MCAR test
            print("\nNo test scenarios defined, using default MCAR test")
            test_patterns = [{"type": "mcar", "rates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
            scenario_results = self._run_test_scenario(
                model, test_X_clean, "default", test_patterns
            )
            all_results.update(scenario_results)
        
        # Save model
        setting = f'{current_config}_patterns'
        checkpoint_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save(os.path.join(checkpoint_path, "model.pypots"), overwrite=True)
        pypots_path = os.path.join(checkpoint_path, "model.pypots")
        print(f"Saved PyPOTS model to: {pypots_path}")

        seed_suffix = f"_seed_{self.args.seed}" if hasattr(self.args, "seed") else ""

        results_dir = f'./lab/results/{current_config}/'
        os.makedirs(results_dir, exist_ok=True)
        pth_path = os.path.join(results_dir, f"model_used_for_testing{seed_suffix}.pth")
        torch.save(model.model.state_dict(), pth_path)
        print(f"Saved PyTorch state_dict to: {os.path.abspath(pth_path)}")
        
        # Aggregate metrics for this iteration
        iteration_metrics = self._aggregate_pattern_metrics(all_results)
        Exp_Imputation_PyPOTS._all_metrics.append(iteration_metrics)
        
        # Save results table
        self._save_pattern_results_table(all_results, current_config, iteration)
        
        # On last iteration, save final results
        if IterationManager.is_last_iteration(iteration, self.args.itr):
            self._save_final_results(current_config)
        
        return all_results
    
    
    def _save_pattern_results_table(self, results, config, iteration):
        """Save pattern results in table format"""
        results_dir = f'./lab/results/{config}/'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save raw results as JSON
        json_path = os.path.join(results_dir, 'pattern_results.json')
        
        # Use the global convert_to_serializable function
        serializable_results = convert_to_serializable(results)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Create summary table
        summary_data = []
        for key, value in results.items():
            if isinstance(value, dict) and 'mse' in value:
                # Direct result
                summary_data.append({
                    'scenario': key,
                    'mse': value['mse'],
                    'mae': value['mae'],
                    'rmse': value['rmse'],
                    'crps': value.get('crps', None),
                    'actual_rate': value.get('actual_missing_rate', 'N/A')
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(results_dir, 'pattern_results_summary.csv')
            df.to_csv(csv_path, index=False)
            
            print("\n" + "="*60)
            print("PATTERN RESULTS SUMMARY")
            print("="*60)
            print(df.to_string(index=False))
        
        print(f"\nResults saved to: {results_dir}")
    
    def _aggregate_pattern_metrics(self, all_results):
        """Aggregate metrics across all scenarios"""
        # Flatten results
        all_metrics = []
        
        for key, value in all_results.items():
            if isinstance(value, dict) and 'mse' in value:
                # Direct metrics
                all_metrics.append(value)
        
        # Calculate averages
        if all_metrics:
            avg_metrics = {}
            for metric in ['mae', 'mse', 'rmse', 'mape', 'mspe', 'crps']:
                values = [m[metric] for m in all_metrics if metric in m and m[metric] is not None and not np.isnan(float(m[metric]))]
                avg_metrics[metric] = float(np.mean(values)) if values else float('nan')
            
            return {
                **avg_metrics,
                'all_results': all_results
            }
        
        return {'all_results': all_results}
    
    
    def test(self, setting, test=0, iteration=0):
        """Simple test function for compatibility"""
        print("Use train() for pattern-based experiments")
        return 0.0
    
    def _profile_pypots_model(self, model, input_shape):
        """Profile PyPOTS model - parameters, FLOPs, and memory"""
        profile_results = {}
        
        # Get the actual PyTorch model inside PyPOTS wrapper
        pytorch_model = model.model
        
        # 1. Count parameters
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
        
        profile_results['total_parameters'] = int(total_params)
        profile_results['trainable_parameters'] = int(trainable_params)
        profile_results['total_parameters_M'] = round(total_params / 1e6, 2)
        profile_results['trainable_parameters_M'] = round(trainable_params / 1e6, 2)
        
        print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    
    def _measure_training_memory(self, model, train_dataloader):
        """Measure peak memory during training (PyPOTS format)"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Simulate a few training steps
                model.model.train()  # Use the internal PyTorch model
                for i, data in enumerate(train_dataloader):
                    if i >= 3:  # Only run 3 batches
                        break
                    

                    indices, X, missing_mask, X_ori, indicating_mask = data
                    X = X.to(self.device)
                    missing_mask = missing_mask.to(self.device)
                    X_ori = X_ori.to(self.device)
                    indicating_mask = indicating_mask.to(self.device)
                    
                    inputs = {
                        "X": X,
                        "missing_mask": missing_mask,
                        "X_ori": X_ori,
                        "indicating_mask": indicating_mask,
                    }
                    results = model.model(inputs, calc_criterion=True)
                    loss = results["loss"].sum()
                    loss.backward()
                    # Use the PyPOTS model's optimizer
                    if hasattr(model, 'optimizer') and hasattr(model.optimizer, 'zero_grad'):
                        model.optimizer.zero_grad()
                    else:
                        # If optimizer not accessible, just clear gradients from parameters
                        for param in model.model.parameters():
                            if param.grad is not None:
                                param.grad.zero_()
                
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
                
                print(f"Estimated Training Peak Memory: {peak_memory_mb:.2f} MB")
                return round(peak_memory_mb, 2)
            else:
                return None
        except Exception as e:
            print(f"Warning: Could not measure training memory: {e}")
            return None
    
    
    def _save_final_results(self, current_config):
        """Save final results with statistics across iterations"""
        metric_names = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'crps']
        
        # Calculate overall statistics
        metrics_stats = MetricsAggregator.calculate_statistics(
            Exp_Imputation_PyPOTS._all_metrics, metric_names
        )
        
        # Pattern system - aggregate scenario results
        all_scenario_results = {}
        
        if Exp_Imputation_PyPOTS._all_metrics and 'all_results' in Exp_Imputation_PyPOTS._all_metrics[0]:
            # Get all scenario keys from first iteration
            scenario_keys = list(Exp_Imputation_PyPOTS._all_metrics[0]['all_results'].keys())
            
            for scenario_key in scenario_keys:
                # Collect this scenario's results from all iterations
                scenario_results = []
                for metrics in Exp_Imputation_PyPOTS._all_metrics:
                    if 'all_results' in metrics and scenario_key in metrics['all_results']:
                        scenario_results.append(metrics['all_results'][scenario_key])
                
                if scenario_results:
                    scenario_stats = {}
                    for metric_name in metric_names:
                        values = [r[metric_name] for r in scenario_results if metric_name in r and r[metric_name] is not None and not np.isnan(float(r[metric_name]))]
                        if values:
                            scenario_stats[metric_name] = {
                                'mean': float(np.mean(values)),
                                'std': float(np.std(values))
                            }
                        else:
                            scenario_stats[metric_name] = {
                                'mean': float('nan'),
                                'std': float('nan')
                            }
                    all_scenario_results[scenario_key] = scenario_stats
        
        metrics_stats['scenario_results'] = all_scenario_results

        final_rows = []

        for scenario, stats in all_scenario_results.items():
            row = {'scenario': scenario}
            for metric, values in stats.items():
                row[f'{metric}_mean'] = values['mean']
                row[f'{metric}_std'] = values['std']
            final_rows.append(row)

        if final_rows:
            df = pd.DataFrame(final_rows)
            results_dir = f'./lab/results/{current_config}/'
            df.to_csv(os.path.join(results_dir, 'final_results_mean_std.csv'), index=False)
        
        # Add configuration info
        from omegaconf import OmegaConf
        config_info = {
            'data': self.args.data,
            'model': self.args.model,
            'seq_len': self.args.seq_len,
            'mit_patterns': OmegaConf.to_container(getattr(self.args, 'mit_patterns', None), resolve=True) if hasattr(self.args, 'mit_patterns') else None,
            'test_scenarios': OmegaConf.to_container(getattr(self.args, 'test_scenarios', None), resolve=True) if hasattr(self.args, 'test_scenarios') else None
        }
        
        metrics_stats['config'] = config_info
        
        # Save using MetricsAggregator.save_final_results
        MetricsAggregator.save_final_results(
            current_config,  # setting
            metrics_stats,   # metrics_stats
            self.args,       # args
            self.model_profile  # model_profile
        )
        
        print(f"\nFinal results saved with {len(Exp_Imputation_PyPOTS._all_metrics)} iterations")