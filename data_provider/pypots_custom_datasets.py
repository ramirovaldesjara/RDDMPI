"""
Custom PyPOTS dataset implementations for Weather, Exchange, and Illness datasets.
These datasets are formatted to work with PyPOTS and BenchPOTS style loading.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Global cache for preprocessed data
_DATA_CACHE = {}


def _get_cached_or_process(dataset_name, data_loader_func, rate, n_steps, **kwargs):
    """
    Helper function to handle caching logic for all datasets.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset for cache key
    data_loader_func : callable
        Function that loads and processes the data
    rate : float
        Missing rate
    n_steps : int
        Sequence length
    **kwargs : dict
        Additional arguments (root_path, data_path, etc.)
    """
    # Create cache key (include split_style to differentiate caches)
    data_path = kwargs.get('data_path', '')
    root_path = kwargs.get('root_path', '')
    split_style = kwargs.get('split_style', 'traditional')
    test_non_overlapping = kwargs.get('test_non_overlapping', False)
    test_stride = kwargs.get('test_stride', None)
    cache_key = (dataset_name, n_steps, root_path, data_path, split_style, test_non_overlapping, test_stride)
    
    # Check cache for clean data (rate=0.0)
    if rate == 0.0 and cache_key in _DATA_CACHE:
        print(f"Loading {dataset_name} data from cache...")
        cached = _DATA_CACHE[cache_key]
        return {
            'n_steps': cached['n_steps'],
            'n_features': cached['n_features'],
            'train_X': cached['train_X'].copy(),
            'val_X': cached['val_X'].copy(),
            'test_X': cached['test_X'].copy(),
            'scaler': cached['scaler']
        }
    
    # Load and process data
    result = data_loader_func(n_steps, **kwargs)
    
    # Cache if rate is 0.0
    if rate == 0.0:
        _DATA_CACHE[cache_key] = {
            'n_steps': result['n_steps'],
            'n_features': result['n_features'],
            'train_X': result['train_X'].copy(),
            'val_X': result['val_X'].copy(),
            'test_X': result['test_X'].copy(),
            'scaler': result['scaler']
        }
        print(f"Cached {dataset_name} data for future use")
    
    # Apply missing rate if needed
    if rate > 0:
        from pygrinder import mcar
        result['train_X'] = mcar(result['train_X'], p=rate)
        result['val_X'] = mcar(result['val_X'], p=rate)
        result['test_X'] = mcar(result['test_X'], p=rate)
    
    return result


def create_sliding_windows(data, n_steps, step=1):
    n_samples = (len(data) - n_steps) // step + 1
    X = np.array([data[i:i + n_steps] for i in range(0, len(data) - n_steps + 1, step)])
    return X


def split_train_val_test(X, train_ratio=0.7, val_ratio=0.15, split_style="traditional",
    test_non_overlapping = False,
    n_steps = None,
    test_stride=None,
    train_stride=None
):
    """Split data into train, validation, and test sets
    
    Parameters:
    -----------
    X : array-like
        The data to split
    train_ratio : float
        Ratio of training data (default: 0.7)
    val_ratio : float
        Ratio of validation data (default: 0.15 for pypots, ignored for traditional)
    split_style : str
        "traditional" (default): 70%-10%-20% split (matching exp_imputation)
        "pypots": 70%-15%-15% split
    """
    n_samples = len(X)
    
    if split_style == "traditional":
        # exp_imputation style: 70%-10%-20%
        val_ratio = 0.1
        test_ratio = 0.2
    else:
        # pypots style: 70%-15%-15%
        test_ratio = 1 - train_ratio - val_ratio
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # ----------- TRAIN STRIDE -----------
    if train_stride is None:
        train_stride = 1
    if train_stride <= 0:
        raise ValueError(f"train_stride must be >= 1, got {train_stride}")

    train_X = X[:train_end:train_stride]
    val_X = X[train_end:val_end:train_stride]

    # Backward-compatible stride logic
    if test_stride is None:
        if test_non_overlapping:
            if n_steps is None:
                raise ValueError("n_steps must be provided when test_non_overlapping=True")
            test_stride = n_steps
        else:
            test_stride = 1

    if test_stride <= 0:
        raise ValueError(f"test_stride must be >= 1, got {test_stride}")

    test_X = X[val_end::test_stride]

    return train_X, val_X, test_X


def _load_weather_data(n_steps: int, **kwargs) -> Dict:
    """Internal function to load and process weather data."""
    # Get split style from kwargs
    split_style = kwargs.get('split_style', 'traditional')
    
    # Check if data path is provided
    data_path = kwargs.get('data_path', 'weather.csv')
    root_path = kwargs.get('root_path', '../dataset')
    
    # Handle root_path that is './' or '.'
    if root_path == './' or root_path == '.':
        root_path = '../dataset'
    
    # Extract filename and construct correct path
    file_name = os.path.basename(data_path)
    full_path = os.path.join(root_path, 'weather', file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Weather dataset not found at {full_path}")
    
    # Read the data
    df = pd.read_csv(full_path)
    
    # Remove date column if exists
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Convert to numpy array
    data = df.values.astype(np.float32)
    
    # Standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Create sliding windows
    X = create_sliding_windows(data, n_steps)
    
    # Split into train, val, test
    train_X, val_X, test_X = split_train_val_test(
        X,
        split_style=split_style,
        test_non_overlapping=kwargs.get("test_non_overlapping", False),
        n_steps=n_steps,
        test_stride=kwargs.get("test_stride", None),
        train_stride=kwargs.get("train_stride", 6),
    )
    
    return {
        'n_steps': n_steps,
        'n_features': data.shape[1],
        'train_X': train_X,
        'val_X': val_X,
        'test_X': test_X,
        'scaler': scaler
    }


def preprocess_weather(rate: float = 0.0, n_steps: int = 96, **kwargs) -> Dict:
    """
    Load and preprocess Weather dataset in PyPOTS format.
    
    Parameters:
    -----------
    rate : float
        Initial missing rate (usually 0.0 as we apply missing masks dynamically)
    n_steps : int
        Sequence length for sliding windows
    
    Returns:
    --------
    Dict containing preprocessed data in PyPOTS format
    """
    return _get_cached_or_process('weather', _load_weather_data, rate, n_steps, **kwargs)


def _load_exchange_data(n_steps: int, **kwargs) -> Dict:
    """Internal function to load and process exchange data."""
    # Get split style from kwargs
    split_style = kwargs.get('split_style', 'traditional')
    
    # Check if data path is provided
    data_path = kwargs.get('data_path', 'exchange_rate.csv')
    root_path = kwargs.get('root_path', '../dataset')
    
    # Handle root_path that is './' or '.'
    if root_path == './' or root_path == '.':
        root_path = '../dataset'
    
    # Extract filename and construct correct path
    file_name = os.path.basename(data_path)
    full_path = os.path.join(root_path, 'exchange_rate', file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Exchange dataset not found at {full_path}")
    
    # Read the data
    df = pd.read_csv(full_path)
    
    # Remove date column if exists
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Convert to numpy array
    data = df.values.astype(np.float32)
    
    # Standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Create sliding windows
    X = create_sliding_windows(data, n_steps)
    
    # Split into train, val, test
    train_X, val_X, test_X = split_train_val_test(
        X,
        split_style=split_style,
        test_non_overlapping=kwargs.get("test_non_overlapping", False),
        n_steps=n_steps, test_stride=kwargs.get("test_stride", None),
    )
    
    return {
        'n_steps': n_steps,
        'n_features': data.shape[1],
        'train_X': train_X,
        'val_X': val_X,
        'test_X': test_X,
        'scaler': scaler
    }


def preprocess_exchange(rate: float = 0.0, n_steps: int = 96, **kwargs) -> Dict:
    """
    Load and preprocess Exchange dataset in PyPOTS format.
    
    Parameters:
    -----------
    rate : float
        Initial missing rate (usually 0.0 as we apply missing masks dynamically)
    n_steps : int
        Sequence length for sliding windows
    
    Returns:
    --------
    Dict containing preprocessed data in PyPOTS format
    """
    return _get_cached_or_process('exchange', _load_exchange_data, rate, n_steps, **kwargs)


def _load_illness_data(n_steps: int, **kwargs) -> Dict:
    """Internal function to load and process illness data."""
    # Get split style from kwargs
    split_style = kwargs.get('split_style', 'traditional')
    
    # Check if data path is provided
    data_path = kwargs.get('data_path', 'national_illness.csv')
    root_path = kwargs.get('root_path', '../dataset')
    
    # Handle root_path that is './' or '.'
    if root_path == './' or root_path == '.':
        root_path = '../dataset'
    
    # Extract filename and construct correct path
    file_name = os.path.basename(data_path)
    full_path = os.path.join(root_path, 'illness', file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Illness dataset not found at {full_path}")
    
    # Read the data
    df = pd.read_csv(full_path)
    
    # Remove date column if exists
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Convert to numpy array
    data = df.values.astype(np.float32)
    
    # Standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Create sliding windows
    X = create_sliding_windows(data, n_steps)
    
    # Split into train, val, test
    train_X, val_X, test_X = split_train_val_test(
        X,
        split_style=split_style,
        test_non_overlapping=kwargs.get("test_non_overlapping", False),
        n_steps=n_steps, test_stride=kwargs.get("test_stride", None),
    )
    
    return {
        'n_steps': n_steps,
        'n_features': data.shape[1],
        'train_X': train_X,
        'val_X': val_X,
        'test_X': test_X,
        'scaler': scaler
    }


def preprocess_illness(rate: float = 0.0, n_steps: int = 48, **kwargs) -> Dict:
    """
    Load and preprocess Illness (ILI) dataset in PyPOTS format.
    
    Parameters:
    -----------
    rate : float
        Initial missing rate (usually 0.0 as we apply missing masks dynamically)
    n_steps : int
        Sequence length for sliding windows (default 48 for illness)
    
    Returns:
    --------
    Dict containing preprocessed data in PyPOTS format
    """
    return _get_cached_or_process('illness', _load_illness_data, rate, n_steps, **kwargs)


# Alias functions for consistency with BenchPOTS naming
def preprocess_Weather(rate: float = 0.0, n_steps: int = 96, **kwargs) -> Dict:
    """Alias for weather dataset with capital W"""
    return preprocess_weather(rate=rate, n_steps=n_steps, **kwargs)


def preprocess_Exchange(rate: float = 0.0, n_steps: int = 96, **kwargs) -> Dict:
    """Alias for exchange dataset with capital E"""
    return preprocess_exchange(rate=rate, n_steps=n_steps, **kwargs)


def preprocess_Illness(rate: float = 0.0, n_steps: int = 48, **kwargs) -> Dict:
    """Alias for illness dataset with capital I"""
    return preprocess_illness(rate=rate, n_steps=n_steps, **kwargs)


def _load_pems03_data(n_steps: int, **kwargs) -> Dict:
    """Internal function to load and process PEMS03 data."""
    # Get split style from kwargs
    split_style = kwargs.get('split_style', 'traditional')
    
    # Check if data path is provided
    data_path = kwargs.get('data_path', 'PEMS03.npz')
    root_path = kwargs.get('root_path', '../dataset')
    
    # Handle root_path that is './' or '.'
    if root_path == './' or root_path == '.':
        root_path = '../dataset'
    
    # Extract filename and construct correct path
    file_name = os.path.basename(data_path)
    full_path = os.path.join(root_path, 'PEMS', file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"PEMS03 dataset not found at {full_path}")
    
    # Load npz file
    data_file = np.load(full_path)
    data = data_file['data']  # Shape: (num_samples, num_nodes, num_features)
    
    # Reshape from (samples, nodes, features) to (samples, features)
    # where features = nodes * original_features
    n_samples, n_nodes, n_features = data.shape
    data = data.reshape(n_samples, n_nodes * n_features)
    
    # Standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Create sliding windows
    X = create_sliding_windows(data, n_steps)
    
    # Split into train, val, test
    train_X, val_X, test_X = split_train_val_test(
        X,
        split_style=split_style,
        test_non_overlapping=kwargs.get("test_non_overlapping", False),
        n_steps=n_steps, test_stride=kwargs.get("test_stride", None),
    )
    
    return {
        'n_steps': n_steps,
        'n_features': n_nodes * n_features,
        'train_X': train_X,
        'val_X': val_X,
        'test_X': test_X,
        'scaler': scaler
    }


def preprocess_pems03(rate: float = 0.0, n_steps: int = 96, **kwargs) -> Dict:
    """
    Load and preprocess PEMS03 dataset in PyPOTS format.
    
    Parameters:
    -----------
    rate : float
        Initial missing rate (usually 0.0 as we apply missing masks dynamically)
    n_steps : int
        Sequence length for sliding windows
    
    Returns:
    --------
    Dict containing preprocessed data in PyPOTS format
    """
    return _get_cached_or_process('pems03', _load_pems03_data, rate, n_steps, **kwargs)


def _load_pems04_data(n_steps: int, **kwargs) -> Dict:
    """Internal function to load and process PEMS04 data."""
    # Get split style from kwargs
    split_style = kwargs.get('split_style', 'traditional')
    
    # Check if data path is provided
    data_path = kwargs.get('data_path', 'PEMS04.npz')
    root_path = kwargs.get('root_path', '../dataset/')
    
    # Handle root_path that is './' or '.'
    if root_path == './' or root_path == '.':
        root_path = '../dataset'
    
    # Extract filename and construct correct path
    file_name = os.path.basename(data_path)
    full_path = os.path.join(root_path, 'PEMS', file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"PEMS04 dataset not found at {full_path}")
    
    # Load npz file
    data_file = np.load(full_path)
    data = data_file['data']  # Shape: (num_samples, num_nodes, num_features)
    
    # Reshape from (samples, nodes, features) to (samples, features)
    # where features = nodes * original_features
    n_samples, n_nodes, n_features = data.shape
    data = data.reshape(n_samples, n_nodes * n_features)
    
    # Standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Create sliding windows
    X = create_sliding_windows(data, n_steps)
    
    # Split into train, val, test
    train_X, val_X, test_X = split_train_val_test(
        X,
        split_style=split_style,
        test_non_overlapping=kwargs.get("test_non_overlapping", False),
        n_steps=n_steps, test_stride=kwargs.get("test_stride", None),
    )
    
    return {
        'n_steps': n_steps,
        'n_features': n_nodes * n_features,
        'train_X': train_X,
        'val_X': val_X,
        'test_X': test_X,
        'scaler': scaler
    }


def preprocess_pems04(rate: float = 0.0, n_steps: int = 96, **kwargs) -> Dict:
    """
    Load and preprocess PEMS04 dataset in PyPOTS format.
    
    Parameters:
    -----------
    rate : float
        Initial missing rate (usually 0.0 as we apply missing masks dynamically)
    n_steps : int
        Sequence length for sliding windows
    
    Returns:
    --------
    Dict containing preprocessed data in PyPOTS format
    """
    return _get_cached_or_process('pems04', _load_pems04_data, rate, n_steps, **kwargs)


# Alias functions
def preprocess_PEMS03(rate: float = 0.0, n_steps: int = 96, **kwargs) -> Dict:
    """Alias for PEMS03 dataset with capital letters"""
    return preprocess_pems03(rate=rate, n_steps=n_steps, **kwargs)


def preprocess_PEMS04(rate: float = 0.0, n_steps: int = 96, **kwargs) -> Dict:
    """Alias for PEMS04 dataset with capital letters"""
    return preprocess_pems04(rate=rate, n_steps=n_steps, **kwargs)


# Dataset metadata for reference
DATASET_INFO = {
    'weather': {
        'n_features': 21,
        'default_seq_len': 96,
        'file_path': '../dataset/weather/weather.csv'
    },
    'exchange': {
        'n_features': 8,
        'default_seq_len': 96,
        'file_path': '../dataset/exchange_rate/exchange_rate.csv'
    },
    'illness': {
        'n_features': 7,
        'default_seq_len': 48,
        'file_path': '../dataset/illness/national_illness.csv'
    },
    'pems03': {
        'n_features': 358,  # Will be determined by actual data shape
        'default_seq_len': 96,
        'file_path': '../dataset/PEMS/PEMS03.npz'
    },
    'pems04': {
        'n_features': 307,  # Will be determined by actual data shape
        'default_seq_len': 96,
        'file_path': '../dataset/PEMS/PEMS04.npz'
    }
}


def _load_ett_data(n_steps: int, **kwargs) -> Dict:
    """Internal function to load and process ETT data."""
    # Get subset from kwargs
    subset = kwargs.get('subset')
    if not subset:
        raise ValueError("subset parameter is required for ETT data loading")
    
    # Get split style from kwargs
    split_style = kwargs.get('split_style', 'traditional')
    
    # Get root path from kwargs or use default
    root_path = kwargs.get('root_path', '../dataset')
    data_path = kwargs.get('data_path', f'{subset}.csv')
    
    # Handle root_path that is './' or '.'
    if root_path == './' or root_path == '.':
        root_path = '../dataset'
    
    # Construct path
    file_name = os.path.basename(data_path)
    full_path = os.path.join(root_path, 'ETT-small', file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"ETT dataset not found at {full_path}")
    
    # Read the data
    df = pd.read_csv(full_path)
    
    # Remove date column
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Convert to numpy array
    data = df.values.astype(np.float32)
    
    # Standardize the data
    scaler = StandardScaler()
    
    if split_style == "traditional":
        # Use traditional ETT split (fixed size: 12+4+4 months)
        # This mimics exp_imputation's Dataset_ETT_hour and Dataset_ETT_minute
        if 'h' in subset.lower():  # Hourly data
            # 12 months for train, 4 for val, 4 for test
            train_size = 12 * 30 * 24
            val_size = 4 * 30 * 24
            test_size = 4 * 30 * 24
        else:  # Minute data (ETTm1, ETTm2)
            # 12 months for train, 4 for val, 4 for test (15-minute intervals)
            train_size = 12 * 30 * 24 * 4
            val_size = 4 * 30 * 24 * 4
            test_size = 4 * 30 * 24 * 4
        
        # Only use the first 20 months of data
        total_size = train_size + val_size + test_size
        data = data[:total_size]
        
        # Fit scaler on train data only
        scaler.fit(data[:train_size])
        data = scaler.transform(data)
        
        # Create sliding windows for each split
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        train_X = create_sliding_windows(train_data, n_steps)
        val_X = create_sliding_windows(val_data, n_steps)
        test_stride = kwargs.get("test_stride", None)
        if test_stride is None:
            test_non_overlapping = kwargs.get("test_non_overlapping", False)
            test_stride = n_steps if test_non_overlapping else 1

        if test_stride <= 0:
            raise ValueError(f"test_stride must be >= 1, got {test_stride}")

        test_X = create_sliding_windows(test_data, n_steps, step=test_stride)
    else:
        # Use pypots style (percentage-based split)
        data = scaler.fit_transform(data)
        X = create_sliding_windows(data, n_steps)
        train_X, val_X, test_X = split_train_val_test(X, split_style=split_style)
    
    return {
        'n_steps': n_steps,
        'n_features': data.shape[1],
        'train_X': train_X,
        'val_X': val_X,
        'test_X': test_X,
        'scaler': scaler
    }


def preprocess_ett(subset: str, rate: float = 0.0, n_steps: int = 96, **kwargs) -> Dict:
    """
    Load and preprocess ETT dataset in PyPOTS format.
    
    Parameters:
    -----------
    subset : str
        ETT subset name ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')
    rate : float
        Initial missing rate (usually 0.0)
    n_steps : int
        Sequence length for sliding windows
    
    Returns:
    --------
    Dict containing preprocessed data in PyPOTS format
    """
    # ETT has subsets, so handle differently
    kwargs_with_subset = kwargs.copy()
    kwargs_with_subset['subset'] = subset
    
    # Update data_path to create cache key including subset
    if 'data_path' not in kwargs:
        kwargs_with_subset['data_path'] = f'{subset}.csv'
    
    return _get_cached_or_process(f'ett_{subset}', 
                                 lambda n_steps, **kw: _load_ett_data(n_steps, **kw), 
                                 rate, n_steps, **kwargs_with_subset)


def _load_electricity_data(n_steps: int, **kwargs) -> Dict:
    """Internal function to load and process electricity data."""
    # Get split style from kwargs
    split_style = kwargs.get('split_style', 'traditional')
    
    # Check if data path is provided
    data_path = kwargs.get('data_path', 'electricity.csv')
    root_path = kwargs.get('root_path', '../dataset/TimeSeries/')
    
    # Handle root_path that is './' or '.'
    if root_path == './' or root_path == '.':
        root_path = '../dataset/TimeSeries/'
    
    # Extract filename and construct correct path
    file_name = os.path.basename(data_path)
    full_path = os.path.join(root_path, 'electricity', file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Electricity dataset not found at {full_path}")
    
    # Read the data
    df = pd.read_csv(full_path)
    
    # Remove date column if exists
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Convert to numpy array
    data = df.values.astype(np.float32)
    
    # Standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Create sliding windows
    X = create_sliding_windows(data, n_steps)
    
    # Split into train, val, test
    train_X, val_X, test_X = split_train_val_test(
        X,
        split_style=split_style,
        test_non_overlapping=kwargs.get("test_non_overlapping", False),
        n_steps=n_steps, test_stride=kwargs.get("test_stride", None),
    )
    
    return {
        'n_steps': n_steps,
        'n_features': data.shape[1],
        'train_X': train_X,
        'val_X': val_X,
        'test_X': test_X,
        'scaler': scaler
    }


def preprocess_electricity(rate: float = 0.0, n_steps: int = 96, **kwargs) -> Dict:
    """
    Load and preprocess electricity dataset in PyPOTS format.
    
    Parameters:
    -----------
    rate : float
        Initial missing rate (usually 0.0 as we apply missing masks dynamically)
    n_steps : int
        Sequence length for sliding windows
    
    Returns:
    --------
    Dict containing preprocessed data in PyPOTS format
    """
    return _get_cached_or_process('electricity', _load_electricity_data, rate, n_steps, **kwargs)

if __name__ == "__main__":
    """Test the dataset loading functions for both overlapping and non-overlapping test windows"""


    def run_dataset_test(name, loader_fn, n_steps, strides, **kwargs):
        for test_stride in strides:
            try:
                print(f"Testing {name} dataset (test_stride={test_stride})...")
                data = loader_fn(
                    rate=0.0,
                    n_steps=n_steps,
                    test_stride=test_stride,
                    **kwargs,
                )
                print(f"{name} (stride={test_stride}) - Train shape: {data['train_X'].shape}")
                print(f"{name} (stride={test_stride}) - Val shape: {data['val_X'].shape}")
                print(f"{name} (stride={test_stride}) - Test shape: {data['test_X'].shape}")
                print(f"{name} (stride={test_stride}) - Features: {data['n_features']}")
                print(f"✓ {name} dataset loaded successfully (stride={test_stride})\n")
            except Exception as e:
                print(f"✗ {name} dataset failed (stride={test_stride}): {e}\n")


    run_dataset_test("Weather", preprocess_weather, n_steps=96, strides=[1, 24, 96])
    run_dataset_test("Exchange", preprocess_exchange, n_steps=96, strides=[1, 24, 96])
    run_dataset_test("Illness", preprocess_illness, n_steps=48, strides=[1, 12, 48])

