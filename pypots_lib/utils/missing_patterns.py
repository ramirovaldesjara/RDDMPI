"""
Missing pattern generation utilities for PyPOTS.

This module provides various missing pattern generators that can create
realistic missing data patterns for time series imputation experiments.
"""

import numpy as np
from typing import List, Dict, Union, Tuple, Optional
import warnings


class MissingPatternGenerator:
    """Generate and combine various missing patterns for time series data.
    
    This class supports multiple missing patterns including:
    - MCAR (Missing Completely At Random)
    - Block missing
    - Sequence missing 
    - MAR (Missing At Random)
    - MNAR (Missing Not At Random)
    
    Patterns can be combined using OR operation to create complex missing scenarios.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the pattern generator.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        """
        if seed is None:
            self.rng = np.random
            self.rng_mode = "global_numpy_rng"
            state = np.random.get_state()
            state_preview = state[1][:5]
        else:
            self.rng = np.random.RandomState(seed)
            self.rng_mode = "private_randomstate"
            state = self.rng.get_state()
            state_preview = state[1][:5]

        # print(
        #     f"[MissingPatternGenerator] rng_mode={self.rng_mode}, "
        #     f"seed={seed}, state_preview={state_preview}"
        # )
    
    def generate_combined_mask(self, 
                             shape: Tuple[int, ...], 
                             patterns: List[Dict]) -> np.ndarray:
        """Generate combined missing mask from multiple patterns.
        
        Parameters
        ----------
        shape : tuple
            Shape of the data (n_samples, n_steps, n_features)
        patterns : list of dict
            List of pattern configurations
            
        Returns
        -------
        np.ndarray
            Boolean mask where True indicates missing values
        """
        combined_mask = np.zeros(shape, dtype=bool)
        
        for pattern in patterns:
            mask = self._generate_single_pattern(shape, pattern)
            combined_mask = combined_mask | mask
        
        return combined_mask
    
    def _generate_single_pattern(self, shape: Tuple, pattern: Dict) -> np.ndarray:
        """Generate mask for a single pattern."""
        pattern_type = pattern.get('type')
        
        if pattern_type == 'mcar':
            return self._generate_mcar_mask(shape, pattern['rate'])
        elif pattern_type == 'block':
            return self._generate_block_mask(shape, pattern)
        elif pattern_type == 'sequence':
            return self._generate_sequence_mask(shape, pattern)
        elif pattern_type == 'mar':
            return self._generate_mar_mask(shape, pattern)
        elif pattern_type == 'mnar_x':
            return self._generate_mnar_x_mask(shape, pattern)
        elif pattern_type == 'mnar_t':
            return self._generate_mnar_t_mask(shape, pattern)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    def _generate_mcar_mask(self, shape: Tuple, rate: float) -> np.ndarray:
        """Generate MCAR (completely random) missing mask."""
        return self.rng.rand(*shape) < rate
    
    def _generate_sequence_mask(self, shape: Tuple, pattern: Dict) -> np.ndarray:
        """Generate sequence missing mask with consecutive missing values."""
        mask = np.zeros(shape, dtype=bool)
        
        if len(shape) == 3:
            n_samples, seq_len, n_features = shape
        else:
            raise ValueError(f"Expected 3D shape, got {len(shape)}D")
        
        # Handle sequence length: fixed or range
        seq_length = pattern.get('seq_len', 10)
        if isinstance(seq_length, dict) and 'range' in seq_length:
            min_len, max_len = seq_length['range']
        elif isinstance(seq_length, (int, float)):
            min_len = max_len = int(seq_length)
        else:
            raise ValueError(f"Invalid seq_len format: {seq_length}")
        
        # Calculate target missing count
        total_elements = n_samples * seq_len * n_features
        target_missing = int(total_elements * pattern['rate'])
        current_missing = 0
        
        # Generate sequences
        max_attempts = target_missing * 10
        attempts = 0
        
        while current_missing < target_missing * 0.95 and attempts < max_attempts:  # 95% tolerance
            # Random sequence length
            length = self.rng.randint(min_len, max_len + 1)
            
            # Random position
            sample_idx = self.rng.randint(0, n_samples)
            feature_idx = self.rng.randint(0, n_features)
            
            # Check if sequence fits
            if seq_len >= length:
                start_idx = self.rng.randint(0, seq_len - length + 1)
                
                # Apply mask
                mask[sample_idx, start_idx:start_idx+length, feature_idx] = True
                current_missing = mask.sum()
            
            attempts += 1
        
        if attempts >= max_attempts:
            warnings.warn(f"Max attempts reached. Achieved {current_missing/total_elements:.2%} "
                         f"missing rate instead of {pattern['rate']:.2%}")
        
        return mask
    
    def _generate_block_mask(self, shape: Tuple, pattern: Dict) -> np.ndarray:
        """Generate block missing mask with rectangular missing regions."""
        mask = np.zeros(shape, dtype=bool)
        
        if len(shape) == 3:
            n_samples, seq_len, n_features = shape
        else:
            raise ValueError(f"Expected 3D shape, got {len(shape)}D")
        
        # Handle block size: fixed or range
        block_size = pattern.get('block_size', [5, 5])
        
        if isinstance(block_size, dict) and 'range' in block_size:
            # Range format: {"range": [[h_min, h_max], [w_min, w_max]]}
            h_min, h_max = block_size['range'][0]
            w_min, w_max = block_size['range'][1]
        elif isinstance(block_size, list) and len(block_size) == 2:
            # Fixed size: [height, width]
            h_min = h_max = block_size[0]
            w_min = w_max = block_size[1]
        else:
            raise ValueError(f"Invalid block_size format: {block_size}")
        
        # Calculate target missing count
        total_elements = n_samples * seq_len * n_features
        target_missing = int(total_elements * pattern['rate'])
        current_missing = 0
        
        # Generate blocks
        max_attempts = target_missing * 10
        attempts = 0
        
        while current_missing < target_missing * 0.95 and attempts < max_attempts:
            # Random block size
            height = self.rng.randint(h_min, min(h_max + 1, seq_len + 1))
            width = self.rng.randint(w_min, min(w_max + 1, n_features + 1))
            
            # Random position
            sample_idx = self.rng.randint(0, n_samples)
            
            # Check if block fits
            if seq_len >= height and n_features >= width:
                start_t = self.rng.randint(0, seq_len - height + 1)
                start_f = self.rng.randint(0, n_features - width + 1)
                
                # Apply block
                mask[sample_idx, start_t:start_t+height, start_f:start_f+width] = True
                current_missing = mask.sum()
            
            attempts += 1
        
        if attempts >= max_attempts:
            warnings.warn(f"Max attempts reached. Achieved {current_missing/total_elements:.2%} "
                         f"missing rate instead of {pattern['rate']:.2%}")
        
        return mask
    
    def _generate_mar_mask(self, shape: Tuple, pattern: Dict) -> np.ndarray:
        """Generate MAR (Missing At Random) mask based on observed variables."""
        if len(shape) == 3:
            n_samples, seq_len, n_features = shape
        else:
            raise ValueError(f"Expected 3D shape, got {len(shape)}D")
        
        mask = np.zeros(shape, dtype=bool)
        obs_rate = pattern.get('obs_rate', 0.1)
        
        # Select observed features
        n_obs_features = max(1, int(n_features * obs_rate))
        obs_features = self.rng.choice(n_features, n_obs_features, replace=False)
        
        # For each sample, create missing pattern based on observed features
        for i in range(n_samples):
            # Use observed features to determine missingness
            obs_values = self.rng.randn(seq_len, n_obs_features)
            
            # Simple logistic model
            weights = self.rng.randn(n_obs_features)
            logits = obs_values @ weights
            probs = 1 / (1 + np.exp(-logits))
            
            # Scale probabilities to achieve target rate
            probs = probs * pattern['rate'] / probs.mean()
            probs = np.clip(probs, 0, 1)
            
            # Apply to non-observed features
            for j in range(n_features):
                if j not in obs_features:
                    mask[i, :, j] = self.rng.rand(seq_len) < probs
        
        return mask
    
    def _generate_mnar_x_mask(self, shape: Tuple, pattern: Dict) -> np.ndarray:
        """Generate MNAR mask based on value magnitude (extreme values missing)."""
        # This requires actual data values, so we'll create a placeholder
        # In practice, this should be called with actual data
        warnings.warn("MNAR-x pattern requires actual data values. "
                     "Using random data for demonstration.")
        
        # Generate synthetic data
        data = self.rng.randn(*shape)
        mask = np.zeros(shape, dtype=bool)
        
        offset = pattern.get('offset', 1.0)
        
        # Mask extreme values
        for i in range(shape[-1]):  # For each feature
            feature_data = data[:, :, i]
            mean = feature_data.mean()
            std = feature_data.std()
            threshold = mean + offset * std
            
            # Mask values above threshold
            feature_mask = feature_data > threshold
            
            # Adjust to match target rate
            current_rate = feature_mask.mean()
            if current_rate > 0:
                scale = pattern['rate'] / current_rate
                if scale < 1:
                    # Randomly keep only some masked values
                    keep_mask = self.rng.rand(*feature_mask.shape) < scale
                    feature_mask = feature_mask & keep_mask
            
            mask[:, :, i] = feature_mask
        
        return mask
    
    def _generate_mnar_t_mask(self, shape: Tuple, pattern: Dict) -> np.ndarray:
        """Generate MNAR mask based on temporal patterns."""
        if len(shape) == 3:
            n_samples, seq_len, n_features = shape
        else:
            raise ValueError(f"Expected 3D shape, got {len(shape)}D")
        
        mask = np.zeros(shape, dtype=bool)
        
        cycle = pattern.get('cycle', 24)
        pos = pattern.get('pos', 0)
        scale = pattern.get('scale', 3)
        
        # Create temporal intensity function
        t = np.arange(seq_len)
        intensity = np.exp(scale * np.sin(2 * np.pi * (t - pos) / cycle))
        intensity = intensity / intensity.sum()  # Normalize
        
        # Scale to achieve target rate
        intensity = intensity * pattern['rate'] * seq_len
        
        # Apply to each sample and feature
        for i in range(n_samples):
            for j in range(n_features):
                mask[i, :, j] = self.rng.rand(seq_len) < intensity
        
        return mask
    
    @staticmethod
    def calculate_actual_missing_rate(mask: np.ndarray) -> float:
        """Calculate the actual missing rate from a mask."""
        return mask.sum() / mask.size
    
    @staticmethod
    def apply_mask_to_data(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply missing mask to data."""
        masked_data = data.copy()
        masked_data[mask] = np.nan
        return masked_data


def validate_pattern_config(patterns: List[Dict]) -> None:
    """Validate pattern configuration.
    
    Parameters
    ----------
    patterns : list of dict
        Pattern configurations to validate
        
    Raises
    ------
    ValueError
        If configuration is invalid
    """
    if not isinstance(patterns, list):
        raise ValueError("Patterns must be a list")
    
    for i, pattern in enumerate(patterns):
        if not isinstance(pattern, dict):
            raise ValueError(f"Pattern {i} must be a dictionary")
        
        if 'type' not in pattern:
            raise ValueError(f"Pattern {i} must have 'type' field")
        
        if 'rate' not in pattern and 'rates' not in pattern:
            raise ValueError(f"Pattern {i} must have 'rate' or 'rates' field")
        
        # Type-specific validation
        pattern_type = pattern['type']
        
        if pattern_type == 'sequence' and 'seq_len' in pattern:
            seq_len = pattern['seq_len']
            if isinstance(seq_len, dict) and 'range' not in seq_len:
                raise ValueError(f"Pattern {i}: dict seq_len must have 'range' key")
            elif isinstance(seq_len, dict):
                if len(seq_len['range']) != 2:
                    raise ValueError(f"Pattern {i}: seq_len range must have 2 values")
        
        if pattern_type == 'block' and 'block_size' in pattern:
            block_size = pattern['block_size']
            if isinstance(block_size, dict) and 'range' not in block_size:
                raise ValueError(f"Pattern {i}: dict block_size must have 'range' key")
            elif isinstance(block_size, dict):
                if len(block_size['range']) != 2:
                    raise ValueError(f"Pattern {i}: block_size range must have 2 dimensions")
                for dim in block_size['range']:
                    if len(dim) != 2:
                        raise ValueError(f"Pattern {i}: each dimension range must have 2 values")