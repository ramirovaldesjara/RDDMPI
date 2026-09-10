"""
Dataset class for the imputation model SAITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Iterable, List, Dict, Optional

import torch
from pygrinder import mcar, fill_and_get_mask_torch
import numpy as np

from ...data.dataset.base import BaseDataset
from ...utils.missing_patterns import MissingPatternGenerator


class DatasetForSAITS(BaseDataset):
    """Dataset for models that need MIT (masked imputation task) in their training, such as SAITS.

    For more information about MIT, please refer to :cite:`du2023SAITS`.

    Parameters
    ----------
    data :
        The dataset for model input, should be a dictionary including keys as 'X' and 'y',
        or a path string locating a data file.
        If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
        which is time-series data for input, can contain missing values, and y should be array-like of shape
        [n_samples], which is classification labels of X.
        If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
        key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

    return_y :
        Whether to return labels in function __getitem__() if they exist in the given data. If `True`, for example,
        during training of classification models, the Dataset class will return labels in __getitem__() for model input.
        Otherwise, labels won't be included in the data returned by __getitem__(). This parameter exists because we
        need the defined Dataset class for all training/validating/testing stages. For those big datasets stored in h5
        files, they already have both X and y saved. But we don't read labels from the file for validating and testing
        with function _fetch_data_from_file(), which works for all three stages. Therefore, we need this parameter for
        distinction.

    file_type :
        The type of the given file if train_set and val_set are path strings.

    rate : float, in (0,1), optional
        Artificially missing rate for backward compatibility.
        If provided without mit_patterns, will use MCAR pattern with this rate.
        Deprecated - use mit_patterns instead.
        
    mit_patterns : list of dict, optional
        List of missing patterns to apply for MIT (Masked Imputation Task).
        Each pattern dict should contain 'type' and 'rate' fields.
        If not provided and rate is given, will use MCAR with the specified rate.
        If neither provided, defaults to MCAR with rate=0.2.
    """

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
        rate: Optional[float] = None,
        mit_patterns: Optional[List[Dict]] = None,
        seed: Optional[float] = None,
    ):
        super().__init__(
            data=data,
            return_X_ori=return_X_ori,
            return_X_pred=False,
            return_y=return_y,
            file_type=file_type,
        )
        
        # Handle backward compatibility
        if rate is not None and mit_patterns is None:
            # Old API: use MCAR with specified rate
            mit_patterns = [{"type": "mcar", "rate": rate}]
        elif mit_patterns is None:
            # Default: MCAR with 0.2 rate
            mit_patterns = [{"type": "mcar", "rate": 0.2}]
        
        self.rate = rate  # Keep for backward compatibility
        self.mit_patterns = mit_patterns
        self.pattern_generator = MissingPatternGenerator(seed)
        # self.pattern_generator = MissingPatternGenerator()
        
        # PM25 specific attributes
        # Check if this is PM25 dataset from data dict
        if isinstance(data, dict) and 'dataset_name' in data and data['dataset_name'] == 'pm25':
            self.is_pm25 = True
        else:
            self.is_pm25 = False
        self.training = True  # Default to training mode

    def set_pm25_mode(self):
        """Set this dataset as PM25 dataset to enable whiten_prob during training."""
        self.is_pm25 = True
        
    def train(self):
        """Set the dataset to training mode."""
        self.training = True
        
    def eval(self):
        """Set the dataset to evaluation mode."""
        self.training = False

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data according to index.

        Parameters
        ----------
        idx :
            The index to fetch the specified sample.

        Returns
        -------
        sample :
            A list contains

            index :
                The index of the sample.

            X_ori :
                Original time-series for calculating mask imputation loss.

            X :
                Time-series data with artificially missing values for model input.

            missing_mask :
                The mask records all missing values in X.

            indicating_mask :
                The mask indicates artificially missing values in X.
        """

        if self.return_X_ori:
            X = self.X[idx]
            X_ori = self.X_ori[idx]
            missing_mask = self.missing_mask[idx]
            indicating_mask = self.indicating_mask[idx]
        else:
            X_ori = self.X[idx]
            
            # Apply missing patterns
            if hasattr(X_ori, 'numpy'):
                X_ori_np = X_ori.numpy()
            else:
                X_ori_np = X_ori
                
            # Generate combined mask
            mask = self.pattern_generator.generate_combined_mask(
                X_ori_np.shape, self.mit_patterns
            )
            
            # Apply mask
            X = X_ori_np.copy()
            X[mask] = np.nan
            
            # Convert to torch and get masks
            X = torch.from_numpy(X).to(torch.float32)
            X, missing_mask = fill_and_get_mask_torch(X)
            
            # Handle X_ori
            if isinstance(X_ori, np.ndarray):
                X_ori = torch.from_numpy(X_ori).to(torch.float32)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            indicating_mask = (X_ori_missing_mask - missing_mask).to(torch.float32)

        sample = [
            torch.tensor(idx),
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ]

        if self.return_y:
            sample.append(self.y[idx].to(torch.long))

        # Apply PM25 specific whiten_prob during training
        # Only apply when return_X_ori is False (training mode)
        # When return_X_ori is True, it's validation mode
        if self.is_pm25 and not self.return_X_ori:
            # Use ImputeFormer's whiten_prob values
            whiten_probs = [0.2, 0.5, 0.8]
            p = np.random.choice(whiten_probs)
            
            # Extract current masks
            X, missing_mask, X_ori, indicating_mask = sample[1:5]
            
            # Generate whiten mask
            whiten_mask = torch.rand_like(missing_mask, dtype=torch.float32) > p
            
            # Apply whiten mask to create new missing mask
            new_missing_mask = missing_mask * whiten_mask
            
            # Calculate additionally masked positions
            additional_masked = missing_mask - new_missing_mask
            
            # Update indicating mask to include whiten masked positions
            new_indicating_mask = indicating_mask + additional_masked
            
            # Apply new mask to X
            X = X * new_missing_mask
            
            # Update sample
            sample[1] = X
            sample[2] = new_missing_mask
            sample[4] = new_indicating_mask
        
        # PM25 validation special handling
        if self.is_pm25 and self.return_X_ori:
            # For validation, evaluate on all missing positions (where ground truth exists)
            sample[4] = 1 - sample[2]  # indicating_mask = 1 - missing_mask

        return sample

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        """Fetch data with the lazy-loading strategy, i.e. only loading data from the file while requesting for samples.
        Here the opened file handle doesn't load the entire dataset into RAM but only load the currently accessed slice.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

        Returns
        -------
        sample :
            The collated data sample, a list including all necessary sample info.
        """

        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        if self.return_X_ori:
            X = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
            X_ori = torch.from_numpy(self.file_handle["X_ori"][idx]).to(torch.float32)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            X, missing_mask = fill_and_get_mask_torch(X)
            indicating_mask = (X_ori_missing_mask - missing_mask).to(torch.float32)
        else:
            X_ori = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
            X = mcar(X_ori, p=self.rate)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            X, missing_mask = fill_and_get_mask_torch(X)
            indicating_mask = (X_ori_missing_mask - missing_mask).to(torch.float32)

        sample = [torch.tensor(idx), X, missing_mask, X_ori, indicating_mask]

        # if the dataset has labels and is for training, then fetch it from the file
        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
