"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Tuple

import torch
from torch.nn.modules.loss import _Loss

from ..functional import (
    calc_mae,
    calc_mse,
    calc_rmse,
    calc_mre,
)


class Criterion(_Loss):
    def __init__(
        self,
        lower_better: bool = True,
    ):
        """The base class for all class implementation loss functions and metrics in PyPOTS.

        Parameters
        ----------
        lower_better :
            Whether the lower value of the criterion directs to a better model performance.
            Default as True which is the case for most loss functions (e.g. MSE, Cross Entropy).
            If False, it makes that the higher value leads to a better model performance (e.g. Accuracy).

        """
        super().__init__()
        self.lower_better = lower_better

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """The criterion calculation process.

        Parameters
        ----------
        logits:
            The model outputs, predicted unnormalized logits.

        targets:
            The ground truth values.

        """
        raise NotImplementedError


class MSE(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        value = calc_mse(logits, targets, masks)
        return value


class MAE(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        value = calc_mae(logits, targets, masks)
        return value


class RMSE(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        value = calc_rmse(logits, targets, masks)
        return value


class ORTMITLoss(Criterion):
    """ORT+MIT Loss for time series imputation tasks.
    
    Combines Observed Reconstruction Task (ORT) and Masked Imputation Task (MIT) losses.
    
    Parameters
    ----------
    ORT_weight : float
        Weight for the ORT loss component.
    MIT_weight : float
        Weight for the MIT loss component.
    loss_func : Criterion
        The base loss function to use (default: MAE).
    """
    
    def __init__(
        self,
        ORT_weight: float = 1.0,
        MIT_weight: float = 1.0,
        loss_func: Criterion = None,
    ):
        super().__init__()
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.loss_func = loss_func if loss_func is not None else MAE()
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        X_ori: torch.Tensor,
        missing_mask: torch.Tensor,
        indicating_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate ORT+MIT loss.
        
        Parameters
        ----------
        reconstruction : torch.Tensor
            The reconstructed values from the model.
        X_ori : torch.Tensor
            The original complete data.
        missing_mask : torch.Tensor
            Mask indicating observed values (1) vs missing values (0).
        indicating_mask : torch.Tensor
            Mask indicating artificially masked values for MIT (1) vs others (0).
        
        Returns
        -------
        loss : torch.Tensor
            The total loss (ORT + MIT).
        ORT_loss : torch.Tensor
            The ORT loss component.
        MIT_loss : torch.Tensor
            The MIT loss component.
        """
        # Calculate ORT loss on observed values (only if ORT_weight > 0)
        if self.ORT_weight > 0:
            ORT_loss = self.ORT_weight * self.loss_func(reconstruction, X_ori, missing_mask)
        else:
            ORT_loss = torch.tensor(0.0, device=reconstruction.device)
        
        # Calculate MIT loss on artificially masked values
        MIT_loss = self.MIT_weight * self.loss_func(reconstruction, X_ori, indicating_mask)
        
        # Total loss
        loss = ORT_loss + MIT_loss
        
        return loss, ORT_loss, MIT_loss


class MRE(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        value = calc_mre(logits, targets, masks)
        return value


class ImputeFormerLoss(Criterion):
    """ImputeFormer Loss combining ORT+MIT Loss with Frequency Regularization.
    
    This loss function combines the standard ORT+MIT loss used in PyPOTS
    with the frequency regularization (F-reg) loss from the original ImputeFormer paper.
    
    Parameters
    ----------
    ORT_weight : float
        Weight for the ORT loss component.
    MIT_weight : float
        Weight for the MIT loss component.
    f1_loss_weight : float
        Weight for the frequency regularization loss (default: 0.01).
    loss_func : Criterion
        The base loss function to use (default: MAE).
    """
    
    def __init__(
        self,
        ORT_weight: float = 1.0,
        MIT_weight: float = 1.0,
        f1_loss_weight: float = 0.01,
        loss_func: Criterion = None,
    ):
        super().__init__()
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.f1_loss_weight = f1_loss_weight
        self.loss_func = loss_func if loss_func is not None else MAE()
        
    def _frequency_regularization(
        self,
        reconstruction: torch.Tensor,
        X_ori: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate frequency regularization loss using FFT.
        
        Parameters
        ----------
        reconstruction : torch.Tensor
            The reconstructed values from the model.
        X_ori : torch.Tensor
            The original complete data.
        missing_mask : torch.Tensor
            Mask indicating observed values (1) vs missing values (0).
            
        Returns
        -------
        f_reg_loss : torch.Tensor
            The frequency regularization loss.
        """
        # Replace masked values with reconstructed values
        y_tilde = torch.where(missing_mask.bool(), X_ori, reconstruction)
        
        # Apply FFT
        y_tilde_fft = torch.fft.fftn(y_tilde, dim=[1, 2])  # FFT on time and feature dimensions
        
        # Flatten batch dimensions
        batch_size = y_tilde_fft.shape[0]
        y_tilde_fft_flat = y_tilde_fft.reshape(batch_size, -1)
        
        # Calculate F-reg loss
        f_reg_loss = torch.mean(torch.sum(torch.abs(y_tilde_fft_flat), dim=1) / y_tilde_fft_flat.shape[1])
        
        return f_reg_loss
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        X_ori: torch.Tensor,
        missing_mask: torch.Tensor,
        indicating_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate ImputeFormer loss (ORT+MIT+F-reg).
        
        Parameters
        ----------
        reconstruction : torch.Tensor
            The reconstructed values from the model.
        X_ori : torch.Tensor
            The original complete data.
        missing_mask : torch.Tensor
            Mask indicating observed values (1) vs missing values (0).
        indicating_mask : torch.Tensor
            Mask indicating artificially masked values for MIT (1) vs others (0).
        
        Returns
        -------
        loss : torch.Tensor
            The total loss (ORT + MIT + F-reg).
        ORT_loss : torch.Tensor
            The ORT loss component.
        MIT_loss : torch.Tensor
            The MIT loss component.
        """
        # Calculate ORT loss on observed values
        if self.ORT_weight > 0:
            ORT_loss = self.ORT_weight * self.loss_func(reconstruction, X_ori, missing_mask)
        else:
            ORT_loss = torch.tensor(0.0, device=reconstruction.device)
        
        # Calculate MIT loss on artificially masked values
        MIT_loss = self.MIT_weight * self.loss_func(reconstruction, X_ori, indicating_mask)
        
        # Calculate frequency regularization loss
        f_reg_loss = self.f1_loss_weight * self._frequency_regularization(reconstruction, X_ori, missing_mask)
        
        # Total loss
        loss = ORT_loss + MIT_loss + f_reg_loss
        
        return loss, ORT_loss, MIT_loss


class CrossEntropy(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.nn.functional.cross_entropy(logits, targets)
        return value


class NLL(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.nn.functional.nll_loss(log_probs, targets)
        return value
