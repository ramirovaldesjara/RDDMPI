"""
The core wrapper assembles the submodules of T1 imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from omegaconf import DictConfig

from ...nn.modules import ModelCore
from ...nn.modules.t1 import BackboneT1Imputation


class _T1(ModelCore):
    def __init__(self, cfg_dict: dict, training_loss, validation_metric):
        super().__init__()
        
        # Convert dict to DictConfig
        self.cfg = DictConfig(cfg_dict)
        
        # Initialize T1 backbone (imputation only)
        self.model = BackboneT1Imputation(self.cfg)
        self.training_loss = training_loss
        self.validation_metric = validation_metric if validation_metric.__class__.__name__ != "Criterion" else training_loss
        
    def forward(self, inputs: dict, calc_criterion: bool = False) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        
        # Call T1's forward method (simplified interface)
        # Use PyPOTS mask as-is (1=observed, 0=missing)
        reconstruction,_ = self.model(X, missing_mask)
        
        # Keep original values for observed parts
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }
        
        if calc_criterion:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            if self.training:
                # training loss receives reconstruction, X_ori, missing_mask, indicating_mask
                loss, ORT_loss, MIT_loss = self.training_loss(reconstruction, X_ori, missing_mask, indicating_mask)
                results.update({
                    "ORT_loss": ORT_loss,
                    "MIT_loss": MIT_loss,
                    "loss": loss
                })
            else:
                # validation metric receives reconstruction, X_ori, indicating_mask
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)
                
        return results