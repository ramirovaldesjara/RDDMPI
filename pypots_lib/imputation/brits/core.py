"""
The core wrapper assembles the submodules of BRITS imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch

from ...nn.modules import ModelCore
from ...nn.modules.brits import BackboneBRITS
from ...nn.modules.loss import Criterion, ORTMITLoss


class _BRITS(ModelCore):
    """model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.model = BackboneBRITS(
            n_steps,
            n_features,
            rnn_hidden_size,
            training_loss,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
        ) = self.model(inputs)

        results = {
            "imputation": imputed_data,
            "consistency_loss": consistency_loss,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction": (f_reconstruction + b_reconstruction) / 2,
            "f_reconstruction": f_reconstruction,
            "b_reconstruction": b_reconstruction,
        }

        if calc_criterion:
            # BRITS uses its own loss calculation with reconstruction_loss and consistency_loss
            loss = reconstruction_loss + consistency_loss
            
            if self.training:  # if in the training mode (the training stage)
                results["loss"] = loss
                results["consistency_loss"] = consistency_loss
                results["reconstruction_loss"] = reconstruction_loss
            else:  # if in the eval mode (the validation stage)
                # For validation, calculate metric on reconstructed vs original values
                # Use the average of forward and backward reconstructions
                reconstruction = (results["f_reconstruction"] + results["b_reconstruction"]) / 2
                
                # Get original data and mask from inputs
                if "X_ori" in inputs and "indicating_mask" in inputs:
                    # If X_ori is provided (for validation)
                    X_ori = inputs["X_ori"]
                    indicating_mask = inputs["indicating_mask"]
                    results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)
                else:
                    # Fallback: use forward X as ground truth with missing_mask
                    X_forward = inputs["forward"]["X"]
                    missing_mask = inputs["forward"]["missing_mask"]
                    results["metric"] = self.validation_metric(reconstruction, X_forward, missing_mask)

        return results
