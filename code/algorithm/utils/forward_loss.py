"""An implementation of forward-corrected NLL Loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.estimators.base import AbstractEstimator


class ForwardNLLLoss(nn.Module):
    """Initialise `ForwardNLLLoss` instance."""

    def __init__(self, estimator: AbstractEstimator, reduction: str = "mean"):
        """Initialise `ForwardNLLLoss` instance."""
        super().__init__()
        self.estimator = estimator
        self.reduction = reduction

    def forward(self, preds, targets):
        """Return the forward corrected cross entropy loss for the labels."""
        noisy_preds = preds @ self.estimator.transitions
        log_preds = torch.log(noisy_preds)
        # print(f"{self.estimator.transitions=}")
        # print(f"{noisy_preds=}")
        # print(f"{log_preds=}")
        return F.nll_loss(log_preds, targets, reduction=self.reduction)
