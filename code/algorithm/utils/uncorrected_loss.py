"""An implementation of uncorrected NLL Loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.estimators.base import AbstractEstimator


class UncorrectedNLLLoss(nn.Module):
    """Initialise `UncorrectedNLLLoss` instance."""

    def __init__(self, estimator: AbstractEstimator, reduction: str = "mean"):
        """Initialise `UncorrectedNLLLoss` instance."""
        super().__init__()
        self.estimator = estimator
        self.reduction = reduction

    def forward(self, preds, targets):
        """Return the forward corrected cross entropy loss for the labels."""
        return F.nll_loss(torch.log(preds), targets, reduction=self.reduction)
