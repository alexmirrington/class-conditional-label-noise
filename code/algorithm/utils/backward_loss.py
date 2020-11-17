"""An implementation of backward-corrected NLL Loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.estimators.base import AbstractEstimator


class BackwardNLLLoss(nn.Module):
    """Initialise `BackwardNLLLoss` instance."""

    def __init__(self, estimator: AbstractEstimator, reduction: str = "mean"):
        """Initialise `BackwardNLLLoss` instance."""
        super().__init__()
        self.estimator = estimator
        self.reduction = reduction

    def forward(self, preds, targets):
        """Return the backward corrected cross entropy loss for the labels."""
        num_classes = preds.size()[-1]
        preds = preds / torch.sum(preds, dim=-1, keepdim=True)
        preds = torch.clamp(preds, 1e-8, 1 - 1e-8)
        log_preds = torch.log(preds)
        one_hot_targets = F.one_hot(targets, num_classes=num_classes).type(torch.float32)
        reweighted_targets = one_hot_targets @ self.estimator.inverse_transitions
        unreduced_loss = -torch.sum(reweighted_targets * log_preds, dim=-1)
        if torch.isnan(preds[0][0]):
            exit()
        # print(f"{preds=}")
        # print(f"{targets=}")
        # print(f"{self.estimator.transitions=}")
        # print(f"{self.estimator.inverse_transitions=}")
        # print(f"{log_preds=}")
        # print(f"{reweighted_targets=}")
        # print(f"{unreduced_loss=}")
        loss = self._reduce_loss(unreduced_loss, self.reduction)
        # print(f"{loss=}")
        return loss

    def _reduce_loss(self, loss, reduction="mean"):
        return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss
