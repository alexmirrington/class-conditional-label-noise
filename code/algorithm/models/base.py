"""Base classes for label-noise-robust models."""
from typing import Tuple

import torch
from torch import nn


class LabelNoiseRobustModel(nn.Module):
    """Base class for all label-noise-robust models."""

    def __init__(self, backbone: nn.Module, estimator: nn.Module) -> None:
        """Create a `LabelNoiseRobustModel` instance."""
        super().__init__()
        self.backbone = backbone
        self.estimator = estimator

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate data through the model.

        Params:
        -------
        `features`: input features of shape (batch_size, *, n)

        Returns:
        --------
        `(clean_posteriors, noisy_posteriors)`: The estimated posterior
        probabilities of the clean labels and noisy labels respectively.
        If the model is in eval mode, `noisy_posteriors` is set to `None`.
        `noisy_posteriors` may be a set of raw activations instead of a
        probability distribution for compatibility with `BCEWithLogitsLoss`
        or `CrossEntropyLoss`.
        """
        # Get backbone output features. The backbone features should model P(Y|X)
        clean_posteriors = self.backbone(features)
        # Pass features to estimator if training, otherwise return only the
        # backbone features. The estimator should model P(Y~|X)
        noisy_posteriors = None
        if self.training:
            noisy_posteriors = self.estimator(clean_posteriors)
        return clean_posteriors, noisy_posteriors
