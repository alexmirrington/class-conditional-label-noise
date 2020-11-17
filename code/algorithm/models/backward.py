"""Backward label noise robust model."""
from typing import Tuple

import torch

from .base import LabelNoiseRobustModel


class BackwardRobustModel(LabelNoiseRobustModel):
    """Create `BackwardRobustModel` instance."""

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate data through the model.

        Params:
        -------
        `features`: input features of shape (batch_size, *, n)

        Returns:
        --------
        `(clean_activations, noisy_activations)`: The estimated posterior
        probabilities of the clean labels and noisy labels respectively.
        """
        # Get backbone output features. The backbone features should model P(Y~|X)
        noisy_posteriors, noisy_activations = self.backbone(features)
        # Pass features to estimator and extract clean posteriors P(Y|X) using transition matrix
        clean_activations = noisy_posteriors @ self.estimator.inverse_transitions
        return clean_activations, noisy_activations
