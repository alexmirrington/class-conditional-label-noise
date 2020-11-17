"""Forward label noise robust model."""
from typing import Tuple

import torch

from .base import LabelNoiseRobustModel


class ForwardRobustModel(LabelNoiseRobustModel):
    """Create `ForwardRobustModel` instance."""

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
        # Get backbone output features. The backbone features should model P(Y|X)
        clean_posteriors, clean_activations = self.backbone(features)
        # Pass features to estimator. The estimator should model P(Y~|X)
        noisy_activations = clean_posteriors @ self.estimator.transitions
        return clean_activations, noisy_activations
