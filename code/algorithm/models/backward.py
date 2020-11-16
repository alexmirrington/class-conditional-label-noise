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
        `(clean_posteriors, noisy_posteriors)`: The estimated posterior
        probabilities of the clean labels and noisy labels respectively.
        If the model is in eval mode, `noisy_posteriors` is set to `None`.
        `noisy_posteriors` may be a set of raw activations instead of a
        probability distribution for compatibility with `BCEWithLogitsLoss`
        or `CrossEntropyLoss`.
        """
        # Get backbone output features. The backbone features should model P(Y~|X)
        noisy_posteriors = self.backbone(features)
        # Pass features to estimator and extract clean posteriors P(Y|X) using transition matrix
        clean_posteriors = noisy_posteriors @ self.estimator.inverse_transitions
        if not self.training:
            noisy_posteriors = None  # For consistency with the forward model.
        return clean_posteriors, noisy_posteriors
