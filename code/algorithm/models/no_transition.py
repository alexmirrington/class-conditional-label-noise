"""Model which does not use any transition matrix (ie. its outputs )."""
from typing import Tuple

import torch

from .base import LabelNoiseRobustModel


class NoTransitionModel(LabelNoiseRobustModel):
    """Create `NoTransitionModel` instance."""

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate data through the model.

        Params:
        -------
        `features`: input features of shape (batch_size, *, n)

        Returns:
        --------
        `(activations, activations)`: Since this model does not use a transition matrix, there
        is nothing distinguishing the noisy and clean posteriors. We return a tuple containing
        both for consistency with the behaviour of other models.
        """
        # Get backbone output posteriors.
        _, activations = self.backbone(features)
        return activations, activations
