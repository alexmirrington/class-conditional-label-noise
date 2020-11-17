"""Model which does not use any transition matrix (ie. its outputs )."""
from typing import Tuple

import torch

from .base import LabelNoiseRobustModel


class NoTransitionModel(LabelNoiseRobustModel):
    """Create `ForwardRobustModel` instance."""

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate data through the model.

        Params:
        -------
        `features`: input features of shape (batch_size, *, n)

        Returns:
        --------
        `(posteriors, posteriors)`: Since this model does not use a transition matrix, there
        is nothing distinguishing the noisy and clean posteriors. We return a tuple containing
        both for consistency with the behaviour of other models.
        If the model is in eval mode, the second `posteriors` is set to `None`.
        """
        # Get backbone output posteriors.
        posteriors, _ = self.backbone(features)

        # For consistency with Forward and Backward models, if we are training, return the
        # posteriors also as a second element of the tuple.
        posteriors2 = None
        if self.training:
            posteriors2 = posteriors
        return posteriors, posteriors2
