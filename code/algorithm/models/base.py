"""Base classes for label-noise-robust models."""
from typing import Tuple

import torch
from torch import nn


class LabelNoiseRobustModel(nn.Module):
    """Base class for all label-noise-robust models."""

    def __init__(self, backbone: nn.Module, estimator: nn.Module = None) -> None:
        """Create a `LabelNoiseRobustModel` instance."""
        super().__init__()
        self.backbone = backbone
        # TODO: add option for no estimator/identity estimator
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
        """
        ...
