"""Base classes for estimator modules."""
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class AbstractEstimator(nn.Module, ABC):
    """Base class for all estimator modules."""

    def __init__(self, class_count: int, **kwargs: Any) -> None:
        """Create an `AbstractEstimator` instance."""
        super().__init__()
        self.class_count = class_count

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Propagate data through the model.

        Params:
        -------
        `features`: input of shape (batch_size, class_count)

        Returns:
        --------
        `output`: output of shape (batch_size, class_count)
        """
