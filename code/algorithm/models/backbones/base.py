"""Base classes for backbone modules."""
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class AbstractBackbone(nn.Module, ABC):
    """Base class for all backbone modules."""

    def __init__(self, input_size: int, class_count: int, **kwargs: Any) -> None:
        """Create an `AbstractBackbone` instance."""
        super().__init__()
        self.input_size = input_size
        self.class_count = class_count

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Propagate data through the model.

        Params:
        -------
        `features`: input features of shape `(batch_size, *, input_size)`

        Returns:
        --------
        `output`: output of shape `(batch_size, class_count)`.
        """
