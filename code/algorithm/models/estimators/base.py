"""Base classes for estimator modules."""
from abc import ABC
from typing import Any

import torch


class AbstractEstimator(ABC):
    """Base class for all estimator modules."""

    def __init__(self, class_count: int, **kwargs: Any) -> None:
        """Create an `AbstractEstimator` instance."""
        super().__init__()
        self.class_count = class_count

    def transition(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Multiply the given probabilities by the transition matrix.

        Args
        ---
        probabilities: torch.Tensor of shape (class_count).
        """
        return self.probabilities @ self.transitions

    def inverse_transition(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Multiply the given probabilities by the inverse of the transition matrix.

        Args
        ---
        probabilities: torch.Tensor of shape (class_count).
        """
        return self.probabilities @ self.inverse_transitions
