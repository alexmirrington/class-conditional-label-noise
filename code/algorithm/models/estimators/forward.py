"""Implementation of a forward-method transition matrix estimator."""
import torch
from torch import nn

from .base import AbstractEstimator


class ForwardEstimator(AbstractEstimator):
    """A forward-method transition matrix estimator."""

    def __init__(self, class_count: int, frozen: bool = False) -> None:
        """Create a `ForwardEstimator` instance."""
        super().__init__(class_count)
        self.transitions = nn.Parameter(torch.empty((class_count, class_count)))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the module's parameters."""
        # Intiialise transition matrix as the identity matrix, i.e. we assume
        # there is no class-conditional label noise
        self.transitions.data = torch.eye(self.class_count)
        # self.transitions.data = torch.nn.init.kaiming_uniform_(self.transitions)
        # self.transitions.data = torch.ones_like(self.transitions) / self.class_count

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Propagate data through the model.

        Params:
        -------
        `features`: input of shape (batch_size, class_count)

        Returns:
        --------
        `output`: output of shape (batch_size, class_count)
        """
        return torch.matmul(features, self.transitions)
