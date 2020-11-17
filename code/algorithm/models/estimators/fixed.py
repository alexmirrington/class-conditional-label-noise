"""Implementation of a fixed transition matrix which adjusts noisy posteriors for noise."""
import torch

from .base import AbstractEstimator


class FixedEstimator(AbstractEstimator):
    """For use when the transition matrix is given."""

    def __init__(
        self, class_count: int, given_matrix: torch.Tensor, device: str, frozen: bool = True
    ) -> None:
        """Create a `FixedEstimator` instance."""
        super().__init__(class_count)
        self.device = self.device
        self.transitions = torch.tensor(given_matrix).to(device)
        self.inverse_transitions = torch.inverse(self.transitions)
