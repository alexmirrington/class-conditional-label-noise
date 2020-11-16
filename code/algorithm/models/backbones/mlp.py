"""A simple 3-layer MLP backbone with ReLU activations."""
import torch
from torch import nn

from .base import AbstractBackbone


class MLPBackbone(AbstractBackbone):
    """Base class for all label-noise-robust models."""

    def __init__(self, input_size: int, class_count: int) -> None:
        """Create a `MLPBackbone` instance."""
        super().__init__(input_size, class_count)
        hidden_size = (input_size + class_count) // 2
        self.fc_0 = nn.Linear(input_size, hidden_size)
        self.act_0 = nn.ReLU()
        self.dropout_0 = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(hidden_size, hidden_size)
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_size, class_count)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Propagate data through the model.

        Params:
        -------
        `features`: input features of shape `(batch_size, input_size)` or `(batch_size, *)`
        where `*` flattens to the same dimension as `input_size`.

        Returns:
        --------
        `output`: output of shape `(batch_size, class_count)`.
        """
        features = torch.flatten(features, start_dim=1)
        features = self.fc_0(features)
        features = self.act_0(features)
        features = self.dropout_0(features)
        features = self.fc_1(features)
        features = self.act_1(features)
        features = self.fc_2(features)
        output = self.sm(features)
        return output
