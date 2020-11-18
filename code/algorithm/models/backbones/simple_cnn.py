"""A small cnn model."""
import torch
from torch import nn

from .base import AbstractBackbone


class SimpleCNNBackbone(AbstractBackbone):
    """Resnet18 backbone model."""

    def __init__(self, input_size: int, class_count: int) -> None:
        """Create a `Resnet18Backbone` instance."""
        channels = input_size[0]
        input_dim = input_size[1]

        # Input size looks like (channels, dim, dim)
        super().__init__(channels, class_count)
        self.conv_0 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.act_0 = nn.ReLU()
        self.pool_0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc_0 = nn.Linear(input_dim * input_dim * 64 // 16, 512)
        self.act_2 = nn.ReLU()
        self.dropout_0 = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(512, class_count)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Propagate data through the model.

        Params:
        -------
        `features`: input features of shape (batch_size, channels, H, W)

        Returns:
        --------
        `output`: output of shape `(batch_size, class_count)`.
        `features`: features before the final softmax layer of shape `(batch_size, class_count)`.
        """
        features = self.pool_0(self.act_0(self.conv_0(features)))
        features = self.pool_1(self.act_1(self.conv_1(features)))
        features = self.flatten(features)
        features = self.fc_0(features)
        features = self.act_2(features)
        features = self.dropout_0(features)
        features = self.fc_1(features)
        output = self.sm(features)
        return output, features
