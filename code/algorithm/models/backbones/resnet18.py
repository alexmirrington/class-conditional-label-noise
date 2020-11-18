"""A resnet18 cnn model."""
import torch
import torchvision.models as models
from torch import nn

from .base import AbstractBackbone


class Resnet18Backbone(AbstractBackbone):
    """Resnet18 backbone model."""

    def __init__(self, channels: int, class_count: int) -> None:
        """Create a `Resnet18Backbone` instance."""
        super().__init__(channels, class_count)
        self.resnet18 = models.resnet18()
        self.resnet18.conv1 = nn.Conv2d(
            channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet18.fc = nn.Linear(512, class_count)
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
        features = self.resnet18(features)
        output = self.sm(features)
        return output, features
