"""Configuration options and utilities for models."""
from enum import Enum


class Backbone(Enum):
    """Enum outlining available model backbones."""

    MLP = "mlp"
    RESNET18 = "resnet18"
    SIMPLE_CNN = "simple_cnn"


class Estimator(Enum):
    """Enum outlining available transition matrix estimators."""

    ANCHOR = "anchor"
    FIXED = "fixed"
    IDENTITY = "identity"
