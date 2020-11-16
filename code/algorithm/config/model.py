"""Configuration options and utilities for models."""
from enum import Enum


class Backbone(Enum):
    """Enum outlining available model backbones."""

    MLP = "mlp"
    RESNET18 = "resnet18"


class Estimator(Enum):
    """Enum outlining available transition matrix estimators."""

    FORWARD = "forward"
    ANCHOR = "anchor"
    FIXED = "fixed"


class RobustModel(Enum):
    """Enum outlining the types of robust models, which each use transition matrices differently."""

    FORWARD = "forward"
    BACKWARD = "backward"
