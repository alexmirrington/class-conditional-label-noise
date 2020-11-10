"""Configuration options and utilities for models."""
from enum import Enum


class Backbone(Enum):
    """Enum outlining available model backbones."""

    MLP = "mlp"


class Estimator(Enum):
    """Enum outlining available transition matrix estimators."""

    FORWARD = "forward"
