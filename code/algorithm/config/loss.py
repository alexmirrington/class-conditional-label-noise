"""Configuration options and utilities for loss correction."""
from enum import Enum


class LossCorrection(Enum):
    """Enum outlining the types of loss correction methods matrices differently."""

    FORWARD = "forward"
    BACKWARD = "backward"
    NONE = "none"
