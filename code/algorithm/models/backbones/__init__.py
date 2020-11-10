"""Implementations of various model backbones."""
from .base import AbstractBackbone
from .mlp import MLPBackbone

__all__ = [AbstractBackbone.__name__, MLPBackbone.__name__]
