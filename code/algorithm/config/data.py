"""Configuration options and utilities for datasets."""
from enum import Enum


class Dataset(Enum):
    """Enum outlining available datasets."""

    MNIST_FASHION_05 = "mnist_fashion_05"
    MNIST_FASHION_06 = "mnist_fashion_06"
    CIFAR = "cifar"
