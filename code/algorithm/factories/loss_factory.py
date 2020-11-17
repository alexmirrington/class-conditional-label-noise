"""Classes to aid model creation given config parameters."""
import argparse

import torch
import torch.nn as nn
from config import LossCorrection
from utils.backward_loss import BackwardNLLLoss
from utils.forward_loss import ForwardNLLLoss
from utils.uncorrected_loss import UncorrectedNLLLoss


class LossFactory:
    """Factory class to aid loss function creation given config parameters."""

    def __init__(self):
        """Initialise a `LossFactory` instance.

        Params:
        -------
        `input_size`: the shape of each example from the dataset.
        `class_count`: the number of output classes for the backbone predictor.
        """
        self._factory_methods = {
            LossCorrection.FORWARD: self._create_forward,
            LossCorrection.BACKWARD: self._create_backward,
            LossCorrection.NONE: self._create_none,
        }

    def create(self, estimator: nn.Module, config: argparse.Namespace) -> torch.nn.Module:
        """Create a loss function from an estimator."""
        method = self._factory_methods.get(config.loss_correction)
        if method is None:
            raise NotImplementedError()
        return method(estimator)

    def _create_forward(self, estimator: nn.Module) -> torch.nn.Module:
        """Create a loss function from a config."""
        return ForwardNLLLoss(estimator)

    def _create_backward(self, estimator: nn.Module) -> torch.nn.Module:
        """Create a loss function from a config."""
        return BackwardNLLLoss(estimator)

    def _create_none(self, estimator: nn.Module) -> torch.nn.Module:
        """Create a loss function from a config."""
        return UncorrectedNLLLoss(estimator)
