"""Classes to aid model creation given config parameters."""
import argparse

import torch.nn as nn
from config import RobustModel
from models import BackwardRobustModel, ForwardRobustModel, LabelNoiseRobustModel

# from .backbone_factory import BackboneFactory
# from .estimator_factory import EstimatorFactory


class ModelFactory:
    """Factory class to aid model creation given config parameters."""

    def __init__(self):
        """Initialise a `ModelFactory` instance.

        Params:
        -------
        `input_size`: the shape of each example from the dataset.
        `class_count`: the number of output classes for the backbone predictor.
        """
        self._factory_methods = {
            RobustModel.FORWARD: self._create_forward,
            RobustModel.BACKWARD: self._create_backward,
        }

    def create(
        self, backbone: nn.Module, estimator: nn.Module, config: argparse.Namespace
    ) -> LabelNoiseRobustModel:
        """Create a LabelNoiseRobustModel from given backbone and estimator."""
        method = self._factory_methods.get(config.robust_type)
        if method is None:
            raise NotImplementedError()
        return method(backbone, estimator, config)

    def _create_forward(
        self, backbone: nn.Module, estimator: nn.Module, config: argparse.Namespace
    ) -> LabelNoiseRobustModel:
        """Create a model from a config."""
        return ForwardRobustModel(backbone, estimator)

    def _create_backward(
        self, backbone: nn.Module, estimator: nn.Module, config: argparse.Namespace
    ) -> LabelNoiseRobustModel:
        """Create a model from a config."""
        return BackwardRobustModel(backbone, estimator)


# TODO: different ways to use transition matrix. ie. forward, backward etc.
