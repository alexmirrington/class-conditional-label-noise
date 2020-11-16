"""Classes to aid model creation given config parameters."""
import argparse

import torch.nn as nn
from models import LabelNoiseRobustModel

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
        # self._backbone_factory = BackboneFactory(input_size, class_count)
        # self._estimator_factory = EstimatorFactory(class_count)

    def create(
        self, backbone: nn.Module, estimator: nn.Module, config: argparse.Namespace
    ) -> LabelNoiseRobustModel:
        """Create a model from a dataset and config."""
        return LabelNoiseRobustModel(backbone, estimator)


# TODO: different ways to use transition matrix. ie. forward, backward etc.
