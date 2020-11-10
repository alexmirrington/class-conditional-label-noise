"""Classes to aid model creation given config parameters."""
import argparse
from typing import Tuple

from models import LabelNoiseRobustModel

from .backbone_factory import BackboneFactory
from .estimator_factory import EstimatorFactory


class ModelFactory:
    """Factory class to aid model creation given config parameters."""

    def __init__(self, input_size: Tuple[int, ...], class_count: int):
        """Initialise a `ModelFactory` instance.

        Params:
        -------
        `input_size`: the shape of each example from the dataset.
        `class_count`: the number of output classes for the backbone predictor.
        """
        self._backbone_factory = BackboneFactory(input_size, class_count)
        self._estimator_factory = EstimatorFactory(class_count)

    def create(self, config: argparse.Namespace) -> LabelNoiseRobustModel:
        """Create a model from a dataset and config."""
        backbone = self._backbone_factory.create(config)
        estimator = self._estimator_factory.create(config)
        return LabelNoiseRobustModel(backbone, estimator)
