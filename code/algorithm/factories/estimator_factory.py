"""Classes to aid label-noise estimator model creation."""
import argparse

from config import Estimator
from models.estimators import AbstractEstimator, ForwardEstimator


class EstimatorFactory:
    """Factory class to aid label-noise estimator model creation."""

    def __init__(self, class_count: int):
        """Initialise a `EstimatorFactory` instance.

        Params:
        -------
        `class_count`: the number of output classes for the estimator model.
        """
        self._factory_methods = {Estimator.FORWARD: self._create_forward}
        self.class_count = class_count

    def create(self, config: argparse.Namespace) -> AbstractEstimator:
        """Create a model from a dataset and config."""
        method = self._factory_methods.get(config.estimator)
        if method is None:
            raise NotImplementedError()
        return method(config)

    def _create_forward(self, config: argparse.Namespace) -> AbstractEstimator:
        return ForwardEstimator(self.class_count, config.freeze_estimator)
