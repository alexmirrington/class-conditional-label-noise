"""Classes to aid model creation given config parameters."""
import argparse

from config import Model
from models.base import LabelNoiseRobustModel


class ModelFactory:
    """Factory class to aid model creation given config parameters."""

    def __init__(self):
        """Initialise a `ModelFactory` instance."""
        self._factory_methods = {Model.CNN_FORWARD: ModelFactory._create_cnn_forward}

    def create(self, config: argparse.Namespace) -> LabelNoiseRobustModel:
        """Create a model from a dataset and config."""
        method = self._factory_methods.get(config.model)
        if method is None:
            raise NotImplementedError()
        return method(config)

    @staticmethod
    def _create_cnn_forward(config: argparse.Namespace) -> LabelNoiseRobustModel:
        raise NotImplementedError()
