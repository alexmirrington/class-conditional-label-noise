"""Classes to aid model model backbone creation given config parameters."""
import argparse
from typing import Tuple

from config import Backbone
from models.backbones import AbstractBackbone, MLPBackbone


class BackboneFactory:
    """Factory class to aid model backbone creation."""

    def __init__(self, input_size: Tuple[int, ...], class_count: int):
        """Initialise a `BackboneFactory` instance.

        Params:
        -------
        `input_size`: the shape of each example from the dataset.
        `class_count`: the number of output classes for the backbone predictor.
        """
        self._factory_methods = {Backbone.MLP: self._create_mlp}
        self.input_size = input_size
        self.class_count = class_count

    def create(self, config: argparse.Namespace) -> AbstractBackbone:
        """Create a model from a dataset and config."""
        method = self._factory_methods.get(config.backbone)
        if method is None:
            raise NotImplementedError()
        return method(config)

    def _create_mlp(self, config: argparse.Namespace) -> AbstractBackbone:
        flat_input_size = 1
        for size in self.input_size:
            flat_input_size *= size
        return MLPBackbone(flat_input_size, self.class_count)
