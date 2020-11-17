"""Classes to aid label-noise estimator model creation."""
import argparse

import torch
import torch.nn as nn
from config import Dataset, Estimator
from models.estimators import AbstractEstimator, AnchorPointEstimator, FixedEstimator
from torch.utils.data import DataLoader


class EstimatorFactory:
    """Factory class to aid label-noise estimator model creation."""

    def __init__(self, class_count: int):
        """Initialise a `EstimatorFactory` instance.

        Params:
        -------
        `class_count`: the number of output classes for the estimator model.
        """
        self._factory_methods = {
            Estimator.ANCHOR: self._create_anchor,
            Estimator.FIXED: self._create_fixed,
            Estimator.IDENTITY: self._create_identity,
        }
        self.class_count = class_count

    def create(
        self,
        config: argparse.Namespace,
        pretrained_backbone: nn.Module = None,
        samples: DataLoader = None,
    ) -> AbstractEstimator:
        """Create a model from a dataset and config."""
        method = self._factory_methods.get(config.estimator)
        if method is None:
            raise NotImplementedError()
        return method(config, pretrained_backbone, samples)

    def _create_anchor(
        self, config: argparse.Namespace, pretrained_backbone: nn.Module, samples: DataLoader
    ) -> AbstractEstimator:
        if pretrained_backbone is None or samples is None:
            raise ValueError(
                "Initialising a transition matrix using the anchor point method "
                "requires a trained classifier and samples for which to produce noisy "
                "posteriors."
            )
        return AnchorPointEstimator(
            pretrained_backbone,
            samples,
            config.device,
            self.class_count,
            config.anchor_outlier_threshold,
            config.freeze_estimator
        )

    def _create_fixed(self, config: argparse.Namespace, *args, **kwargs) -> AbstractEstimator:
        if config.dataset == Dataset.MNIST_FASHION_05:
            given_matrix = torch.tensor(
                [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]], requires_grad=False
            ).T
        elif config.dataset == Dataset.MNIST_FASHION_06:
            given_matrix = torch.tensor(
                [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]], requires_grad=False
            ).T
        else:
            raise ValueError(f"{config.dataset.value} does not have a given transition matrix.")

        return FixedEstimator(
            self.class_count, given_matrix, config.device, config.freeze_estimator
        )

    def _create_identity(self, *args, **kwargs) -> None:
        raise NotImplementedError()
