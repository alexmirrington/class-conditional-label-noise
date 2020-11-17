"""Main entrypoint for model training and evaluation."""
import argparse
import hashlib
import json
import os.path
import sys
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from config import Backbone, Dataset, Estimator, RobustModel
from datasets import load_data
from factories import BackboneFactory, EstimatorFactory, ModelFactory
from loggers import JSONLLogger, Logger, StreamLogger, WandbLogger
from sklearn.metrics import accuracy_score
from termcolor import colored
from torch.utils.data import DataLoader
from utils import LabelSmoothingCrossEntropyLoss


def main(config: argparse.Namespace):
    """Train or evaluate a label-noise-robust model."""
    # Get environment info
    print(colored("environment:", attrs=["bold"]))
    print(
        f"device: {torch.cuda.get_device_name(config.device) if config.device != 'cpu' else 'cpu'}"
    )

    # Load dataset
    print(colored("dataset:", attrs=["bold"]))
    print(config.dataset)
    # Determine dataset directory
    train_data, val_data, test_data = load_data(
        os.path.join(os.path.dirname(__file__), os.path.pardir, "data"),
        config.dataset,
        config.subset,
        config.seed,
    )
    print(f"train: {[tuple(t.shape) for t in train_data.tensors]}")
    print(f"val: {[tuple(t.shape) for t in val_data.tensors]}")
    print(f"test: {[tuple(t.shape) for t in test_data.tensors]}")

    # Create loggers
    loggers = [StreamLogger(), JSONLLogger(Path(config.results_dir) / f"{config.id}.jsonl")]
    if config.wandb:
        loggers.append(WandbLogger())

    input_size = tuple(train_data.tensors[0].size()[1:])
    class_count = len(set(train_data.tensors[1].tolist()))

    if config.label_smoothing > 0 and config.label_smoothing < 1:
        criterion = LabelSmoothingCrossEntropyLoss(config.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # Create backbone
    print(colored("backbone:", attrs=["bold"]))
    backbone_factory = BackboneFactory(input_size, class_count)
    backbone = backbone_factory.create(config)
    print(backbone)

    # Perform pretraining on noisy data without the transition matrix if necessary
    if config.backbone_pretrain_epochs > 0:
        print(colored("pretraining backbone:", attrs=["bold"]))
        backbone = backbone.to(config.device)
        pretrain_backbone(
            backbone,
            train_data,
            torch.optim.SGD(backbone.parameters(), lr=1e-3),
            criterion,
            loggers,
            config,
        )

    # Estimator could be None if we don't want to use a transition matrix
    # Create transition matrix
    print(colored("estimator:", attrs=["bold"]))
    estimator_factory = EstimatorFactory(class_count)
    estimator = estimator_factory.create(
        config,
        backbone,
        DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=0),
    )
    if estimator is not None:
        print(
            f"Transition matrix to be usd by the Label Noise Robust Model:\n"
            f"{estimator.transitions=}"
        )
        wandb.run.summary["transitions"] = estimator.transitions.t()

    # Create model from backbone and estimator
    model_factory = ModelFactory()
    model = model_factory.create(backbone, estimator, config)

    # Train and evaluate model
    print(colored("training:", attrs=["bold"]))
    model = model.to(config.device)
    train(
        model,
        train_data,
        val_data,
        torch.optim.SGD(backbone.parameters(), lr=1e-3),
        torch.nn.NLLLoss(),
        loggers,
        config,
    )
    test(model, test_data, loggers, config)


def train(
    model: torch.nn.Module,
    train_data: torch.utils.data.Dataset,
    val_data: torch.utils.data.Dataset,
    optimiser: torch.optim.Optimizer,
    criterion: Optional[Callable[..., torch.Tensor]],
    loggers: Iterable[Logger],
    config: argparse.Namespace,
):
    """Train a model."""
    dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    class_count = len(set(dataloader.dataset.tensors[1].tolist()))
    model.train()
    for epoch in range(config.epochs):
        for batch, (feats, labels) in enumerate(dataloader):
            # Move data to GPU
            feats = feats.to(config.device)
            # Convert labels to one-hots if using BCEWithLogitsLoss
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                labels = F.one_hot(labels, num_classes=class_count).type(torch.float32)
            labels = labels.to(config.device)
            optimiser.zero_grad()
            clean_activations, noisy_activations = model(feats)
            loss = criterion(torch.log(noisy_activations), labels)
            loss.backward()

            optimiser.step()

            preds = torch.argmax(noisy_activations, dim=-1).cpu().numpy()
            if batch % config.log_step == config.log_step - 1 or batch == len(dataloader) - 1:
                metrics = {
                    "train/epoch": epoch + batch * dataloader.batch_size / len(dataloader.dataset),
                    "train/loss": loss.item(),
                    "train/accuracy": accuracy_score(labels.cpu().numpy(), preds),
                }
                for logger in loggers:
                    logger(metrics)
        evaluate(model, epoch + 1, val_data, criterion, loggers, config)


def evaluate(
    model: torch.nn.Module,
    epoch: int,
    data: torch.utils.data.Dataset,
    criterion: Optional[Callable[..., torch.Tensor]],
    loggers: Iterable[Logger],
    config: argparse.Namespace,
):
    """Evaluate a backbone model which produces posteriors based on the noisy data."""
    dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_preds = None
    all_labels = None
    for feats, labels in dataloader:
        # Move data to GPU
        feats = feats.to(config.device)
        clean_activations, noisy_activations = model(feats)
        loss = criterion(torch.log(noisy_activations), labels)
        preds = torch.argmax(noisy_activations, dim=-1)

        if all_preds is None:
            all_preds = preds.cpu().numpy()
            all_labels = labels.cpu().numpy()
        else:
            all_preds = np.concatenate((all_preds, preds.cpu().numpy()))
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

    acc = accuracy_score(all_labels, all_preds)

    class_names = None
    if config.dataset == Dataset.MNIST_FASHION_05 or config.dataset == Dataset.MNIST_FASHION_06:
        class_names = ["0 (T-shirt)", "1 (Trouser)", "2 (Dress)"]
    elif config.dataset == Dataset.CIFAR:
        class_names = ["0 (Plane)", "1 (Car)", "2 (Cat)"]

    for logger in loggers:
        metrics = {"train/epoch": float(epoch), "val/loss": loss.item(), "val/accuracy": acc}
        if isinstance(logger, WandbLogger) and config.wandb:
            metrics["val/confusion"] = wandb.plot.confusion_matrix(
                all_preds, all_labels, class_names
            )
        logger(metrics)


def test(
    model: torch.nn.Module,
    data: torch.utils.data.Dataset,
    loggers: Iterable[Logger],
    config: argparse.Namespace,
):
    """Evaluate a backbone model which produces posteriors based on the noisy data."""
    dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_preds = None
    all_labels = None
    for feats, labels in dataloader:
        # Move data to GPU
        feats = feats.to(config.device)
        clean_activations, noisy_activations = model(feats)
        preds = torch.argmax(clean_activations, dim=-1)

        if all_preds is None:
            all_preds = preds.cpu().numpy()
            all_labels = labels.cpu().numpy()
        else:
            all_preds = np.concatenate((all_preds, preds.cpu().numpy()))
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

    acc = accuracy_score(all_labels, all_preds)

    class_names = None
    if config.dataset == Dataset.MNIST_FASHION_05 or config.dataset == Dataset.MNIST_FASHION_06:
        class_names = ["0 (T-shirt)", "1 (Trouser)", "2 (Dress)"]
    elif config.dataset == Dataset.CIFAR:
        class_names = ["0 (Plane)", "1 (Car)", "2 (Cat)"]

    for logger in loggers:
        metrics = {"test/accuracy": acc}
        if isinstance(logger, WandbLogger) and config.wandb:
            metrics["test/confusion"] = wandb.plot.confusion_matrix(
                all_preds, all_labels, class_names
            )
        logger(metrics)


def pretrain_backbone(
    backbone: torch.nn.Module,
    data: torch.utils.data.Dataset,
    optimiser: torch.optim.Optimizer,
    criterion: Optional[Callable[..., torch.Tensor]],
    loggers: Iterable[Logger],
    config: argparse.Namespace,
):
    """Pretrain a backbone model."""
    dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    class_count = len(set(dataloader.dataset.tensors[1].tolist()))
    backbone.train()
    for epoch in range(config.backbone_pretrain_epochs):
        for batch, (feats, labels) in enumerate(dataloader):
            # Move data to GPU
            feats = feats.to(config.device)
            # Convert labels to one-hots if using BCEWithLogitsLoss
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                labels = F.one_hot(labels, num_classes=class_count).type(torch.float32)
            labels = labels.to(config.device)
            optimiser.zero_grad()
            noisy_posteriors, noisy_activations = backbone(feats)
            loss = criterion(noisy_activations, labels)
            loss.backward()

            optimiser.step()

            if batch % config.log_step == config.log_step - 1 or batch == len(dataloader) - 1:
                metrics = {
                    "pretrain/epoch": epoch
                    + batch * dataloader.batch_size / len(dataloader.dataset),
                    "pretrain/loss": loss.item(),
                }
                for logger in loggers:
                    logger(metrics)


# def eval_backbone(
#     backbone: torch.nn.Module,
#     data: torch.utils.data.Dataset,
#     loggers: Iterable[Logger],
#     config: argparse.Namespace,
# ):
#     """Evaluate a backbone model which produces posteriors based on the noisy data."""
#     dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=False, num_workers=0)
#     backbone.eval()
#     all_preds = None
#     all_labels = None
#     for feats, labels in dataloader:
#         feats = feats.to(config.device)
#         labels = labels.to(config.device)
#         noisy_posteriors, _ = backbone(feats)
#         preds = torch.argmax(noisy_posteriors, dim=-1)

#         if all_preds is None:
#             all_preds = preds.cpu().numpy()
#             all_labels = labels.cpu().numpy()
#         else:
#             all_preds = np.concatenate((all_preds, preds.cpu().numpy()))
#             all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

#     acc = accuracy_score(all_labels, all_preds)

#     class_names = None
#     if config.dataset == Dataset.MNIST_FASHION_05 or config.dataset == Dataset.MNIST_FASHION_06:
#         class_names = ["T-shirt", "Trouser", "Dress"]
#     elif config.dataset == Dataset.CIFAR:
#         class_names = ["Plane", "Car", "Cat"]

#     for logger in loggers:
#         metrics = {"val/accuracy": acc}
#         if isinstance(logger, WandbLogger) and config.wandb:
#             metrics["val/confusion"] = wandb.plot.confusion_matrix(
#                 all_preds, all_labels, class_names
#             )
#         logger(metrics)


def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse a list of command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data_parser = parser.add_argument_group("data")
    data_parser.add_argument(
        "--dataset",
        type=Dataset,
        default=Dataset.CIFAR,
        choices=list(iter(Dataset)),
        metavar=str({str(dataset.value) for dataset in iter(Dataset)}),
        help="The dataset to use.",
    )
    data_parser.add_argument(
        "--subset",
        type=float,
        default=0.8,
        help="A float between 0 and 1, the amount of training data to train on.",
    )
    data_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed to use when choosing a subset of the data to train on.",
    )
    model_parser = parser.add_argument_group("model")
    model_parser.add_argument(
        "--robust_type",
        type=RobustModel,
        default=RobustModel.FORWARD,
        choices=list(iter(RobustModel)),
        metavar=str({str(b.value) for b in iter(RobustModel)}),
        help="The type of robust model (which determines how the transition matrix is used).",
    )
    model_parser.add_argument(
        "--backbone",
        type=Backbone,
        default=Backbone.MLP,
        choices=list(iter(Backbone)),
        metavar=str({str(b.value) for b in iter(Backbone)}),
        help="The backbone to use in the model.",
    )
    model_parser.add_argument(
        "--estimator",
        type=Estimator,
        default=Estimator.ANCHOR,
        choices=list(iter(Estimator)),
        metavar=str({str(e.value) for e in iter(Estimator)}),
        help="The estimator to use for class-label noise robustness.",
    )
    model_parser.add_argument(
        "--freeze_estimator",
        type=bool,
        default=True,
        help="Whether to freeze the parameters of the estimator model.",
    )
    model_parser.add_argument(
        "--anchor_outlier_threshold",
        type=float,
        default=0.95,
        help="Threshold value to use for outliers when using the anchor point estimator.",
    )
    model_parser.add_argument(
        "--epochs",
        type=int,
        default=32,
        help="The maximum number of epochs to train the model for.",
    )
    model_parser.add_argument(
        "--backbone_pretrain_epochs",
        type=int,
        default=32,
        help=(
            "Number of epochs to pretrain the backbone on the noisy data (necessary for using "
            "the anchor point based estimator). Set to 0 to skip."
        ),
    )
    model_parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="If >0, label smoothing with this value will be applied for loss calculations.",
    )
    model_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size to use when training the model.",
    )
    model_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to train the model on.",
    )
    logging_parser = parser.add_argument_group("logging")
    logging_parser.add_argument(
        "--id",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="A unique name to identify the run.",
    )
    logging_parser.add_argument(
        "--log_step",
        type=int,
        default=32,
        help="The number of batches between logging metrics.",
    )
    logging_parser.add_argument(
        "--results_dir",
        type=str,
        default=str((Path(__file__).resolve().parent.parent / "results")),
        help="The directory to save results to.",
    )
    logging_parser.add_argument(
        "--wandb", action="store_true", help="Sync results to wandb if specified."
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    # Parse arguments
    config = parse_args(sys.argv[1:])
    # Assert that only no_transition model can have estimator as None
    if config.estimator == Estimator.NONE and config.robust_type != RobustModel.NO_TRANS:
        raise ValueError(
            (
                f"Model of type '{config.robust_type.value}' requires an estimator which "
                "is not 'none'."
            )
        )
    # Create folders for results if they do not exist
    if not Path(config.results_dir).exists():
        Path(config.results_dir).mkdir()
    # Init wandb with an additional `group` field, which is identical for runs
    # that have the same config ignoring `config.seed` and other irrelevant fields.
    # This allows us to report mean and variance of runs with different training data
    # across separate processes.
    config_dict = copy(config.__dict__)
    del config_dict["id"]
    del config_dict["seed"]
    del config_dict["results_dir"]
    del config_dict["log_step"]
    del config_dict["device"]
    # CAVEAT: The following assumes the config is not nested.
    group = hashlib.sha256(
        bytes(json.dumps(config_dict, sort_keys=True, default=str), encoding="UTF-8")
    ).hexdigest()

    if config.wandb:
        # Set up wandb
        wandb.init(project="class-conditional-label-noise", dir=config.results_dir, group=group)
        new_results_dir = Path(config.results_dir) / "local" / config.id
        new_results_dir.mkdir(parents=True)
        config.results_dir = str(new_results_dir)
        config.id = wandb.run.id
        # Serialise and deserialise config to convert enums to strings before sending to wandb
        wandb_config = json.dumps(config.__dict__, sort_keys=True, default=lambda x: x.value)
        wandb_config = json.loads(wandb_config)
        del wandb_config["id"]
        wandb.config.update(wandb_config)
    else:
        new_results_dir = Path(config.results_dir) / "local" / config.id
        new_results_dir.mkdir(parents=True)
        config.results_dir = str(new_results_dir)
        config_dict = copy(config.__dict__)
        config_dict["group"] = group
        with open(new_results_dir / "config.json", "w") as config_file:
            json.dump(config_dict, config_file, default=lambda x: x.value, indent=2)
    # Run the model
    main(config)
