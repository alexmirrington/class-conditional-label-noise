"""Main entrypoint for model training and evaluation."""
import argparse
import hashlib
import json
import os.path
import sys
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import List

from config import Dataset, Model
from datasets import load_data
from factories import ModelFactory
from loggers import JSONLLogger, StreamLogger, WandbLogger
from termcolor import colored

import wandb


def main(config: argparse.Namespace):
    """Train or evaluate a label-noise-robust model."""
    # Load dataset
    print(colored("dataset:", attrs=["bold"]))
    print(config.dataset)
    # Determine dataset directory
    train, val, test = load_data(
        os.path.join(os.path.dirname(__file__), os.path.pardir, "data"),
        config.dataset,
        config.subset,
        config.seed,
    )
    train_feats, train_labels = train
    val_feats, val_labels = val
    test_feats, test_labels = test

    print(f"train: {train_feats.shape}")
    print(f"val: {val_feats.shape}")
    print(f"test: {test_feats.shape}")

    # Create loggers
    loggers = [StreamLogger(), JSONLLogger(Path(config.results_dir) / f"{config.id}.jsonl")]
    if config.wandb:
        loggers.append(WandbLogger())

    # Create model
    print(colored("model:", attrs=["bold"]))
    model_factory = ModelFactory()
    model = model_factory.create(config)
    print(config.model)
    print(model)

    # Train and evaluate model
    # TODO


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
        "--model",
        type=Model,
        default=Model.CNN_FORWARD,
        choices=list(iter(Model)),
        metavar=str({str(model.value) for model in iter(Model)}),
        help="The model to use.",
    )
    model_parser.add_argument(
        "--epochs",
        type=int,
        default=32,
        help="The maximum number of epochs to train the model for.",
    )
    model_parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size to use when training the model.",
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
        default=16,
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
    # CAVEAT: The following assumes the config is not nested.
    group = hashlib.sha256(
        bytes(json.dumps(config_dict, sort_keys=True, default=str), encoding="UTF-8")
    ).hexdigest()

    if config.wandb:
        # Set up wandb
        wandb.init(project="class-conditional-label-noise", dir=config.results_dir, group=group)
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
