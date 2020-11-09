"""Utilities for loading and interacting with the datasets."""
import os.path

import numpy as np
import torch
from config import Dataset


def load_data(root: str, dataset: Dataset, subset: float, seed: int):
    """Load a dataset into memory."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if dataset == Dataset.MNIST_FASHION_05:
        path = os.path.join(root, "FashionMNIST0.5.npz")
    elif dataset == Dataset.MNIST_FASHION_06:
        path = os.path.join(root, "FashionMNIST0.6.npz")
    elif dataset == Dataset.CIFAR:
        path = os.path.join(root, "CIFAR.npz")
    else:
        raise NotImplementedError()

    data = np.load(path)

    # To get val data, we randomly sample 80% of the n examples to train a model
    # and use the remaining 20% to validate the model.
    train_val_feats = data["Xtr"]  # Train features. Size (n, *image_size)
    train_val_labels = data["Str"]  # Noisy labels. Size (n,) each label in {0, 1, 2}

    test_feats = data["Xts"]  # Test features. Size (m, *image_size)
    test_labels = data["Yts"]  # Clean labels. Size (m,) each label in {0, 1, 2}

    # Randomly split train and val data
    indices = np.random.permutation(train_val_labels.shape[0])
    take = int(subset * train_val_labels.shape[0])
    print(f"seed: {seed}")
    print(
        f"shuffle: [{', '.join([str(i) for i in indices[:min(10, take)]])}, ...]",
    )
    shuffled_feats = train_val_feats[indices]
    shuffled_labels = train_val_labels[indices]

    train_feats = shuffled_feats[:take]
    train_labels = shuffled_labels[:take]
    val_feats = shuffled_feats[take:]
    val_labels = shuffled_labels[take:]

    return (train_feats, train_labels), (val_feats, val_labels), (test_feats, test_labels)
