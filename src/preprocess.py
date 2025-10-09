# src/preprocess.py
"""Common preprocessing pipeline with real dataset support.

This module now contains fully-fledged dataloader logic for CIFAR-10 via the
Hugging Face datasets hub (dataset id: "uoft-cs/cifar10").  A lightweight
`FakeData` fallback remains for CI / smoke tests.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List, Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets, transforms

# We lazily import HF datasets to avoid the dependency cost when running only
# smoke tests (which use torchvision FakeData).  ImportError will propagate if
# a real HF dataset is requested without the package installed.
try:
    from datasets import load_dataset
except ModuleNotFoundError:  # pragma: no cover – handled at runtime
    load_dataset = None  # type: ignore


# ------------------------------------------------------------------------- #
# Transform helpers                                                         #
# ------------------------------------------------------------------------- #

def cifar10_transforms() -> transforms.Compose:
    """Standard CIFAR-10 data augmentation + mapping to [-1, 1]."""
    tfms: List[Any] = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0,1] -> [-1,1]
    ]
    return transforms.Compose(tfms)


def dummy_transforms(image_size=(3, 32, 32)) -> transforms.Compose:
    tfms: List[Any] = [
        transforms.ToTensor(),
    ]
    return transforms.Compose(tfms)


# ------------------------------------------------------------------------- #
# HF Dataset wrappers                                                       #
# ------------------------------------------------------------------------- #

class HFImageDataset(Dataset):
    """Thin wrapper converting a Hugging Face dataset into a PyTorch dataset."""

    def __init__(self, hf_ds, tfms):
        self.ds = hf_ds
        self.tfms = tfms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        # The exact field name can vary ("img"|"image") – we try both.
        img = sample.get("img", None)
        if img is None:
            img = sample.get("image", None)
        if img is None:
            raise KeyError("Expected image field 'img' or 'image' in HF dataset but neither found.")
        if self.tfms:
            img = self.tfms(img)
        # We return a dummy label to keep the 2-tuple contract expected by the
        # training pipeline (image, target).
        return img, 0


# ------------------------------------------------------------------------- #
# Dataset factory                                                           #
# ------------------------------------------------------------------------- #

def get_dataset(name: str, train: bool, config: dict):
    """Returns a torch.utils.data.Dataset instance for the requested dataset."""

    # ------------------------------------------------------------------ #
    # Smoke-test / CI dataset                                            #
    # ------------------------------------------------------------------ #
    if name == "dummy":
        image_size = config.get("data", {}).get("image_size", (3, 32, 32))
        return tv_datasets.FakeData(
            size=config.get("data", {}).get("num_samples", 256),
            image_size=image_size,
            num_classes=10,
            transform=dummy_transforms(image_size),
        )

    # ------------------------------------------------------------------ #
    # CIFAR-10 (HuggingFace)                                             #
    # ------------------------------------------------------------------ #
    if name == "cifar10":
        if load_dataset is None:
            raise ImportError(
                "The 'datasets' package is required for CIFAR-10.  Please install it via pip install datasets"
            )
        split = "train" if train else "test"
        hf_ds = load_dataset("uoft-cs/cifar10", split=split)
        return HFImageDataset(hf_ds, cifar10_transforms())

    # ---------------------------- fallback ---------------------------- #
    raise NotImplementedError(f"Dataset '{name}' is not implemented.")


# ------------------------------------------------------------------------- #
# Dataloader helper                                                         #
# ------------------------------------------------------------------------- #

def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader | None]:
    batch_size = config.get("training", {}).get("batch_size", 16)
    num_workers = config.get("data", {}).get("num_workers", os.cpu_count() // 2)

    dataset_name = config.get("dataset")
    train_dataset = get_dataset(dataset_name, train=True, config=config)

    val_loader = None
    if config.get("training", {}).get("validation_split", 0.0) > 0.0:
        val_split = config["training"]["validation_split"]
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader