# src/preprocess.py
"""Common preprocessing pipeline now fully implemented for real datasets.
Supports:
  • dummy            – small FakeData for CI / smoke tests.
  • cifar10          – torchvision CIFAR-10 download.
  • cifar10_hf       – HuggingFace version (uoft-cs/cifar10).
  • imagenet64       – HuggingFace subset (huggan/imagenet-64-32k).
  • lsun_bedroom     – HuggingFace LSUN-bedroom subset (huggan/lsun_bedroom).

All images are converted to tensors in the range [-1,1].  Additional datasets
can be added by extending `get_dataset` following the same pattern.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Optional: Hugging Face datasets (installed via pyproject)
try:
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover
    load_dataset = None  # Will raise later if user requests HF dataset

from PIL import Image

# ------------------------------------------------------------------------- #
# Transform helpers                                                         #
# ------------------------------------------------------------------------- #

def get_transforms(config: dict):
    """Creates torchvision transforms that output tensors in [-1, 1]."""
    tfms: List = []
    resize = config.get("data", {}).get("resize")
    if resize is not None:
        tfms.append(transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR))
    tfms.extend([
        transforms.ToTensor(),  # → [0,1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # → [-1,1]
    ])
    return transforms.Compose(tfms)


# ------------------------------------------------------------------------- #
# HuggingFace → PyTorch bridge                                              #
# ------------------------------------------------------------------------- #

class HFDatasetTorch(torch.utils.data.Dataset):
    """Lightweight wrapper around a HuggingFace dataset that yields (tensor, 0)."""

    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        img = example["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img_t = self.transform(img)
        return img_t, 0  # dummy label so that downstream uses batch[0]


# ------------------------------------------------------------------------- #
# Dataset factory                                                           #
# ------------------------------------------------------------------------- #

def _require_hf(pkg_name: str):  # pragma: no cover
    if load_dataset is None:
        raise ImportError(
            f"datasets library not installed – needed for dataset '{pkg_name}'. "
            "Please install with `pip install datasets` or add to dependencies."
        )


def get_dataset(name: str, train: bool, config: dict):
    """Returns a torch.utils.data.Dataset instance."""

    name = name.lower()
    split = "train" if train else "test"

    # ------------------------------------------------------------------ #
    # 1. Dummy (for CI)
    # ------------------------------------------------------------------ #
    if name == "dummy":
        image_size = config.get("data", {}).get("image_size", (3, 32, 32))
        return datasets.FakeData(
            size=config.get("data", {}).get("num_samples", 256),
            image_size=image_size,
            num_classes=10,
            transform=get_transforms(config),
        )

    # ------------------------------------------------------------------ #
    # 2. CIFAR-10 (torchvision)
    # ------------------------------------------------------------------ #
    if name == "cifar10":
        root = Path(config.get("data", {}).get("root", "./data"))
        return datasets.CIFAR10(root=root, train=train, transform=get_transforms(config), download=True)

    # ------------------------------------------------------------------ #
    # 3. CIFAR-10 (HuggingFace – uoft-cs/cifar10)
    # ------------------------------------------------------------------ #
    if name == "cifar10_hf":
        _require_hf(name)
        hf_ds = load_dataset("uoft-cs/cifar10", split=split)
        return HFDatasetTorch(hf_ds, get_transforms(config))

    # ------------------------------------------------------------------ #
    # 4. ImageNet-64 subset (huggan/imagenet-64-32k)
    # ------------------------------------------------------------------ #
    if name == "imagenet64":
        _require_hf(name)
        hf_ds = load_dataset("huggan/imagenet-64-32k", split="train") if train else load_dataset(
            "huggan/imagenet-64-32k", split="validation"
        )
        # Optional subsampling for quick iterations
        subset_size = config.get("data", {}).get("subset_size")
        if subset_size is not None and subset_size < len(hf_ds):
            hf_ds = hf_ds.shuffle(seed=42).select(range(subset_size))
        return HFDatasetTorch(hf_ds, get_transforms(config))

    # ------------------------------------------------------------------ #
    # 5. LSUN-bedroom (huggan/lsun_bedroom)
    # ------------------------------------------------------------------ #
    if name == "lsun_bedroom":
        _require_hf(name)
        hf_ds = load_dataset("huggan/lsun_bedroom", split="train")
        if not train:
            # Use last 10k images as a pseudo-test set
            hf_ds = hf_ds.select(range(-10000, 0))
        return HFDatasetTorch(hf_ds, get_transforms(config))

    # ------------------------------------------------------------------ #
    # Unknown dataset
    # ------------------------------------------------------------------ #
    raise NotImplementedError(f"Dataset '{name}' is not implemented.")


# ------------------------------------------------------------------------- #
# DataLoader helper                                                         #
# ------------------------------------------------------------------------- #

def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader | None]:
    batch_size = config.get("training", {}).get("batch_size", 16)
    num_workers = config.get("data", {}).get("num_workers", max(1, os.cpu_count() // 2))

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
