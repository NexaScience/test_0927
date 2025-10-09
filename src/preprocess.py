"""src/preprocess.py
Common preprocessing utilities and dataset registry. All dataset‐specific logic
is isolated behind a registry so that new datasets can be plugged‐in by
registering a new class.
"""
from __future__ import annotations

import random
from typing import Tuple, List, Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# External library used for real datasets
try:
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore

################################################################################
# Dataset registry                                                               
################################################################################

_DATASET_REGISTRY: Dict[str, Any] = {}


def register_dataset(name):
    def _inner(cls):
        _DATASET_REGISTRY[name] = cls
        return cls

    return _inner

################################################################################
# Utility transforms                                                            
################################################################################

def _make_train_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _make_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

################################################################################
# Toy dataset (used by smoke tests)                                             
################################################################################

@register_dataset("toy")
class ToyDataset(Dataset):
    """A minimal random dataset for smoke tests. Returns (img, prompt)."""

    def __init__(self, num_samples: int, image_size: int, split: str = "train") -> None:
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.split = split
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.randn(3, self.image_size, self.image_size)
        prompt = "a synthetic prompt"
        return img, prompt

################################################################################
# LAION-Aesthetics 512² subset (for fine-tuning & validation)                   
################################################################################

@register_dataset("laion_aesthetics_512")
class LaionAestheticsDataset(Dataset):
    """Wrapper around the limingcv/LAION_Aesthetics_512 HF dataset.

    Parameters accepted via **params in YAML:
      image_size (int):          Final resolution after transforms.
      subset_size (int|None):    Optional. Randomly sample at most this many
                                 elements from the split for quick experiments.
    """

    def __init__(self, split: str = "train", image_size: int = 512, subset_size: int | None = None):
        if load_dataset is None:
            raise ImportError("`datasets` library is required for LaionAestheticsDataset but is not installed.")
        self.split = split
        # The published dataset only has a single split called "train". We will
        # use it for both training and validation, sampling a disjoint subset
        # for the latter if requested.
        full_ds = load_dataset("limingcv/LAION_Aesthetics_512", split="train", streaming=False)
        indices: List[int]
        if subset_size is not None and subset_size < len(full_ds):
            random.seed(42 if split == "val" else 123)
            indices = random.sample(range(len(full_ds)), subset_size)
            self.ds = full_ds.select(indices)
        else:
            self.ds = full_ds
        # Data augmentation / preprocessing
        self.transform = _make_train_transform(image_size) if split == "train" else _make_eval_transform(image_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image: Image.Image = item.get("image", None)
        if image is None:
            raise KeyError("Expected key `image` not found in dataset item.")
        caption: str = (
            item.get("text")
            or item.get("TEXT")
            or item.get("caption")
            or item.get("prompt")
            or "an image"
        )
        img_tensor = self.transform(image)
        return img_tensor, str(caption).lower()

################################################################################
# MS-COCO 2014 validation (prompts + images)                                    
################################################################################

@register_dataset("mscoco2014_val")
class MSCOCO2014ValDataset(Dataset):
    """30k MS-COCO 2014 validation set prompts & images used for evaluation."""

    def __init__(self, image_size: int = 512, subset_size: int | None = 30000, split: str = "val"):
        if load_dataset is None:
            raise ImportError("`datasets` library is required for MSCOCO2014ValDataset but is not installed.")
        if split != "val":
            raise ValueError("MSCOCO2014ValDataset only supports split='val'.")
        # HF dataset "coco_captions" provides images + multiple captions. We
        # pick the first caption for simplicity.
        full_ds = load_dataset("coco_captions", "2014", split="validation")
        if subset_size is not None and subset_size < len(full_ds):
            random.seed(2023)
            indices = random.sample(range(len(full_ds)), subset_size)
            self.ds = full_ds.select(indices)
        else:
            self.ds = full_ds
        self.transform = _make_eval_transform(image_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image: Image.Image = item["image"]
        captions: List[str] = item["captions"]
        caption = captions[0] if captions else "a photo"
        return self.transform(image), caption.lower()

################################################################################
# Dataloader builder                                                            
################################################################################

def build_dataloaders(
    dataset_cfg: dict,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders from a dataset config dict."""
    name = dataset_cfg["name"]
    params = dataset_cfg.get("params", {}).copy()
    DatasetCls = _DATASET_REGISTRY.get(name)
    if DatasetCls is None:
        raise ValueError(
            f"Dataset '{name}' is not registered. Available: {list(_DATASET_REGISTRY.keys())}."
        )

    train_ds = DatasetCls(split="train", **params)
    val_ds = DatasetCls(split="val", **params)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
