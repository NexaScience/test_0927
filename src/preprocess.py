"""src/preprocess.py
Common preprocessing utilities and dataset registry. All dataset‐specific logic
is isolated behind a registry so that new datasets can be plugged‐in by
registering a new class.
"""
from __future__ import annotations

import random
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# HuggingFace datasets library
try:
    from datasets import load_dataset, Dataset as HFDataset
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The 'datasets' package is required for the experiment datasets.\n"
        "Install with: pip install datasets"
    ) from e

################################################################################
# Dataset registry                                                               
################################################################################

_DATASET_REGISTRY = {}


def register_dataset(name: str):
    """Decorator to register a dataset class."""

    def _inner(cls):
        _DATASET_REGISTRY[name] = cls
        return cls

    return _inner

################################################################################
# Transform helpers                                                              
################################################################################

def _build_transforms(image_size: int):
    # All diffusion models expect inputs in the range [-1, 1]. The following
    # pipeline converts PIL → tensor in [0,1] → rescale/normalise to [-1,1].
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # -> [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1,1]
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
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.randn(3, self.image_size, self.image_size)
        prompt = "a synthetic prompt"
        return img, prompt

################################################################################
# Generic HuggingFace image/text dataset                                        
################################################################################

@register_dataset("hf_text_image")
class HFTextImageDataset(Dataset):
    """Generic dataset wrapper for any HF dataset with an image & caption column.

    Params expected in `params` dict:
      hf_name            – str  – HuggingFace dataset identifier
      image_column       – str  – column containing the image (default: "image")
      caption_column     – str  – column containing caption/text (default: "text")
      resolution         – int  – resize/crop size fed into the model
      val_ratio          – float – if dataset has no dedicated validation split, take
                                   this fraction from the *end* of the dataset as val
      max_samples        – int  – optional cap on the number of samples loaded
    """

    def __init__(self, split: str, **params: Any) -> None:  # type: ignore[override]
        super().__init__()
        hf_name: str = params["hf_name"]
        img_col: str = params.get("image_column", "image")
        cap_col: str = params.get("caption_column", "text")
        resolution: int = int(params.get("resolution", 512))
        val_ratio: float = float(params.get("val_ratio", 0.05))
        max_samples: int | None = params.get("max_samples")

        # ------------------------------------------------------------------
        # 1. Load dataset split                                             
        # ------------------------------------------------------------------
        available_splits = load_dataset(hf_name, split=None).keys()  # type: ignore[arg-type]
        if split in available_splits:
            ds: HFDataset = load_dataset(hf_name, split=split)  # type: ignore[arg-type]
        else:
            # Fallback: load full training set and manually create val slice
            full_ds: HFDataset = load_dataset(hf_name, split="train")  # type: ignore[arg-type]
            n_total = len(full_ds)
            n_val = int(val_ratio * n_total)
            if split == "train":
                indices = list(range(0, n_total - n_val))
            elif split in {"val", "validation"}:
                indices = list(range(n_total - n_val, n_total))
            else:
                raise ValueError(f"Unknown split '{split}' for dataset {hf_name}")
            ds = full_ds.select(indices)

        # Optionally cap dataset size (useful for quick tests)
        if max_samples is not None and len(ds) > max_samples:
            indices = random.sample(range(len(ds)), max_samples)
            ds = ds.select(indices)

        self.data = ds
        self.image_column = img_col
        self.caption_column = cap_col
        self.transform = _build_transforms(resolution)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx):  # type: ignore[override]
        example = self.data[idx]

        img = example[self.image_column]
        if not hasattr(img, "size"):
            # In some datasets the image is stored as bytes – rely on datasets to decode
            img = example[self.image_column].convert("RGB")
        img = self.transform(img)

        caption = example.get(self.caption_column, "")
        # Many datasets store labels as ints – convert to a descriptive prompt
        if not isinstance(caption, str):
            caption = str(caption)
        return img, caption

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
    params = dataset_cfg["params"].copy()
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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
