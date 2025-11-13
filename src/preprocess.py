"""
preprocess.py – Common dataset loading & preprocessing utilities.
All dataset-specific logic is FORBIDDEN in this foundation layer and therefore
placed behind explicit placeholders.
"""
from typing import Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# =============================================================================
# Placeholders that WILL be replaced in later stages
# =============================================================================
class DatasetPlaceholder(Dataset):
    """PLACEHOLDER: Replace with actual dataset implementation.

    The dataset must return a dict with keys:
        - 'view1': Tensor
        - 'view2': Tensor
        - 'frame_dist': Tensor or int (optional, required for TW-BYOL)
    """

    def __init__(self, root: Path, split: str, transform=None, **kwargs):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.data = []  # PLACEHOLDER: populate with actual data indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # PLACEHOLDER: implement real loading & augmentation
        dummy = torch.randn(3, 224, 224)
        if self.transform:
            dummy = self.transform(dummy)
        sample = {
            "view1": dummy,
            "view2": dummy.clone(),
            "frame_dist": torch.tensor(0),
        }
        return sample


# =============================================================================
# Public API
# =============================================================================

def get_transforms(train: bool = True, cfg: Dict = None):
    cfg = cfg or {}
    if train:
        # basic augmentation pipeline (can be overridden)
        return T.Compose([
            T.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ])


def get_dataset(cfg: Dict, split: str):
    """Factory that returns correct dataset instance.

    cfg: The 'dataset' section of the run configuration.
    split: 'train', 'val', or 'test'
    """
    name = cfg["name"].lower()
    root = Path(cfg.get("root", "DATASET_ROOT_PLACEHOLDER"))  # PLACEHOLDER path
    params = cfg.get("params", {})
    transform = get_transforms(train=(split == "train"), cfg=params.get("transforms"))

    if name == "dataset_placeholder":
        return DatasetPlaceholder(root, split, transform=transform, **params)
    else:
        raise ValueError(
            f"Dataset '{name}' not recognised. "
            "# PLACEHOLDER: register dataset in preprocess.get_dataset()."
        )