"""src/preprocess.py
Common data loading / preprocessing utilities with dataset placeholders.
"""
from __future__ import annotations
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, Any

# ================================================================
# Seed control – deterministic behaviour across experiments
# ================================================================

def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================
# Dataset loading – with placeholders for future replacements
# ================================================================

def _build_dummy_dataset(n_samples: int = 512, input_dim: int = 10, n_classes: int = 2):
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(x, y)


def get_dataloaders(dataset_cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Returns train & validation dataloaders.

    PLACEHOLDER: Will be replaced with task-specific dataset logic in later steps.
    """
    name = dataset_cfg.get("name", "dummy")
    batch_size = int(dataset_cfg.get("batch_size", 32))

    if name == "dummy":
        ds = _build_dummy_dataset()
        train_size = int(0.8 * len(ds))
        val_size = len(ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        # PLACEHOLDER: Will be replaced with specific dataset loading logic
        raise NotImplementedError(f"Dataset '{name}' not yet implemented in common foundation.")
