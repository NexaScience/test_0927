"""src/preprocess.py
Common utilities for data loading / preprocessing (dummy supervision path) and
seed control.  Gym-based RL environments are handled directly in train.py.
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
# Dummy supervised dataset used only for smoke-tests
# ================================================================

def _build_dummy_dataset(n_samples: int = 512, input_dim: int = 10, n_classes: int = 2):
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(x, y)


def get_dataloaders(dataset_cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Returns train & validation loaders for dummy supervised tasks."""
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

    raise NotImplementedError(
        "Only 'dummy' dataset is supported via this path. RL tasks are handled directly in train.py."
    )
