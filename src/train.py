# src/train.py

"""
Runs a single experiment variation.
This file should be executed ONLY by src.main.  It performs the complete
training loop, optional validation, sampling/FID evaluation and finally saves
all metrics + figures in a structured directory so that src.evaluate can later
aggregate across runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports (relative to repo root)
from . import preprocess as preprocess
from . import model as model_lib

# ----------------------------- Utility helpers ----------------------------- #

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ----------------------------- Main training ------------------------------- #

def train(config: Dict, results_dir: Path, run_id: str) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------- #
    # 1.  Data                                                               #
    # --------------------------------------------------------------------- #
    train_loader, val_loader = preprocess.get_dataloaders(config)

    # --------------------------------------------------------------------- #
    # 2.  Model + diffusion utilities                                        #
    # --------------------------------------------------------------------- #
    model = model_lib.get_model(config)
    model.to(device)

    # Optimiser & schedulers
    optim_cfg = config.get("optimizer", {})
    lr = optim_cfg.get("lr", 1e-4)
    betas = optim_cfg.get("betas", (0.9, 0.999))
    weight_decay = optim_cfg.get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    epochs = config.get("training", {}).get("epochs", 1)
    grad_clip = config.get("training", {}).get("grad_clip_norm", 1.0)

    # --------------------------------------------------------------------- #
    # 3.  Training loop                                                      #
    # --------------------------------------------------------------------- #
    history: Dict[str, List] = {"train_loss": [], "val_loss": []}
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=config.get("training", {}).get("amp", True))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"[Run {run_id}] Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            imgs = batch[0].to(device)  # torchvision FakeData returns tuple(img, target)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config.get("training", {}).get("amp", True)):
                loss_dict = model.training_step(imgs)
                loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            # Gradient clipping
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(1, num_batches)
        history["train_loss"].append(avg_train_loss)

        # --------------------- optional validation ---------------------- #
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch[0].to(device)
                    loss_dict = model.training_step(imgs)
                    val_running_loss += loss_dict["loss"].item()
                    val_batches += 1
            avg_val_loss = val_running_loss / max(1, val_batches)
        else:
            avg_val_loss = None
        history["val_loss"].append(avg_val_loss)

        # ---------------- progress logging ----------------------------- #
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                }
            )
        )

    training_time = time.time() - start_time

    # --------------------------------------------------------------------- #
    # 4.  Evaluation (FID)                                                   #
    # --------------------------------------------------------------------- #
    metrics: Dict[str, float] = {}
    if config.get("evaluation", {}).get("compute_fid", False):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except ImportError:
            raise ImportError(
                "torchmetrics not installed. Please add 'torchmetrics' to your dependencies."
            )

        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        model.eval()

        # Accumulate real images (limited to avoid OOM during smoke tests)
        max_real_batches = config.get("evaluation", {}).get("fid_num_batches", 1)
        real_batches = 0
        for batch in train_loader:
            imgs_real = batch[0].to(device)
            fid_metric.update(imgs_real, real=True)
            real_batches += 1
            if real_batches >= max_real_batches:
                break

        # Generate synthetic images (simple ancestral sampling)
        num_gen = imgs_real.shape[0] * max_real_batches
        model_samples = model.generate(num_gen, device=device)
        fid_metric.update(model_samples, real=False)
        fid_score = fid_metric.compute().item()
        metrics["fid"] = fid_score

    # --------------------------------------------------------------------- #
    # 5.  Persist metrics & figures                                          #
    # --------------------------------------------------------------------- #
    # Save metrics
    metrics["final_train_loss"] = history["train_loss"][-1]
    if avg_val_loss is not None:
        metrics["final_val_loss"] = avg_val_loss
    metrics["training_time_sec"] = training_time

    results = {
        "run_id": run_id,
        "config": config,
        "history": history,
        "metrics": metrics,
    }

    save_json(results, results_dir / run_id / "results.json")

    # Figures directory
    img_dir = results_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training loss curve
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    xs = list(range(1, epochs + 1))
    plt.plot(xs, history["train_loss"], label="train_loss")
    if any(v is not None for v in history["val_loss"]):
        plt.plot(xs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss – {run_id}")
    # Annotate final value
    plt.annotate(
        f"{history['train_loss'][-1]:.4f}",
        xy=(xs[-1], history["train_loss"][-1]),
        xytext=(xs[-1], history["train_loss"][-1] * 1.05),
    )
    plt.legend()
    plt.tight_layout()
    out_path = img_dir / f"training_loss_{run_id}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------ #
    # 6.  Print final JSON to STDOUT (required by structured logging)    #
    # ------------------------------------------------------------------ #
    print(json.dumps({"run_id": run_id, "status": "completed", "metrics": metrics}))

    return results


# ----------------------------- CLI wrapper -------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single experiment variation.")
    p.add_argument("--config", type=str, required=True, help="Path to config JSON file specific to this run.")
    p.add_argument("--results-dir", type=str, required=True, help="Root directory where outputs will be stored.")
    p.add_argument("--run-id", type=str, required=True, help="Unique identifier for this run variation.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config (written by main orchestrator)
    with open(args.config, "r") as f:
        config = json.load(f)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.get("seed", 42))

    train(config, results_dir, args.run_id)


if __name__ == "__main__":
    main()
