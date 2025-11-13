"""
train.py – Train a single self-supervised run variation (BYOL / TW-BYOL, etc.)
The script is launched ONLY by src/main.py. It therefore assumes that all CLI
arguments originate from main.py and are validated there.
"""
import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import preprocess as pp
from src import model as models

################################################################################
# ------------------------------   helpers   ----------------------------------#
################################################################################

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


################################################################################
# ------------------------------   training   ---------------------------------#
################################################################################

def byol_step(batch: Dict[str, torch.Tensor], learner, optimizer, scaler, config):
    """One optimisation step for BYOL/TW-BYOL.

    Args
    ----
    batch : Dict – must have keys 'view1', 'view2', 'frame_dist' (frame_dist optional)
    learner : models.BYOL – model wrapper that returns p_online & z_target
    optimizer : torch Optimizer
    scaler : GradScaler or None
    config : dict – algorithm section of YAML
    """
    view1 = batch["view1"].to(get_device(), non_blocking=True)
    view2 = batch["view2"].to(get_device(), non_blocking=True)
    frame_dist = batch.get("frame_dist")  # may be None for ordinary BYOL
    if frame_dist is not None:
        frame_dist = frame_dist.to(get_device(), non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=config.get("mixed_precision", True)):
        p_online, z_target = learner(view1, view2)
        if config["type"].lower() == "tw-byol":
            tau = config["params"].get("tau", 30.0)
            loss = models.time_weighted_byol_loss(
                p_online, z_target, frame_dist=frame_dist, tau=tau
            )
        else:  # ordinary BYOL
            loss = models.byol_loss(p_online, z_target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    learner.update_target_network()
    return loss.item()


################################################################################
# ------------------------------   main   -------------------------------------#
################################################################################

def run_training(cfg: Dict, results_dir: Path):
    description = cfg.get("description", "No description provided.")
    run_id = cfg["run_id"]
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ------------------------------------------------------------------ paths
    run_dir = results_dir / run_id
    images_dir = run_dir / "images"
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- device
    device = get_device()

    # --------------------------------------------------------- dataset / dataloader
    dataset_cfg = cfg["dataset"]
    train_ds = pp.get_dataset(dataset_cfg, split="train")
    val_ds = pp.get_dataset(dataset_cfg, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=dataset_cfg.get("num_workers", 8),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"].get("val_batch_size", cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 8),
        pin_memory=True,
    )

    # ------------------------------------------------------------- model / opt
    model_cfg = cfg["model"]
    algorithm_cfg = cfg["algorithm"]

    online_backbone, projector, predictor = models.build_backbone_and_heads(model_cfg)
    learner = models.BYOL(
        backbone=online_backbone,
        projector=projector,
        predictor=predictor,
        moving_average_decay=algorithm_cfg.get("ema_decay", 0.996),
    ).to(device)

    optimizer = optim.Adam(
        learner.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=algorithm_cfg.get("mixed_precision", True))

    # ------------------------------------------------------------- training loop
    epochs = cfg["training"]["epochs"]
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "time_sec": [],
    }

    best_val_loss = float("inf")
    start_time_total = time.time()
    for epoch in range(1, epochs + 1):
        learner.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}")
        for batch in pbar:
            loss_val = byol_step(batch, learner, optimizer, scaler, algorithm_cfg)
            train_losses.append(loss_val)
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        # ---------------- validation (BYOL self-supervised loss on val set)
        learner.eval()
        with torch.no_grad():
            val_losses = []
            for batch in val_loader:
                view1 = batch["view1"].to(device, non_blocking=True)
                view2 = batch["view2"].to(device, non_blocking=True)
                frame_dist = batch.get("frame_dist")
                if frame_dist is not None:
                    frame_dist = frame_dist.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=algorithm_cfg.get("mixed_precision", True)):
                    p_online, z_target = learner(view1, view2)
                    if algorithm_cfg["type"].lower() == "tw-byol":
                        tau = algorithm_cfg["params"].get("tau", 30.0)
                        val_loss_val = models.time_weighted_byol_loss(
                            p_online, z_target, frame_dist=frame_dist, tau=tau
                        ).item()
                    else:
                        val_loss_val = models.byol_loss(p_online, z_target).item()
                val_losses.append(val_loss_val)

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        epoch_time = time.time() - start_time_total

        history["epoch"].append(epoch)
        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)
        history["time_sec"].append(epoch_time)

        # Save best model checkpoint
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            ckpt_path = run_dir / "best_model.pt"
            torch.save({"epoch": epoch, "state_dict": learner.state_dict()}, ckpt_path)

        # Epoch-level JSON logging (append-safe)
        with open(run_dir / "epoch_metrics.jsonl", "a", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_loss": mean_val_loss,
                "time_sec": epoch_time,
            }) + "\n")

    total_time = time.time() - start_time_total

    # --------------------------------------------------------- save figures
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")

    # Training & validation loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_loss"], label="Train")
    plt.plot(history["epoch"], history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss – {run_id}")
    # Annotate final value
    plt.annotate(f"{history['val_loss'][-1]:.4f}",
                 xy=(history["epoch"][-1], history["val_loss"][-1]),
                 xytext=(5, -10), textcoords='offset points')
    plt.legend()
    plt.tight_layout()
    fig_name = f"training_loss_{run_id}.pdf"
    plt.savefig(images_dir / fig_name, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------- final results
    results = {
        "run_id": run_id,
        "description": description,
        "algorithm": algorithm_cfg["type"],
        "dataset": dataset_cfg["name"],
        "model": model_cfg["type"],
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "final_val_loss": history["val_loss"][-1],
        "total_time_sec": total_time,
        "figure_files": [fig_name],
    }

    with open(run_dir / "results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # ----------------------------------------------------- stdout requirements
    print("\n===== Experiment Description =====")
    print(description)
    print("===== Numerical Results (JSON) =====")
    print(json.dumps(results))


################################################################################
# ------------------------------   CLI   --------------------------------------#
################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="Train one experiment variation.")
    parser.add_argument("--run-config", type=str, required=True,
                        help="Path to JSON or YAML file with a SINGLE run configuration.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory where outputs will be written.")
    return parser.parse_args()


def load_run_config(path: str) -> Dict:
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        import yaml
        with open(path, "r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp)
    else:
        with open(path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
    return cfg


if __name__ == "__main__":
    args = parse_args()
    cfg = load_run_config(args.run_config)
    run_training(cfg, Path(args.results_dir))