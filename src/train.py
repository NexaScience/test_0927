"""src/train.py
Runs a single experiment variation specified by --run-id in a YAML config.
All experiment artefacts are written to <results-dir>/<run-id>/.
The script prints a single JSON line with the final metrics to stdout so that
main.py can parse it. Stdout/stderr are also captured by main.py and mirrored
into log files.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from . import preprocess as pp
from . import model as mdl

################################################################################
# Helper functions                                                               
################################################################################

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# Core training / validation utilities                                          
################################################################################

def _train_one_epoch(
    model: mdl.BaseExperimentModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float | None = None,
) -> float:
    model.train()
    running_loss: List[float] = []
    pbar = tqdm(dataloader, desc=f"Train E{epoch}", leave=False)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        batch_on_device = [x.to(device) if torch.is_tensor(x) else x for x in batch]
        out = model.training_step(batch_on_device)
        loss: torch.Tensor = out["loss"]
        loss.backward()
        if max_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        running_loss.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return float(sum(running_loss) / len(running_loss))


def _validate(
    model: mdl.BaseExperimentModel,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    model.eval()
    val_losses: List[float] = []
    pbar = tqdm(dataloader, desc=f"Val   E{epoch}", leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch_on_device = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            out = model.validation_step(batch_on_device)
            loss: torch.Tensor = out["val_loss"]
            val_losses.append(loss.item())
            pbar.set_postfix(val_loss=f"{loss.item():.4f}")
    return float(sum(val_losses) / len(val_losses))

################################################################################
# Inference‐time utilities                                                      
################################################################################

def _measure_inference_time(model: mdl.BaseExperimentModel, prompts: List[str], device: torch.device) -> float:
    """Measures the average wall‐clock inference latency per sample.
    For diffusion models this calls the generate method; for toy models this
    performs a single forward pass.
    """
    model.eval()
    with torch.no_grad():
        start = time.time()
        _ = model.generate(prompts) if hasattr(model, "generate") else model(torch.randn(1, 3, 32, 32, device=device))
        torch.cuda.synchronize() if device.type == "cuda" else None
        end = time.time()
    return float(end - start)

################################################################################
# Main entry point                                                              
################################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single experiment variation")
    parser.add_argument("--run-id", required=True, help="Name of the run variation as defined in the YAML config")
    parser.add_argument("--config-file", required=True, help="Path to YAML experiment config")
    parser.add_argument("--results-dir", required=True, help="Directory where results will be stored")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = setup_device()

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    run_cfg: Dict = None  # type: ignore
    for r in config["runs"]:
        if r["name"] == args.run_id:
            run_cfg = r
            break
    if run_cfg is None:
        raise ValueError(f"Run id {args.run_id} not found in config {args.config_file}")

    # ------------------------------------------------------------------
    # 2. Prepare IO paths
    # ------------------------------------------------------------------
    run_dir = Path(args.results_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(exist_ok=True)

    # Save run config for reproducibility
    with open(run_dir / "cfg.yaml", "w") as f:
        yaml.safe_dump(run_cfg, f)

    # ------------------------------------------------------------------
    # 3. Build dataset & dataloaders
    # ------------------------------------------------------------------
    train_loader, val_loader = pp.build_dataloaders(
        dataset_cfg=run_cfg["dataset"], batch_size=run_cfg["training"]["batch_size"], num_workers=run_cfg["training"].get("num_workers", 4)
    )

    # ------------------------------------------------------------------
    # 4. Build model & optimiser
    # ------------------------------------------------------------------
    model: mdl.BaseExperimentModel = mdl.build_model(run_cfg["model"], device=device)
    optimizer = optim.AdamW(model.parameters(), lr=run_cfg["training"]["lr"])

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    history: List[Dict] = []
    epochs: int = run_cfg["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, device, epoch, max_norm=run_cfg["training"].get("grad_clip", None)
        )
        val_loss = _validate(model, val_loader, device, epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    # ------------------------------------------------------------------
    # 6. Final evaluation metrics (FID, CLIPScore, inference‐time)
    # ------------------------------------------------------------------
    fid_score = None
    clip_score = None
    if run_cfg.get("evaluation", {}).get("compute_fid", False):
        fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(feature=2048).to(device)
        clip_metric = torchmetrics.functional.clip_score
        model.eval()
        with torch.no_grad():
            for imgs, prompts in tqdm(val_loader, desc="Computing FID/CLIP"):
                imgs = imgs.to(device)
                # Generate images using current model
                gen_imgs = model.generate(prompts) if hasattr(model, "generate") else imgs  # type: ignore
                gen_imgs = (gen_imgs.clamp(-1, 1) + 1) / 2  # to [0,1]
                fid_metric.update(gen_imgs, real=False)
                fid_metric.update((imgs.clamp(-1, 1) + 1) / 2, real=True)
                clip_score_batch = clip_metric(gen_imgs, prompts)
                if clip_score is None:
                    clip_score = clip_score_batch.mean()
                else:
                    clip_score += clip_score_batch.mean()
        fid_score = float(fid_metric.compute())
        clip_score = float(clip_score / len(val_loader)) if clip_score is not None else None

    inference_latency = _measure_inference_time(model, prompts=["a test prompt"], device=device)

    # ------------------------------------------------------------------
    # 7. Save artefacts & print JSON
    # ------------------------------------------------------------------
    torch.save(model.state_dict(), run_dir / "model.pt")

    results = {
        "run_id": args.run_id,
        "final_epoch": epochs,
        "metrics": {
            "train_loss": history[-1]["train_loss"] if history else None,
            "val_loss": history[-1]["val_loss"] if history else None,
            "fid": fid_score,
            "clip_score": clip_score,
            "inference_time": inference_latency,
        },
        "history": history,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results))  # <-- structured output expected by main.py


if __name__ == "__main__":
    main()
