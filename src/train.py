"""src/train.py
Runs a single experiment variation defined by a YAML config file.  Implements
BOIL with optional SACC curve–compression, optional alternative compressors and
supports Gym-based RL tasks (DQN on CartPole/LunarLander/Acrobot) as well as
dummy supervised datasets used during smoke-tests.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import random
import yaml
import math
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import norm  # For EI
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import gym

from src.preprocess import get_dataloaders, set_global_seeds
from src.model import (
    get_model,
    sigmoid_weighted_average,
    sacc_compressed_score,
    last_k_average,
    run_dqn,
)
import matplotlib
matplotlib.use("Agg")  # Headless
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single experiment variation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config for this run.")
    parser.add_argument("--results-dir", type=str, required=True, help="Root results directory provided by orchestrator.")
    parser.add_argument("--run-id", type=str, required=True, help="Unique identifier for this variation (matches config entry).")
    return parser.parse_args()


def expected_improvement(
    X_candidates: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: GaussianProcessRegressor,
    xi: float = 0.01,
):
    """Computes EI for a set of candidate hyper-parameters."""
    mu, sigma = model.predict(X_candidates, return_std=True)
    mu = mu.ravel()
    sigma = sigma.ravel()
    y_best = y_train.max()
    with np.errstate(divide="warn"):
        imp = mu - y_best - xi
        Z = np.zeros_like(mu)
        mask = sigma > 0
        Z[mask] = imp[mask] / sigma[mask]
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei


# ---------------------------------------------------------------------------
# Curve compressors
# ---------------------------------------------------------------------------

def transform_curve(
    curve: List[float],
    compressor: str,
    midpoint: float,
    growth: float,
    lam: float,
    tail_frac: float,
):
    """Converts a learning-curve into a scalar according to *compressor*."""
    if compressor == "sigmoid":
        return sigmoid_weighted_average(curve, midpoint, growth)
    elif compressor == "sigmoid+sacc":
        return sacc_compressed_score(curve, midpoint, growth, lam, tail_frac)
    elif compressor == "last10-average":
        return last_k_average(curve, tail_frac)
    else:
        raise ValueError(f"Unknown compressor '{compressor}'")


# ---------------------------------------------------------------------------
# Optimise midpoint/growth/(lambda) for BOIL compressors
# ---------------------------------------------------------------------------

def optimise_transform_hyperparams(
    curves: List[List[float]],
    X_params: np.ndarray,
    compressor: str,
    tail_frac: float,
    initial: np.ndarray,
    bounds: List[tuple],
):
    """Learns midpoint, growth (and λ if SACC) by maximising GP log-marginal likelihood."""

    def objective(params):
        midpoint, growth, lam = params
        y = np.array([
            transform_curve(c, compressor, midpoint, growth, lam, tail_frac) for c in curves
        ])
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X_params.shape[1]), length_scale_bounds=(1e-2, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=2)
        gp.fit(X_params, y)
        lml = gp.log_marginal_likelihood_value_
        return -lml  # minimise negative log-likelihood

    res = minimize(objective, initial, bounds=bounds, method="L-BFGS-B")
    return res.x  # best parameters


# ---------------------------------------------------------------------------
# Training helper – classification vs RL
# ---------------------------------------------------------------------------

def train_single_model(
    hparams: Dict[str, Any],
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    device: torch.device,
):
    """Trains a model (supervised or RL) and returns the validation / episode curve."""

    dataset_name = data_cfg.get("name", "dummy")

    # ------------------------------------------------------------------
    # Gym RL pathway (DQN)
    # ------------------------------------------------------------------
    if dataset_name == "gym":
        env_name = data_cfg["env_name"]
        train_episodes = int(data_cfg.get("train_episodes", 500))
        seed = int(data_cfg.get("seed", 42))
        curve = run_dqn(
            env_name=env_name,
            hparams=hparams,
            train_episodes=train_episodes,
            seed=seed,
            device=device,
        )
        return curve

    # ------------------------------------------------------------------
    # Supervised dummy pathway (used only for smoke tests)
    # ------------------------------------------------------------------
    train_loader, val_loader = get_dataloaders(data_cfg)

    model_cfg_full = model_cfg.copy()
    model_cfg_full.update(hparams)  # allow structural HPs (e.g., hidden size) to be tuned
    model = get_model(model_cfg_full).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.get("learning_rate", 1e-3))

    epochs = training_cfg.get("epochs", 3)
    val_metric_curve = []
    model.train()
    for epoch in range(epochs):
        for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        # ---- validation ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for (xv, yv) in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv).argmax(dim=1)
                correct += (pred == yv).sum().item()
                total += yv.size(0)
        acc = correct / total if total else 0.0
        val_metric_curve.append(acc)
        model.train()
    return val_metric_curve


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load YAML for this run
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_id = args.run_id

    # ------------------------------------------------------------------
    # Prepare result directories
    run_dir = os.path.join(args.results_dir, run_id)
    images_dir = os.path.join(run_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Reproducibility setup
    seed = cfg.get("seed", 42)
    set_global_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Describe experiment
    description = cfg.get("description", "No description provided.")
    print("===== Experiment Description =====")
    print(description)
    print("==================================")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Algorithmic configuration
    algo_cfg = cfg["algorithm"]
    total_evals = int(algo_cfg.get("total_evaluations", 25))
    random_init = int(algo_cfg.get("random_initial_points", 5))
    tail_frac = float(algo_cfg.get("tail_frac", 0.1))
    compressor = algo_cfg.get("compression", "sigmoid")  # sigmoid | sigmoid+sacc | last10-average
    learn_lambda = bool(algo_cfg.get("learn_lambda", False))

    # Define search-space (flat numeric ranges)
    search_space = cfg["search_space"]
    param_names = list(search_space.keys())
    dim = len(param_names)

    def sample_random(n: int = 1):
        out = []
        for _ in range(n):
            cand = []
            for p in param_names:
                lo, hi = search_space[p]["min"], search_space[p]["max"]
                cand.append(random.uniform(lo, hi))
            out.append(cand)
        return np.array(out)

    # Containers
    X_evaluated: List[List[float]] = []
    curves: List[List[float]] = []
    y_scores: List[float] = []
    all_evals: List[Dict[str, Any]] = []

    # Initial transform hyper-parameters (midpoint, growth, λ)
    midpoint, growth, lam = 0.0, 1.0, float(algo_cfg.get("lambda", 0.0))
    transform_bounds = [(-6, 6), (1e-2, 6), (0.0, 5.0)]

    success_threshold = algo_cfg.get("success_threshold", None)
    time_to_threshold = None

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------
    for eval_idx in range(total_evals):
        start_time = time.time()

        # --------------------------------------------------------------
        # Choose the next hyper-parameter candidate
        # --------------------------------------------------------------
        if eval_idx < random_init or len(y_scores) < 2:
            x_next = sample_random(1)[0]
        else:
            # Fit GP surrogate on existing data
            X_np = np.array(X_evaluated)
            y_np = np.array(y_scores)
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-3, 1e3))
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
            )
            gp.fit(X_np, y_np)

            # Optionally re-optimise curve-compression hyper-parameters (only for sigmoid(+sacc))
            if compressor in {"sigmoid", "sigmoid+sacc"} and learn_lambda and len(curves) >= 2:
                midpoint, growth, lam = optimise_transform_hyperparams(
                    curves,
                    X_np,
                    compressor,
                    tail_frac,
                    np.array([midpoint, growth, lam]),
                    transform_bounds,
                )

            # Acquisition – Expected Improvement over random Sobol-like sample
            X_cand = sample_random(1000)
            ei = expected_improvement(X_cand, X_np, y_np, gp)
            best_idx = int(np.argmax(ei))
            x_next = X_cand[best_idx]

        # Map candidate vector → dict with correct names
        hparams = {param_names[i]: float(x_next[i]) for i in range(dim)}
        if "learning_rate" not in hparams:
            hparams["learning_rate"] = 1e-3  # default for dummy tasks

        # --------------------------------------------------------------
        # Run training / evaluation
        # --------------------------------------------------------------
        curve = train_single_model(
            hparams=hparams,
            data_cfg=cfg["dataset"],
            model_cfg=cfg["model"],
            training_cfg=cfg.get("training", {}),
            device=device,
        )

        score = transform_curve(curve, compressor, midpoint, growth, lam, tail_frac)

        # --------------------------------------------------------------
        # Book-keeping
        # --------------------------------------------------------------
        X_evaluated.append(list(x_next))
        curves.append(curve)
        y_scores.append(score)

        if success_threshold is not None and score >= success_threshold and time_to_threshold is None:
            time_to_threshold = eval_idx + 1

        all_evals.append(
            {
                "index": eval_idx,
                "hyperparameters": hparams,
                "curve": curve,
                "compressed_score": score,
                "duration_sec": time.time() - start_time,
            }
        )

        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "eval_index": eval_idx,
                    "score": score,
                    "midpoint": midpoint,
                    "growth": growth,
                    "lambda": lam,
                }
            )
        )
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Final reporting & artefacts
    # ------------------------------------------------------------------
    best_idx = int(np.argmax(y_scores))
    best_score = float(y_scores[best_idx])
    best_hparams = all_evals[best_idx]["hyperparameters"]

    results = {
        "run_id": run_id,
        "description": description,
        "algorithm_cfg": algo_cfg,
        "search_space": search_space,
        "transform_params": {
            "midpoint": midpoint,
            "growth": growth,
            "lambda": lam,
        },
        "evaluations": all_evals,
        "best_index": best_idx,
        "best_score": best_score,
        "best_hyperparameters": best_hparams,
        "time_to_threshold": time_to_threshold,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # -------------------- Visualisations ---------------------------
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=list(range(len(y_scores))), y=y_scores, marker="o")
    plt.title(f"Compressed Score Progression – {run_id}")
    plt.xlabel("Evaluation #")
    plt.ylabel("Compressed Score")
    plt.annotate(
        f"best={best_score:.3f}",
        (best_idx, best_score),
        textcoords="data",
        xytext=(5, 5),
        textcoords_offset="offset points",
        arrowprops=dict(arrowstyle="->"),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f"score_progression_{run_id}.pdf"), bbox_inches="tight")
    plt.close()

    best_curve = curves[best_idx]
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=list(range(len(best_curve))), y=best_curve, marker="o")
    plt.title(f"Best-run Learning Curve – {run_id}")
    plt.xlabel("Episode" if cfg["dataset"]["name"] == "gym" else "Epoch")
    plt.ylabel("Reward" if cfg["dataset"]["name"] == "gym" else "Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f"learning_curve_{run_id}.pdf"), bbox_inches="tight")
    plt.close()

    print(json.dumps({"run_id": run_id, "status": "completed", "best_score": best_score}))


if __name__ == "__main__":
    main()
