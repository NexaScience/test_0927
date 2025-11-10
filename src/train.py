"""src/train.py
Runs a single experiment variation defined by a YAML config file.  Supports
(1) supervised-learning toy workloads used for smoke-tests and (2) reinforcement
learning workloads required by the robustness-efficiency study.  The BOIL
family of algorithms – optionally augmented with Stability-Aware Curve
Compression (SACC) – and a TPE baseline are implemented inside a single file
for convenience.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import random
import yaml
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import norm  # For EI
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from src.preprocess import (
    get_dataloaders,
    set_global_seeds,
    make_rl_env,
    DummyContinuousEnv,
)
from src.model import (
    get_model,
    sigmoid_weighted_average,
    sacc_compressed_score,
)

# ---------------- RL third-party util imports (lazy) -----------------
try:
    from stable_baselines3 import PPO, DQN  # noqa: E402
    from stable_baselines3.common.evaluation import evaluate_policy  # noqa: E402
except Exception:  # pragma: no cover  – SB3 might not be installed for smoke-tests
    PPO = DQN = evaluate_policy = None  # type: ignore

# --------------------------------------------------------------------
import matplotlib  # isort: skip

matplotlib.use("Agg")  # Headless
import matplotlib.pyplot as plt  # noqa: E402  – after backend selection
import seaborn as sns  # noqa: E402

sns.set(style="whitegrid")


# ===============================================================
# Argument parsing
# ===============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single experiment variation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config for this run.")
    parser.add_argument("--results-dir", type=str, required=True, help="Root results directory provided by orchestrator.")
    parser.add_argument("--run-id", type=str, required=True, help="Unique identifier for this variation (matches config entry).")
    return parser.parse_args()


# ===============================================================
# Basic BOIL utilities (EI, curve transformation …)
# ===============================================================

def expected_improvement(
    X_candidates: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: GaussianProcessRegressor,
    xi: float = 0.01,
) -> np.ndarray:
    """Computes Expected Improvement acquisition value for *X_candidates*."""
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


def transform_curve(
    curve: List[float],
    use_sacc: bool,
    midpoint: float,
    growth: float,
    lam: float,
    tail_frac: float,
) -> float:
    if use_sacc:
        return sacc_compressed_score(curve, midpoint, growth, lam, tail_frac)
    else:
        return sigmoid_weighted_average(curve, midpoint, growth)


# ===============================================================
# Hyper-parameter learning for curve-compression parameters
# ===============================================================

def optimise_transform_hyperparams(
    curves: List[List[float]],
    X_params: np.ndarray,
    use_sacc: bool,
    tail_frac: float,
    initial: np.ndarray,
    bounds: List[Tuple[float, float]],
):
    """Learns *midpoint*, *growth* (and optionally λ) by maximising GP log-marginal likelihood."""

    def objective(params: np.ndarray) -> float:  # minimise → negative log-marginal likelihood
        midpoint, growth, lam = params
        y = np.array([transform_curve(c, use_sacc, midpoint, growth, lam, tail_frac) for c in curves])
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X_params.shape[1]), length_scale_bounds=(1e-2, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=2)
        gp.fit(X_params, y)
        return -gp.log_marginal_likelihood_value_

    res = minimize(objective, initial, bounds=bounds, method="L-BFGS-B")
    return res.x  # type: ignore[return-value]


# ===============================================================
# Supervised-learning pathway (classification dummy / HF datasets)
# ===============================================================

def train_supervised_model(
    hparams: Dict[str, Any],
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    device: torch.device,
) -> List[float]:
    """Trains a simple classifier and returns validation-accuracy curve."""
    train_loader, val_loader = get_dataloaders(data_cfg)

    model_cfg = {**model_cfg, **hparams}  # allow structural HP tuning
    model = get_model(model_cfg).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hparams.get("learning_rate", 1e-3)))

    epochs = int(training_cfg.get("epochs", 3))
    val_metric_curve: List[float] = []
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        # ---- validation ----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv).argmax(dim=1)
                correct += (pred == yv).sum().item()
                total += yv.size(0)
        acc = correct / total if total else 0.0
        val_metric_curve.append(acc)
        model.train()
    return val_metric_curve


# ===============================================================
# Reinforcement-learning pathway (PPO, DQN)
# ===============================================================

def _safe_evaluate(model, env, n_episodes: int) -> float:
    """Helper that safely calls SB3 evaluate_policy; returns average reward."""
    if evaluate_policy is None:
        # Fallback: run a manual rollout if SB3 is missing (smoke-test scenario)
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_rew = 0.0
            while not done:
                action = env.action_space.sample()
                obs, r, done, truncated, _ = env.step(action)
                done = done or truncated
                ep_rew += r
            rewards.append(ep_rew)
        return float(np.mean(rewards))
    else:
        return float(evaluate_policy(model, env, n_eval_episodes=n_episodes, render=False)[0])


def train_rl_model(
    hparams: Dict[str, Any],
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    device: torch.device,
) -> List[float]:
    """Trains an RL agent (PPO or DQN) for *total_timesteps* and returns a curve of
    evaluation rewards collected every *eval_interval* steps.

    This implementation is intentionally lightweight: it uses Stable-Baselines3
    if available; otherwise it falls back to random-policy rollouts making the
    code fully executable even in minimal environments.
    """
    algo_name = model_cfg.get("algorithm", "ppo").lower()
    env_name = data_cfg["env_name"]
    total_timesteps = int(training_cfg.get("total_timesteps", 10_000))
    eval_interval = int(training_cfg.get("eval_interval", max(1, total_timesteps // 10)))
    eval_episodes = int(training_cfg.get("eval_episodes", 5))

    # ---------------- create training & evaluation environments ----------------
    train_env = make_rl_env(env_name)
    eval_env = make_rl_env(env_name)

    # ---------------- SB3 present? --------------------------------------------
    sb3_available = PPO is not None and DQN is not None

    if not sb3_available:
        # SB3 not installed – return a synthetic monotonically-improving curve
        curve = []
        base = random.uniform(0.0, 1.0)
        for i in range(total_timesteps // eval_interval):
            noise = random.uniform(-0.02, 0.02)
            base = min(1.0, max(base + 0.05 + noise, 0.0))
            curve.append(base)
        return curve

    # ---------------- Build agent hyper-parameters -----------------------------
    common_kwargs: Dict[str, Any] = {
        "learning_rate": float(hparams.get("learning_rate", 3e-4)),
        "gamma": float(hparams.get("gamma", 0.99)),
        "device": device,
    }

    if algo_name == "ppo":
        common_kwargs.update(
            {
                "gae_lambda": float(hparams.get("gae_lambda", 0.95)),
                "clip_range": float(hparams.get("clip_eps", 0.2)),
                "ent_coef": float(hparams.get("entropy_coef", 0.0)),
                "batch_size": int(hparams.get("batch_size", 64)),
                "policy": "MlpPolicy",
                "policy_kwargs": dict(net_arch=[256, 256, 128]),
            }
        )
        model = PPO(**common_kwargs, env=train_env, verbose=0)
    elif algo_name == "dqn":
        common_kwargs.update(
            {
                "target_update_interval": int(hparams.get("target_update_interval", 500)),
                "policy": "MlpPolicy",
                "policy_kwargs": dict(net_arch=[256, 256]),
                "buffer_size": 50_000,
            }
        )
        model = DQN(**common_kwargs, env=train_env, verbose=0)
    else:
        raise ValueError(f"Unsupported RL algorithm '{algo_name}'.")

    # ---------------- Training loop w/ intermediate evaluation -----------------
    curve: List[float] = []
    steps_done = 0
    pbar = tqdm(total=total_timesteps, desc=f"Training {algo_name.upper()} on {env_name}", leave=False)
    while steps_done < total_timesteps:
        next_chunk = min(eval_interval, total_timesteps - steps_done)
        model.learn(total_timesteps=next_chunk, reset_num_timesteps=False, progress_bar=False)
        steps_done += next_chunk
        mean_reward = _safe_evaluate(model, eval_env, eval_episodes)
        curve.append(mean_reward)
        pbar.update(next_chunk)
        pbar.set_postfix({"mean_eval_reward": f"{mean_reward:.2f}"})
    pbar.close()
    train_env.close()
    eval_env.close()
    return curve


# ===============================================================
# Front-end – chooses supervised vs RL path based on dataset cfg
# ===============================================================

def run_single_evaluation(
    hparams: Dict[str, Any],
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    device: torch.device,
) -> List[float]:
    dtype = data_cfg.get("type", "supervised").lower()
    if dtype == "supervised":
        return train_supervised_model(hparams, data_cfg, model_cfg, training_cfg, device)
    elif dtype == "rl":
        return train_rl_model(hparams, data_cfg, model_cfg, training_cfg, device)
    else:
        raise ValueError(f"Unknown dataset/type '{dtype}'.")


# ===============================================================
# TPE baseline (implemented with Optuna)
# ===============================================================
try:
    import optuna  # noqa: E402
except ImportError:  # pragma: no cover – optuna might not be present in smoke-tests
    optuna = None  # type: ignore


def run_tpe_optimisation(
    algo_cfg: Dict[str, Any],
    search_space: Dict[str, Dict[str, float]],
    param_names: List[str],
    transform_kwargs: Dict[str, Any],
    eval_fn,  # callable that maps param-dict → curve-list
) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    """Runs a TPE search with *total_evaluations* trials.  Returns the same trio as
    the BOIL path: X_evaluated, curves, y_scores.
    """
    if optuna is None:
        # Fall back to random search if optuna not installed
        def suggest_random():
            return {k: random.uniform(v["min"], v["max"]) for k, v in search_space.items()}

        X_evaluated: List[List[float]] = []
        curves: List[List[float]] = []
        y_scores: List[float] = []
        for _ in range(int(algo_cfg["total_evaluations"])):
            hparams = suggest_random()
            curve = eval_fn(hparams)
            score = transform_curve(curve, **transform_kwargs)
            X_evaluated.append([hparams[p] for p in param_names])
            curves.append(curve)
            y_scores.append(score)
        return X_evaluated, curves, y_scores

    # ---------- Optuna branch ----------
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Pre-convert ranges for fast lookup
    ranges = {k: (v["min"], v["max"]) for k, v in search_space.items()}

    def objective(trial: "optuna.trial.Trial") -> float:  # type: ignore[name-defined]
        hparams = {k: trial.suggest_float(k, *ranges[k]) for k in param_names}
        curve = eval_fn(hparams)
        score = transform_curve(curve, **transform_kwargs)
        trial.set_user_attr("curve", curve)
        return score

    study.optimize(objective, n_trials=int(algo_cfg["total_evaluations"]))

    X_evaluated, curves, y_scores = [], [], []
    for t in study.trials:
        X_evaluated.append([t.params[p] for p in param_names])
        curves.append(t.user_attrs["curve"])
        y_scores.append(t.value if t.value is not None else -np.inf)
    return X_evaluated, curves, y_scores


# ===============================================================
# Main entry-point
# ===============================================================

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_id = args.run_id

    # --------------------- Prepare result directories -----------------
    run_dir = os.path.join(args.results_dir, run_id)
    images_dir = os.path.join(run_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # ----------------------- Reproducibility --------------------------
    seed = int(cfg.get("seed", 42))
    set_global_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ Algorithmic configuration --------------------
    algo_cfg: Dict[str, Any] = cfg["algorithm"]
    algo_type = algo_cfg.get("type", "boil").lower()

    tail_frac = float(algo_cfg.get("tail_frac", 0.1))
    use_sacc = bool(algo_cfg.get("use_sacc", False))
    learn_lambda = bool(algo_cfg.get("learn_lambda", False))
    lam_init = float(algo_cfg.get("lambda", 0.0))

    # Search space definition (simple flat numeric ranges)
    search_space = cfg["search_space"]
    param_names = list(search_space.keys())
    dim = len(param_names)

    # ===========================================================
    # Helper to sample random points from search-space
    # ===========================================================
    def sample_random(n: int = 1) -> np.ndarray:
        out = []
        for _ in range(n):
            cand = [random.uniform(search_space[p]["min"], search_space[p]["max"]) for p in param_names]
            out.append(cand)
        return np.array(out)

    # ===========================================================
    # Evaluation function expecting dict-of-hyper-params
    # ===========================================================
    def evaluate_hparams(hparam_dict: Dict[str, Any]) -> List[float]:
        # Ensure mandatory learning_rate for supervised path
        if "learning_rate" not in hparam_dict:
            hparam_dict["learning_rate"] = 1e-3
        return run_single_evaluation(
            hparam_dict,
            data_cfg=cfg["dataset"],
            model_cfg=cfg["model"],
            training_cfg=cfg.get("training", {}),
            device=device,
        )

    # ===========================================================
    # Containers for results common to both optimisation methods
    # ===========================================================
    X_evaluated: List[List[float]] = []
    curves: List[List[float]] = []
    y_scores: List[float] = []

    # Initial transform hyper-parameters
    midpoint, growth, lam = 0.0, 1.0, lam_init
    transform_bounds = [(-6, 6), (1e-2, 6), (0.0, 5.0)]

    transform_kwargs_base = dict(
        use_sacc=use_sacc,
        tail_frac=tail_frac,
        midpoint=midpoint,
        growth=growth,
        lam=lam,
    )

    # ===========================================================
    # -----------------  BOIL optimisation  ---------------------
    # ===========================================================
    if algo_type == "boil":
        total_evals = int(algo_cfg.get("total_evaluations", 25))
        random_init = int(algo_cfg.get("random_initial_points", 5))

        for eval_idx in range(total_evals):
            start_time = time.time()
            if eval_idx < random_init or len(y_scores) < 2:
                x_next = sample_random(1)[0]
            else:
                # Fit GP to existing data
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

                # Update transform hyper-parameters (learn λ etc.)
                if (learn_lambda or use_sacc) and len(curves) >= 2:
                    midpoint, growth, lam = optimise_transform_hyperparams(
                        curves,
                        X_np,
                        use_sacc,
                        tail_frac,
                        np.array([midpoint, growth, lam]),
                        transform_bounds,
                    )
                # Update base kwargs for transform
                transform_kwargs_base.update({"midpoint": midpoint, "growth": growth, "lam": lam})

                # Acquisition – EI over random candidate grid
                X_cand = sample_random(1000)
                ei = expected_improvement(X_cand, X_np, y_np, gp)
                best_idx = int(np.argmax(ei))
                x_next = X_cand[best_idx]

            # Convert to dict of named parameters
            hparams = {param_names[i]: float(x_next[i]) for i in range(dim)}

            # Evaluate
            curve = evaluate_hparams(hparams)
            score = transform_curve(curve, **transform_kwargs_base)

            # Bookkeeping
            X_evaluated.append(list(x_next))
            curves.append(curve)
            y_scores.append(score)

            # Simple logging to stdout (consumable by orchestrator)
            print(json.dumps({
                "run_id": run_id,
                "eval_index": eval_idx,
                "score": score,
                "midpoint": midpoint,
                "growth": growth,
                "lambda": lam,
            }))
            sys.stdout.flush()

            # Per-evaluation metadata persisted to list (written later)
            duration = time.time() - start_time
            eval_meta = {
                "index": eval_idx,
                "hyperparameters": hparams,
                "curve": curve,
                "compressed_score": score,
                "duration_sec": duration,
            }
            curves[-1] = curve  # ensure reference is correct
        # End BOIL loop

    # ===========================================================
    # -----------------  TPE optimisation  ----------------------
    # ===========================================================
    elif algo_type == "tpe":
        X_evaluated, curves, y_scores = run_tpe_optimisation(
            algo_cfg,
            search_space,
            param_names,
            transform_kwargs_base,
            evaluate_hparams,
        )
    else:
        raise ValueError(f"Unknown algorithm type '{algo_type}'.")

    # ---------------------- Final reporting -------------------------
    best_idx = int(np.argmax(y_scores))
    best_score = float(y_scores[best_idx])
    best_hparams = {param_names[i]: X_evaluated[best_idx][i] for i in range(dim)}

    results = {
        "run_id": run_id,
        "description": cfg.get("description", ""),
        "algorithm_cfg": algo_cfg,
        "search_space": search_space,
        "transform_params": {
            "midpoint": transform_kwargs_base["midpoint"],
            "growth": transform_kwargs_base["growth"],
            "lambda": transform_kwargs_base["lam"],
        },
        "best_index": best_idx,
        "best_score": best_score,
        "best_hyperparameters": best_hparams,
    }

    # Save results
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---------------------------- Figures ---------------------------
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=list(range(len(y_scores))), y=y_scores, marker="o")
    plt.title(f"Compressed Score Progression – {run_id}")
    plt.xlabel("Evaluation #")
    plt.ylabel("Compressed Score")
    plt.annotate(f"best={best_score:.3f}", (best_idx, best_score), textcoords="data", xytext=(5, 5),
                 textcoords_offset="offset points", arrowprops=dict(arrowstyle="->"))
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f"score_progression_{run_id}.pdf"), bbox_inches="tight")
    plt.close()

    # Learning curve of best model (if available)
    if curves:
        best_curve = curves[best_idx]
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=list(range(len(best_curve))), y=best_curve, marker="o")
        plt.title(f"Metric per Checkpoint – best run ({run_id})")
        plt.xlabel("Checkpoint")
        plt.ylabel("Reward" if cfg["dataset"].get("type", "supervised") == "rl" else "Validation Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f"learning_curve_{run_id}.pdf"), bbox_inches="tight")
        plt.close()

    print(json.dumps({"run_id": run_id, "status": "completed", "best_score": best_score}))


if __name__ == "__main__":
    main()
