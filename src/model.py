"""src/model.py
Network architectures and curve-compression utilities.  Reinforcement-learning
agents in *train.py* rely on Stable-Baselines3, but we include explicit network
modules here for completeness and potential future use.
"""
from __future__ import annotations
from typing import Dict, Any, List

import numpy as np
import torch
from torch import nn

# ================================================================
# Supervised-learning classifier (used in smoke tests)
# ================================================================


class BaseClassifier(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


# ================================================================
# RL-friendly MLPs (actor, critic, Q-network)
# ================================================================
class MLP(nn.Module):
    """Generic multi-layer perceptron with configurable hidden layers and output
    size.  Activations are ReLU (tanh at final layer optional).
    """

    def __init__(self, in_dim: int, layers: List[int], out_dim: int, final_tanh: bool = False):
        super().__init__()
        net: List[nn.Module] = []
        last = in_dim
        for h in layers:
            net += [nn.Linear(last, h), nn.ReLU()]
            last = h
        net.append(nn.Linear(last, out_dim))
        if final_tanh:
            net.append(nn.Tanh())
        self.seq = nn.Sequential(*net)

    def forward(self, x):  # type: ignore[override]
        return self.seq(x)


class ActorCritic(nn.Module):
    """Simple shared-backbone actor-critic network (continuous actions)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = [256, 256, 128]
        self.backbone = MLP(obs_dim, hidden, hidden[-1])
        self.policy_head = nn.Linear(hidden[-1], act_dim)
        self.value_head = nn.Linear(hidden[-1], 1)

    def forward(self, x):  # type: ignore[override]
        h = self.backbone(x)
        return self.policy_head(h), self.value_head(h)


class QNetwork(nn.Module):
    """MLP Q-network for DQN (discrete actions)."""

    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        hidden = [256, 256]
        self.net = MLP(obs_dim, hidden, num_actions)

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


# ================================================================
# Model factory
# ================================================================

def get_model(model_cfg: Dict[str, Any]) -> nn.Module:
    name = model_cfg.get("name", "dummy_classifier").lower()

    if name == "dummy_classifier":
        return BaseClassifier(
            input_dim=int(model_cfg.get("input_dim", 10)),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_classes=int(model_cfg.get("num_classes", 2)),
        )
    elif name == "actor_critic":
        return ActorCritic(
            obs_dim=int(model_cfg.get("obs_dim", 24)),
            act_dim=int(model_cfg.get("act_dim", 4)),
        )
    elif name == "q_network":
        return QNetwork(
            obs_dim=int(model_cfg.get("obs_dim", 4)),
            num_actions=int(model_cfg.get("num_actions", 2)),
        )
    else:
        raise ValueError(f"Unknown model name '{name}'.")


# ================================================================
# Curve compression utilities (BOIL + optional SACC)
# ================================================================

def sigmoid_weighted_average(curve: List[float], midpoint: float = 0.0, growth: float = 1.0) -> float:
    n = len(curve)
    x_scaled = np.linspace(-6, 6, n)
    weights = 1.0 / (1.0 + np.exp(-growth * (x_scaled - midpoint)))
    weights /= weights.sum()
    return float(np.sum(np.array(curve) * weights))


def sacc_compressed_score(
    curve: List[float],
    midpoint: float,
    growth: float,
    lam: float = 1.0,
    tail_frac: float = 0.1,
) -> float:
    base = sigmoid_weighted_average(curve, midpoint, growth)
    k = max(1, int(len(curve) * tail_frac))
    stability_penalty = np.std(curve[-k:])
    return float(base - lam * stability_penalty)
