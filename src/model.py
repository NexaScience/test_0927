"""src/model.py
Model definitions, RL agent (DQN) and curve-compression utilities.
"""
from __future__ import annotations
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

# ================================================================
# Supervised baseline network (dummy)
# ================================================================
class BaseClassifier(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ================================================================
# DQN components
# ================================================================
class QNetwork(nn.Module):
    """Simple 2-layer MLP Q-function."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, action_dim)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.pos = 0

    def add(self, s, a, r, s2, d):
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, s2, d))
        else:
            self.buffer[self.pos] = (s, a, r, s2, d)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int = 64):
        idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        s, a, r, s2, d = zip(*(self.buffer[i] for i in idxs))
        return (
            np.stack(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.stack(s2),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 50000,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.q = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.action_dim = action_dim

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q(state_t)
        return int(torch.argmax(q_vals, dim=1).item())

    def update(self):
        if len(self.replay) < self.batch_size:
            return 0.0
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.q(s_t).gather(1, a_t)
        with torch.no_grad():
            q_next = self.target_q(s2_t).max(dim=1, keepdim=True)[0]
            target = r_t + self.gamma * q_next * (1.0 - d_t)
        loss = F.mse_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def soft_update(self, tau: float = 1.0):
        """Hard update when tau == 1.0"""
        for tgt, src in zip(self.target_q.parameters(), self.q.parameters()):
            tgt.data.copy_(src.data if tau == 1.0 else tau * src.data + (1 - tau) * tgt.data)


# ================================================================
# Run DQN for a given environment / hyper-parameter config
# ================================================================

def run_dqn(
    env_name: str,
    hparams: Dict[str, float],
    train_episodes: int = 500,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> List[float]:
    """Trains a DQN agent and returns per-episode reward curve."""

    import warnings

    # Gym API compatibility wrappers --------------------------------------------------
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    env = gym.make(env_name)
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    lr = float(hparams["learning_rate"])
    target_interval = int(hparams["target_update"])
    eps_final = float(hparams["epsilon_final"])
    eps_initial = 1.0
    eps_decay = 0.995  # multiplicative decay per episode until eps_final

    agent = DQNAgent(state_dim, action_dim, lr=lr, device=device)

    episode_rewards: List[float] = []
    global_step = 0

    for ep in range(train_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        epsilon = max(eps_final, eps_initial * (eps_decay ** ep))
        while not done:
            action = agent.act(state, epsilon)
            try:
                next_state, reward, done, _ = env.step(action)  # old Gym
            except ValueError:
                next_state, reward, terminated, truncated, _ = env.step(action)  # Gymnasium
                done = terminated or truncated
            agent.replay.add(state, action, reward, next_state, done)
            loss = agent.update()

            state = next_state
            total_reward += reward
            global_step += 1
            if global_step % target_interval == 0:
                agent.soft_update(tau=1.0)  # hard update
        episode_rewards.append(total_reward)
    env.close()
    return episode_rewards


# ================================================================
# Curve compression utilities
# ================================================================

def sigmoid_weighted_average(curve: List[float], midpoint: float = 0.0, growth: float = 1.0) -> float:
    n = len(curve)
    x_scaled = np.linspace(-6, 6, n)
    weights = 1.0 / (1.0 + np.exp(-growth * (x_scaled - midpoint)))
    weights = weights / weights.sum()
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


def last_k_average(curve: List[float], tail_frac: float = 0.1) -> float:
    k = max(1, int(len(curve) * tail_frac))
    return float(np.mean(curve[-k:]))


# ================================================================
# Model factory (currently only dummy classifier used via get_model)
# ================================================================

def get_model(model_cfg: Dict[str, Any]) -> nn.Module:
    name = model_cfg.get("name", "dummy_classifier")
    if name == "dummy_classifier":
        return BaseClassifier(
            input_dim=int(model_cfg.get("input_dim", 10)),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_classes=int(model_cfg.get("num_classes", 2)),
        )

    raise NotImplementedError(f"Model '{name}' is not supported in this code-base.")
