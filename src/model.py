# src/model.py
"""Model architecture implementations (baseline + Auto-ASE variants).

Implemented architectures:
  • unet32              – CIFAR-10 (32×32)
  • unet64              – ImageNet-64 (64×64)
  • unet512_latent      – Stable-Diffusion latent UNet (64×64 latent, 512 px images)
Each architecture is built from the same SimpleUNet template but with different
capacity.  Auto-ASE gating is available through the `lambda_gates` parameter:
    • lambda_gates == 0.0   → no gates (baseline / ASE-linear)
    • lambda_gates  > 0.0   → gates are active and regularised.
"""
from __future__ import annotations

import math
import re
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------- #
# 1.  Positional timestep embeddings                                       #
# ------------------------------------------------------------------------- #

def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Creates sinusoidal timestep embeddings (as in ADM/DDPM code)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ------------------------------------------------------------------------- #
# 2.  Auto-ASE Gated wrapper                                               #
# ------------------------------------------------------------------------- #

class GatedBlock(nn.Module):
    """Wraps an nn.Module and applies a learnable gate g_k(t).

    During training gates are soft (sigmoid).  During evaluation they are
    binarised via a straight-through estimator (STE).
    """

    def __init__(self, block: nn.Module, t_dim: int):
        super().__init__()
        self.block = block
        self.w = nn.Parameter(torch.zeros(1))  # initialise at 0 → gate≈0.5
        self.t_proj = nn.Linear(t_dim, 1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, train: bool = True):
        h_t = 1.0 - torch.sigmoid(self.t_proj(temb))  # shape (B,1)
        gate_cont = torch.sigmoid(self.w * h_t)       # (B,1)
        gate = gate_cont if train else (gate_cont > 0.5).float()  # STE
        while gate.dim() < x.dim():
            gate = gate.unsqueeze(-1)
        y = x + gate * (self.block(x, temb) - x)
        return y, gate_cont.mean()


# ------------------------------------------------------------------------- #
# 3.  Building blocks                                                      #
# ------------------------------------------------------------------------- #

class ConvBlock(nn.Module):
    """Two-conv residual block with timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(t_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = in_ch == out_ch

    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        h = self.act(self.conv1(x))
        h = h + self.emb_proj(temb)[:, :, None, None]
        h = self.act(self.conv2(h))
        if self.skip:
            h = h + x
        return h


# ------------------------------------------------------------------------- #
# 4.  Simple UNet backbone                                                 #
# ------------------------------------------------------------------------- #

class SimpleUNet(nn.Module):
    def __init__(
        self,
        img_channels: int,
        base_channels: int,
        image_size: int,
        time_dim: int = 128,
        gated: bool = False,
        lambda_gate: float = 0.05,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        noise_scale: float = 1.0,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.lambda_gate = lambda_gate
        self.gated = gated
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.noise_scale = noise_scale
        self.image_size = image_size

        # time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder
        self.down1 = self._make_block(img_channels, base_channels)
        self.pool1 = nn.AvgPool2d(2)
        self.down2 = self._make_block(base_channels, base_channels * 2)
        self.pool2 = nn.AvgPool2d(2)

        # Bottleneck
        self.bottleneck = self._make_block(base_channels * 2, base_channels * 2)

        # Decoder
        self.up1 = self._make_block(base_channels * 2 + base_channels * 2, base_channels)
        # Final conv
        self.out_conv = nn.Conv2d(base_channels + base_channels, img_channels, 1)

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _make_block(self, in_ch: int, out_ch: int):
        block = ConvBlock(in_ch, out_ch, self.time_dim)
        if self.gated:
            return GatedBlock(block, self.time_dim)
        return block

    def _apply_block(self, block, x, temb, train: bool, stats: List):
        if isinstance(block, GatedBlock):
            y, stat = block(x, temb, train=train)
            stats.append(stat)
            return y
        else:
            return block(x, temb)

    # ------------------------------------------------------------------ #
    # Forward / sampling                                                 #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, t: torch.Tensor, train: bool = True):
        temb = timestep_embedding(t, self.time_dim)
        temb = self.time_mlp(temb)
        gate_stats: List[torch.Tensor] = []

        # Encoder
        d1 = self._apply_block(self.down1, x, temb, train, gate_stats)
        p1 = self.pool1(d1)
        d2 = self._apply_block(self.down2, p1, temb, train, gate_stats)
        p2 = self.pool2(d2)

        # Bottleneck
        bn = self._apply_block(self.bottleneck, p2, temb, train, gate_stats)

        # Decoder step 1 (upsample + concat with d2)
        up = F.interpolate(bn, scale_factor=2, mode="nearest")
        up = torch.cat([up, d2], dim=1)
        up = self._apply_block(self.up1, up, temb, train, gate_stats)

        # Final upsample, concat with d1 and project to image
        up = F.interpolate(up, scale_factor=2, mode="nearest")
        up = torch.cat([up, d1], dim=1)
        out = self.out_conv(up)
        return out, gate_stats

    # ------------------------------------------------------------------ #
    # Training step                                                     #
    # ------------------------------------------------------------------ #
    def training_step(self, x0: torch.Tensor) -> dict:
        device = x0.device
        B = x0.size(0)
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Linear beta schedule (only schedule currently supported)
        betas = torch.linspace(1e-4, 0.02, self.num_timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        noise = torch.randn_like(x0) * self.noise_scale
        sqrt_ab = torch.sqrt(alpha_bars[t])[:, None, None, None]
        sqrt_one_minus_ab = torch.sqrt(1 - alpha_bars[t])[:, None, None, None]
        x_noisy = sqrt_ab * x0 + sqrt_one_minus_ab * noise

        pred_noise, gate_stats = self.forward(x_noisy, t)
        noise_loss = F.mse_loss(pred_noise, noise)
        gate_reg = (
            torch.stack(gate_stats).mean() if gate_stats else torch.tensor(0.0, device=device)
        )
        total_loss = noise_loss + self.lambda_gate * gate_reg
        return {"loss": total_loss, "noise_loss": noise_loss.detach(), "gate_loss": gate_reg.detach()}

    # ------------------------------------------------------------------ #
    # Simple ancestral sampling (for FID)                                 #
    # ------------------------------------------------------------------ #
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, 3, self.image_size, self.image_size, device=device)
            T = 100  # shorter sampling for speed during evaluation
            betas = torch.linspace(1e-4, 0.02, T, device=device)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)

            for t_idx in reversed(range(T)):
                t_batch = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
                eps_theta, _ = self.forward(x, t_batch, train=False)
                alpha_bar_t = alpha_bars[t_batch][:, None, None, None]
                beta_t = betas[t_batch][:, None, None, None]
                coef1 = 1 / torch.sqrt(alphas[t_batch][:, None, None, None])
                coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)
                x = coef1 * (x - coef2 * eps_theta)
                if t_idx > 0:
                    noise = torch.randn_like(x)
                    x += torch.sqrt(beta_t) * noise
            return torch.clamp(x, -1.0, 1.0).cpu()


# ------------------------------------------------------------------------- #
# 5.  Model factory                                                        #
# ------------------------------------------------------------------------- #

_DEF_ARCH = {
    "unet32": {"img_size": 32, "base_channels": 64},
    "unet64": {"img_size": 64, "base_channels": 128},
    "unet512_latent": {"img_size": 64, "base_channels": 256},  # latent 64×64
}


def get_model(config: dict) -> nn.Module:
    name = config.get("model").lower()
    diff_cfg = config.get("diffusion", {})

    # Identify architecture key (substring match)
    arch_key = None
    for k in _DEF_ARCH.keys():
        if name.startswith(k):
            arch_key = k
            break
    if arch_key is None:
        raise ValueError(f"Unknown/unsupported model architecture in name '{name}'.")

    gated = diff_cfg.get("lambda_gates", 0.0) > 0.0
    arch = _DEF_ARCH[arch_key]

    return SimpleUNet(
        img_channels=3,
        base_channels=arch["base_channels"],
        image_size=arch["img_size"],
        gated=gated,
        lambda_gate=diff_cfg.get("lambda_gates", 0.0),
        num_timesteps=diff_cfg.get("timesteps", 1000),
        beta_schedule=diff_cfg.get("beta_schedule", "linear"),
        noise_scale=diff_cfg.get("corrupt_sigma", 1.0),
    )
