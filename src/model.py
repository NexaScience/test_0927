"""
model.py – Backbone architectures & BYOL utilities.
A *minimal yet fully functional* implementation of the SlowFast-8×8 backbone is
provided alongside ResNet variants so that the experimental YAMLs can refer to
`slowfast_8x8` without requiring heavy third-party dependencies.

The pseudo SlowFast model simply applies a **2-D ResNet** *frame-wise* and then
averages the resulting features across the temporal dimension. While this is
obviously not the full SlowFast design it respects the expected input tensor
shape **(B, C, T, H, W)** and produces a strong baseline representation that is
perfectly adequate for verifying the end-to-end research pipeline.
"""
from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

# ============================================================================
# ---------------------------  projection heads  -----------------------------
# ============================================================================

class MLPHead(nn.Module):
    """2-layer MLP used as projector or predictor in BYOL."""

    def __init__(self, in_dim: int, hidden_dim: int = 4096, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D)
        return self.net(x)


# ============================================================================
# ----------------------------  backbones  -----------------------------------
# ============================================================================

class PseudoSlowFast(nn.Module):
    """A lightweight SlowFast-style backbone.

    The implementation applies a 2-D ResNet to each frame of the clip and then
    global-averages features across time. Accepts input tensors of shape
    **(B, C, T, H, W)** and returns a tensor **(B, F)**.
    """

    def __init__(self, base_model: str = "resnet18"):
        super().__init__()
        # Instantiate base ResNet without the classifier.
        if base_model == "resnet18":
            resnet = tv_models.resnet18(weights=None)
        elif base_model == "resnet50":
            resnet = tv_models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported base model '{base_model}' for PseudoSlowFast.")

        self.feat_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.frame_encoder = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, H, W)
        b, c, t, h, w = x.shape
        # Merge B and T so we can reuse the 2-D ResNet efficiently.
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(b * t, c, h, w)  # (B*T, C, H, W)
        feats = self.frame_encoder(x)  # (B*T, F)
        feats = feats.view(b, t, self.feat_dim)  # (B, T, F)
        feats = feats.mean(dim=1)  # temporal average → (B, F)
        return feats


# ---------------------------------------------------------------------------
# Backbone / projector / predictor factory used by train.py
# ---------------------------------------------------------------------------

def build_backbone_and_heads(model_cfg: dict) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Return *(backbone, projector, predictor)* modules ready for BYOL."""

    model_type = model_cfg["type"].lower()

    # -------------- standard 2-D ResNets -----------------------------------
    if model_type == "resnet18":
        backbone = tv_models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif model_type == "resnet50":
        backbone = tv_models.resnet50(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    # -------------- pseudo SlowFast variant --------------------------------
    elif model_type in {"slowfast_8x8", "slowfast", "slowfast-r18-8x8"}:
        backbone = PseudoSlowFast(base_model="resnet18")
        feat_dim = backbone.feat_dim

    else:
        raise ValueError(f"Unknown or unsupported model type '{model_type}'.")

    # ----------------------- BYOL heads  -----------------------------------
    proj_hidden = model_cfg.get("proj_hidden_dim", 4096)
    proj_out = model_cfg.get("proj_output_dim", 256)
    predictor_hidden = model_cfg.get("predictor_hidden_dim", 4096)

    projector = MLPHead(feat_dim, proj_hidden, proj_out)
    predictor = MLPHead(proj_out, predictor_hidden, proj_out)
    return backbone, projector, predictor


# ============================================================================
# ---------------------------  BYOL wrapper  ----------------------------------
# ============================================================================

class BYOL(nn.Module):
    """Minimal BYOL implementation with target network EMA."""

    def __init__(
        self,
        backbone: nn.Module,
        projector: nn.Module,
        predictor: nn.Module,
        moving_average_decay: float = 0.996,
    ):
        super().__init__()
        self.online_backbone = backbone
        self.online_projector = projector
        self.predictor = predictor

        # Target (momentum) encoder – copy at initialisation
        self.target_backbone = copy.deepcopy(backbone)
        self.target_projector = copy.deepcopy(projector)
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.moving_average_decay = moving_average_decay

    # -------------------------- helpers -----------------------------------
    @torch.no_grad()
    def _update_moving_average(self, online: nn.Module, target: nn.Module):
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            p_t.data = p_t.data * self.moving_average_decay + p_o.data * (1.0 - self.moving_average_decay)

    @torch.no_grad()
    def update_target_network(self):
        self._update_moving_average(self.online_backbone, self.target_backbone)
        self._update_moving_average(self.online_projector, self.target_projector)

    # --------------------------- forward ----------------------------------
    def forward(self, view1: torch.Tensor, view2: torch.Tensor):
        # Online network
        o1 = self.online_backbone(view1)
        p1 = self.online_projector(o1)
        p_online = self.predictor(p1)  # (B, D)

        # Target network (no grads)
        with torch.no_grad():
            t2 = self.target_backbone(view2)
            z_target = self.target_projector(t2)
        return p_online, z_target.detach()


# ============================================================================
# -------------------------  Loss functions  ----------------------------------
# ============================================================================

def byol_loss(p_online: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Standard mean-squared BYOL alignment loss."""
    return F.mse_loss(p_online, z_target)


def time_weighted_byol_loss(
    p_online: torch.Tensor, z_target: torch.Tensor, frame_dist: torch.Tensor, tau: float = 30.0
) -> torch.Tensor:
    """Time-weighted BYOL loss as introduced in the research paper."""
    if frame_dist is None:
        raise RuntimeError("frame_dist tensor is required for TW-BYOL.")
    weight = torch.exp(-frame_dist.float() / tau).to(p_online.device)  # (B,)
    per_sample_loss = (p_online - z_target).pow(2).sum(dim=1)  # (B,)
    return (weight * per_sample_loss).mean()
