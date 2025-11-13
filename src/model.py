"""
model.py – Model architectures & BYOL utilities.
Contains:
1. Backbone builders for 2D (image) and lightweight 3D (video) variants.
2. BYOL wrapper with target network EMA.
3. Loss functions including time-weighted (exponential or binary) variants.
"""
from typing import Tuple
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
from torchvision.models import video as tvm_video

################################################################################
# -------------------------  Utility / helper classes  ------------------------#
################################################################################

class ResNetVideoWrapper(nn.Module):
    """Wraps a 2D ResNet and applies it frame-wise, then averages over time."""

    def __init__(self, resnet_2d: nn.Module):
        super().__init__()
        self.backbone = resnet_2d  # without fc layer (identity)

    def forward(self, x: torch.Tensor):  # x: (B,C,T,H,W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T,C,H,W)
        feats = self.backbone(x)  # (B*T, F)
        feats = feats.view(B, T, -1).mean(dim=1)  # temporal average -> (B,F)
        return feats

################################################################################
# --------------------------   projection heads   -----------------------------#
################################################################################

class MLPHead(nn.Module):
    """2-layer MLP projection/prediction head for BYOL."""

    def __init__(self, in_dim: int, hidden_dim: int = 4096, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

################################################################################
# ------------------------------  backbones  ----------------------------------#
################################################################################

def _remove_fc(module: nn.Module) -> Tuple[nn.Module, int]:
    """Sets .fc to Identity and returns the backbone + feature dim."""
    feat_dim = module.fc.in_features
    module.fc = nn.Identity()
    return module, feat_dim


def build_backbone_and_heads(model_cfg: dict):
    """Returns (backbone, projector, predictor) for BYOL."""
    model_type = model_cfg["type"].lower()

    # -------------------- 2D ResNets wrapped to video --------------------
    if model_type in {"resnet18_video", "resnet18"}:
        backbone_2d, feat_dim = _remove_fc(tv_models.resnet18(weights=None))
        backbone = ResNetVideoWrapper(backbone_2d)
    elif model_type in {"resnet50_video", "resnet50"}:
        backbone_2d, feat_dim = _remove_fc(tv_models.resnet50(weights=None))
        backbone = ResNetVideoWrapper(backbone_2d)

    # --------------------- lightweight 3D CNNs ---------------------------
    elif model_type == "r3d_18":
        r3d = tvm_video.r3d_18(weights=None)
        feat_dim = r3d.fc.in_features
        r3d.fc = nn.Identity()
        backbone = r3d  # already processes (B,C,T,H,W)

    elif model_type == "mc3_18":
        mc3 = tvm_video.mc3_18(weights=None)
        feat_dim = mc3.fc.in_features
        mc3.fc = nn.Identity()
        backbone = mc3

    else:
        raise ValueError(f"Unknown model type '{model_type}' in config.")

    # -------------------------- heads ------------------------------
    proj_hidden = model_cfg.get("proj_hidden_dim", 4096)
    proj_out = model_cfg.get("proj_output_dim", 256)
    predictor_hidden = model_cfg.get("predictor_hidden_dim", 4096)

    projector = MLPHead(feat_dim, proj_hidden, proj_out)
    predictor = MLPHead(proj_out, predictor_hidden, proj_out)
    return backbone, projector, predictor

################################################################################
# -------------------------------  BYOL  --------------------------------------#
################################################################################

class BYOL(nn.Module):
    """Minimal BYOL implementation supporting TW-BYOL loss computation."""

    def __init__(self, backbone: nn.Module, projector: nn.Module, predictor: nn.Module, moving_average_decay: float = 0.996):
        super().__init__()
        self.online_backbone = backbone
        self.online_projector = projector
        self.predictor = predictor

        self.target_backbone = copy.deepcopy(backbone)
        self.target_projector = copy.deepcopy(projector)
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.moving_average_decay = moving_average_decay

    @torch.no_grad()
    def update_target_network(self):
        self._update_moving_average(self.online_backbone, self.target_backbone)
        self._update_moving_average(self.online_projector, self.target_projector)

    @torch.no_grad()
    def _update_moving_average(self, online: nn.Module, target: nn.Module):
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.mul_(self.moving_average_decay).add_(p_o.data, alpha=1.0 - self.moving_average_decay)

    # -------------------------------------------------------------
    def forward(self, view1, view2):
        p_online = self.predictor(self.online_projector(self.online_backbone(view1)))
        with torch.no_grad():
            z_target = self.target_projector(self.target_backbone(view2)).detach()
        return p_online, z_target

################################################################################
# ----------------------------  loss functions  -------------------------------#
################################################################################

def byol_loss(p_online: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(p_online, z_target)


def time_weighted_byol_loss(
    p_online: torch.Tensor,
    z_target: torch.Tensor,
    frame_dist: torch.Tensor,
    tau: float = 30.0,
    mode: str = "exponential",
) -> torch.Tensor:
    """Time-weighted BYOL loss.

    mode='exponential'  : weight = exp(-Δt / τ)
    mode='binary'       : weight = 1 if Δt <= τ else 0
    """
    if frame_dist is None:
        raise RuntimeError("frame_dist tensor is required for TW-BYOL.")

    if mode == "exponential":
        weight = torch.exp(-frame_dist.float() / tau).to(p_online.device)
    elif mode == "binary":
        weight = (frame_dist.float() <= tau).float().to(p_online.device)
    else:
        raise ValueError(f"Unknown weight_mode '{mode}'.")

    per_sample = (p_online - z_target).pow(2).sum(dim=1)
    return (weight * per_sample).mean()