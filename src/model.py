"""src/model.py
Model registry and core algorithm implementations including the Feature
Consistency Regularisation (FCR) diffusion model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODEL_REGISTRY: Dict[str, Any] = {}

################################################################################
# Registry helpers                                                              
################################################################################

def register_model(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(model_cfg: Dict[str, Any], device: torch.device):
    name = model_cfg["name"]
    params = model_cfg.get("params", {})
    ModelCls = _MODEL_REGISTRY.get(name)
    if ModelCls is None:
        raise ValueError(
            f"Model '{name}' is not registered. Available: {list(_MODEL_REGISTRY.keys())}."
        )
    model: BaseExperimentModel = ModelCls(params=params, device=device)  # type: ignore
    return model.to(device)

################################################################################
# Base class                                                                    
################################################################################

class BaseExperimentModel(nn.Module):
    """Abstract base class that every model follows."""

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------
    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def generate(self, prompts: List[str]):
        """Optional for generative models."""
        raise NotImplementedError

################################################################################
# Toy classification model (for smoke tests)                                    
################################################################################

@register_model("toy")
class ToyModel(BaseExperimentModel):
    """Simple 3-layer MLP classifier on flattened images. Used by smoke tests."""

    def __init__(self, params: Dict[str, Any], device: torch.device):
        super().__init__()
        self.input_dim = params["input_dim"]
        hidden_dim = params.get("hidden_dim", 128)
        num_classes = params.get("num_classes", 10)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.to(device)

    def forward(self, x: torch.Tensor):
        return self.net(x.view(x.size(0), -1))

    def training_step(self, batch):
        imgs, _ = batch  # prompts ignored
        logits = self.forward(imgs)
        labels = torch.randint(0, self.num_classes, (imgs.size(0),), device=imgs.device)
        loss = self.loss_fn(logits, labels)
        return {"loss": loss}

    def validation_step(self, batch):
        imgs, _ = batch
        logits = self.forward(imgs)
        labels = torch.randint(0, self.num_classes, (imgs.size(0),), device=imgs.device)
        loss = self.loss_fn(logits, labels)
        return {"val_loss": loss}

    def generate(self, prompts):
        # Not a generative model – return random tensor for API compliance
        return torch.randn(1, 3, 32, 32, device=next(self.parameters()).device)

################################################################################
# Diffusion backbone imports                                                    
################################################################################

# Import heavy deps lazily so that smoke tests remain lightweight
try:
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError:  # pragma: no cover
    AutoencoderKL = UNet2DConditionModel = DDPMScheduler = CLIPTextModel = CLIPTokenizer = None  # type: ignore

################################################################################
# Diffusion w/ FCR model                                                        
################################################################################


def _safe_first_tensor(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


@register_model("diffusion_fcr")
class DiffusionFCRModel(BaseExperimentModel):
    """Stable-Diffusion v1.5 UNet fine-tuned with Feature Consistency Regularisation."""

    def __init__(self, params: Dict[str, Any], device: torch.device):
        super().__init__()
        if AutoencoderKL is None:
            raise ImportError("diffusers and transformers must be installed for diffusion models.")

        model_name = params.get("pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5")
        self.dtype = torch.float16 if params.get("use_fp16", True) else torch.float32
        # Core components
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=self.dtype).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=self.dtype).to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=self.dtype).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

        # FCR specific hyper-params
        self.lambda_fcr: float = float(params.get("lambda_fcr", 0.1))
        self.delta_max: int = int(params.get("delta_max", 10))
        self.default_stride: int = int(params.get("stride", 10))

        self.device = device
        # Forward hook to capture encoder features
        self._enc_feats: Optional[torch.Tensor] = None

        def _hook_fn(_, __, output):
            self._enc_feats = _safe_first_tensor(output)

        # Register hook on the last down block (deepest UNet level)
        self.unet.down_blocks[-1].register_forward_hook(_hook_fn)

    # ------------------------------------------------------------------
    # Helper methods                                                     
    # ------------------------------------------------------------------
    def _encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(imgs).latent_dist.sample()
        return latents * 0.18215  # SD scaling factor

    def _get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        return self.text_encoder(tokens.input_ids)[0]

    # ------------------------------------------------------------------
    # Training & validation                                             
    # ------------------------------------------------------------------
    def training_step(self, batch):
        imgs, prompts = batch
        imgs = imgs.to(self.device, dtype=self.dtype)
        latents = self._encode_images(imgs)
        bsz = latents.size(0)

        # Sample main timestep
        t = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        # Text conditioning
        text_emb = self._get_text_embeddings(prompts)

        # First forward pass – capture encoder feats
        self._enc_feats = None
        noise_pred = self.unet(noisy_latents, t, encoder_hidden_states=text_emb).sample
        loss_main = F.mse_loss(noise_pred, noise)
        f1 = self._enc_feats.detach() if self._enc_feats is not None else None

        # Second forward pass for FCR (only if lambda_fcr>0)
        loss_fcr = torch.tensor(0.0, device=self.device)
        if self.lambda_fcr > 0 and f1 is not None:
            delta = torch.randint(1, self.delta_max + 1, (bsz,), device=self.device)
            t2 = torch.clamp(t - delta, min=0)
            noisy_latents2 = self.scheduler.add_noise(latents, noise, t2)

            self._enc_feats = None
            _ = self.unet(noisy_latents2, t2, encoder_hidden_states=text_emb)
            f2 = self._enc_feats.detach()
            loss_fcr = (f1 - f2).pow(2).mean()

        loss = loss_main + self.lambda_fcr * loss_fcr
        return {"loss": loss, "loss_main": loss_main, "loss_fcr": loss_fcr}

    @torch.no_grad()
    def validation_step(self, batch):
        imgs, prompts = batch
        imgs = imgs.to(self.device, dtype=self.dtype)
        latents = self._encode_images(imgs)
        bsz = latents.size(0)
        t = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        noise_pred = self.unet(noisy_latents, t, encoder_hidden_states=self._get_text_embeddings(prompts)).sample
        val_loss = F.mse_loss(noise_pred, noise)
        return {"val_loss": val_loss}

    # ------------------------------------------------------------------
    # Inference / generation                                            
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompts: List[str], num_inference_steps: int = 50, stride: Optional[int] = None):
        """Very light-weight Faster-Diffusion-style sampler with encoder caching.
        It only re-computes the UNet every `stride` steps.
        """
        stride = stride or self.default_stride
        self.eval()
        text_emb = self._get_text_embeddings(prompts)
        latents = torch.randn(
            len(prompts),
            self.unet.config.in_channels,
            64,
            64,
            device=self.device,
            dtype=self.dtype,
        )
        self.scheduler.set_timesteps(num_inference_steps)
        cached_noise_pred: Optional[torch.Tensor] = None
        last_compute_step = -1
        for i, t in enumerate(self.scheduler.timesteps):
            if cached_noise_pred is None or (i - last_compute_step) >= stride:
                cached_noise_pred = self.unet(latents, t, encoder_hidden_states=text_emb).sample
                last_compute_step = i
            latents = self.scheduler.step(cached_noise_pred, t, latents).prev_sample
        imgs = self.vae.decode(latents / 0.18215).sample
        return imgs

################################################################################
# Baseline (λ=0) diffusion model                                               
################################################################################

@register_model("diffusion_baseline")
class DiffusionBaselineModel(DiffusionFCRModel):
    """Identical to DiffusionFCRModel but with λ=0 (no FCR) and configurable stride."""

    def __init__(self, params: Dict[str, Any], device: torch.device):
        # Force lambda_fcr=0 to disable auxiliary loss.
        params = params.copy()
        params["lambda_fcr"] = 0.0
        super().__init__(params=params, device=device)

    # No additional FCR term – already disabled in parent training_step via lambda=0.

    def generate(self, prompts: List[str]):
        # Use default_stride inherited from params (FD-5 / FD-10 etc.)
        return super().generate(prompts, stride=self.default_stride)
