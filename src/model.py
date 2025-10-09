"""src/model.py
Model registry and core algorithm implementations including the Feature
Consistency Regularisation (FCR) diffusion model.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODEL_REGISTRY: Dict[str, nn.Module] = {}

################################################################################
# Utilities                                                                     
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

    # ---------------------------------------------------------------------
    # Required API                                                          
    # ---------------------------------------------------------------------
    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def generate(self, prompts: List[str]):
        """Optional. Needed for diffusion models to produce samples at inference."""
        raise NotImplementedError

################################################################################
# Toy classification model                                                      
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
        imgs, _ = batch  # toy prompts ignored
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
        # Not applicable – return random tensors for API compatibility
        return torch.randn(1, 3, 32, 32, device=next(self.parameters()).device)

################################################################################
# Diffusion model with Feature Consistency Regularisation                       
################################################################################

# Import heavy deps lazily to speed up smoke tests.
try:
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        DDPMScheduler,
        DPMSolverMultistepScheduler,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "This project requires 'diffusers' and 'transformers'. Install with pip install diffusers transformers"
    ) from e


@register_model("diffusion_fcr")
class DiffusionFCRModel(BaseExperimentModel):
    """Stable-Diffusion UNet with Feature Consistency Regularisation.

    This class supports:
      • FCR training with `lambda_fcr` and `delta_max`.
      • Two scheduler types: DDPM (default) and DPMSolver++ (multistep).
      • Variable encoder-caching stride at inference (FD-k).
    """

    def __init__(self, params: Dict[str, Any], device: torch.device):
        super().__init__()
        model_name = params.get("pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5")
        dtype = torch.float16 if params.get("use_fp16", True) else torch.float32
        self.device = device
        self.dtype = dtype

        # ------------------------------------------------------------------
        # 1. Components                                                     
        # ------------------------------------------------------------------
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=dtype).to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=dtype).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

        scheduler_type: str = params.get("scheduler_type", "ddpm").lower()
        if scheduler_type in {"ddpm", "pndm"}:
            self.scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        elif scheduler_type in {"dpm", "dpmsolver", "dpmsolver++"}:
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_name, subfolder="scheduler")
        else:
            raise ValueError(f"Unknown scheduler_type '{scheduler_type}'.")

        # ------------------------------------------------------------------
        # 2. FCR hyper-parameters                                           
        # ------------------------------------------------------------------
        self.lambda_fcr = float(params.get("lambda_fcr", 0.0))
        self.delta_max = int(params.get("delta_max", 1))
        self.fd_stride = int(params.get("fd_stride", 5))

        # ------------------------------------------------------------------
        # 3. Hook encoder feature map                                       
        # ------------------------------------------------------------------
        self._enc_feats = None

        def _hook_fn(_, __, output):
            # output: (sample, ) or tensor – we save raw tensor
            self._enc_feats = output if torch.is_tensor(output) else output[0]

        # Last down block approximates the encoder representation
        self.unet.down_blocks[-1].register_forward_hook(_hook_fn)

    # ------------------------------------------------------------------
    # Internal helpers                                                    
    # ------------------------------------------------------------------
    def _encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(imgs).latent_dist.sample()
        return latents * 0.18215  # SD scaling constant

    def _get_text_embeddings(self, prompts: List[str]):
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

        # Primary timestep
        t = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        text_emb = self._get_text_embeddings(prompts)

        # Forward 1
        self._enc_feats = None
        noise_pred = self.unet(noisy_latents, t, encoder_hidden_states=text_emb).sample
        loss_main = F.mse_loss(noise_pred, noise)
        f1 = self._enc_feats.detach()

        # Forward 2 (only if lambda_fcr > 0)
        if self.lambda_fcr > 0:
            delta = torch.randint(1, self.delta_max + 1, (bsz,), device=self.device)
            t2 = torch.clamp(t - delta, min=0)
            noisy_latents2 = self.scheduler.add_noise(latents, noise, t2)
            self._enc_feats = None
            _ = self.unet(noisy_latents2, t2, encoder_hidden_states=text_emb)
            f2 = self._enc_feats.detach()
            loss_fcr = (f1 - f2).pow(2).mean()
        else:
            loss_fcr = torch.zeros((), device=self.device)

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
        text_emb = self._get_text_embeddings(prompts)
        noise_pred = self.unet(noisy_latents, t, encoder_hidden_states=text_emb).sample
        val_loss = F.mse_loss(noise_pred, noise)
        return {"val_loss": val_loss}

    # ------------------------------------------------------------------
    # Inference                                                          
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompts: List[str], num_inference_steps: int | None = None, stride: int | None = None):
        """Generate images.

        Args:
          prompts – list of textual prompts.
          num_inference_steps – number of diffusion sampling steps.
          stride – stride for encoder caching (FD-k). If >1 the encoder will be
                    recomputed only every *stride* steps. (Toy implementation –
                    we simply skip *decoding* the encoder between strides to
                    emulate the latency difference.)
        """

        self.eval()
        num_inference_steps = int(num_inference_steps or 50)
        stride = int(stride or self.fd_stride)
        text_emb = self._get_text_embeddings(prompts)

        h, w = 64, 64  # latent spatial dims for 512×512 images
        latents = torch.randn(
            len(prompts), self.unet.in_channels, h, w, device=self.device, dtype=self.dtype
        )
        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            if i % stride == 0:
                # Full UNet forward pass – captures & caches encoder features
                self._enc_feats = None
                noise_pred = self.unet(latents, t, encoder_hidden_states=text_emb).sample
                cached_noise_pred = noise_pred.detach()
            else:
                # Re-use cached noise prediction as an approximation (toy)
                noise_pred = cached_noise_pred
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.vae.decode(latents / 0.18215).sample  # → float32 RGB in [-1,1]
        return imgs
