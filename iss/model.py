from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import DiffusionConfig, LossConfig, ModelConfig
from .diffusion import LinearNoiseScheduler


def expand_unet_conv_in(unet: nn.Module, in_channels: int) -> nn.Conv2d:
    """Expand a pretrained diffusion input layer, preserving only its noisy-latent path."""
    old = unet.conv_in
    if not isinstance(old, nn.Conv2d):
        raise TypeError("unet.conv_in must be torch.nn.Conv2d")
    if old.in_channels < 4:
        raise ValueError("A diffusion UNet input layer must have at least four latent channels.")
    if in_channels < 4:
        raise ValueError("in_channels must be at least four.")
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :4].copy_(old.weight[:, :4])
        if old.bias is not None:
            new.bias.copy_(old.bias)
    unet.conv_in = new
    if hasattr(unet, "register_to_config"):
        unet.register_to_config(in_channels=in_channels)
    elif hasattr(unet, "config"):
        try:
            unet.config["in_channels"] = in_channels
        except (TypeError, AttributeError):
            setattr(unet.config, "in_channels", in_channels)
    return new


def configure_unet_memory(
    unet: nn.Module,
    *,
    gradient_checkpointing: bool = False,
    enable_xformers: bool = False,
    channels_last: bool = False,
) -> None:
    """Apply optional Diffusers UNet memory optimizations with explicit failures."""
    if gradient_checkpointing:
        method = getattr(unet, "enable_gradient_checkpointing", None)
        if method is None:
            raise RuntimeError("This UNet does not support gradient checkpointing.")
        method()
    if enable_xformers:
        method = getattr(unet, "enable_xformers_memory_efficient_attention", None)
        if method is None:
            raise RuntimeError("This UNet does not support xFormers attention.")
        try:
            method()
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "enable_xformers=true requires a compatible xformers installation."
            ) from exc
    if channels_last:
        unet.to(memory_format=torch.channels_last)


def _time_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    scale = -math.log(10_000.0) / max(half - 1, 1)
    frequencies = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32) * scale
    )
    values = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([values.sin(), values.cos()], dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _groups(channels: int) -> int:
    for group in (8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


class TimeResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time(time)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class TinyConditionUNet(nn.Module):
    """Compact conditional UNet used to validate the complete pipeline offline."""

    def __init__(self, in_channels: int = 14, out_channels: int = 4, base: int = 32) -> None:
        super().__init__()
        time_channels = base * 4
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, base, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(base, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.down_block = TimeResBlock(base, base, time_channels)
        self.downsample = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid_block1 = TimeResBlock(base * 2, base * 2, time_channels)
        self.mid_block2 = TimeResBlock(base * 2, base * 2, time_channels)
        self.upsample = nn.Conv2d(base * 2, base, 3, padding=1)
        self.up_block = TimeResBlock(base * 2, base, time_channels)
        self.out_norm = nn.GroupNorm(_groups(base), base)
        self.conv_out = nn.Conv2d(base, out_channels, 3, padding=1)

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.shape[0] == 1 and sample.shape[0] > 1:
            timesteps = timesteps.expand(sample.shape[0])
        time = self.time_mlp(_time_embedding(timesteps, self.conv_in.out_channels).to(sample.dtype))
        first = self.conv_in(sample)
        skip = self.down_block(first, time)
        hidden = self.downsample(skip)
        hidden = self.mid_block1(hidden, time)
        hidden = self.mid_block2(hidden, time)
        hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
        hidden = self.upsample(hidden)
        hidden = self.up_block(torch.cat([hidden, skip], dim=1), time)
        return self.conv_out(F.silu(self.out_norm(hidden)))


class AnalyticAutoencoder(nn.Module):
    """A deterministic RGB-to-four-channel latent codec for fast smoke tests."""

    def __init__(self, downsample: int = 4) -> None:
        super().__init__()
        if downsample < 1:
            raise ValueError("downsample must be positive.")
        self.downsample = downsample

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        latent_rgb = F.interpolate(
            image,
            scale_factor=1.0 / self.downsample,
            mode="area",
        )
        luminance = (
            0.299 * latent_rgb[:, :1]
            + 0.587 * latent_rgb[:, 1:2]
            + 0.114 * latent_rgb[:, 2:3]
        )
        return torch.cat([latent_rgb, luminance], dim=1)

    def decode(self, latent: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            latent[:, :3], size=output_size, mode="bilinear", align_corners=False
        )


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and prediction.shape[1] != 1:
        mask = mask.expand(-1, prediction.shape[1], -1, -1)
    return ((prediction - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def _gradient_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


class ISSModel(nn.Module):
    """Two-reference conditional latent diffusion model for image stitching."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        diffusion_config: DiffusionConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config or ModelConfig()
        self.diffusion_config = diffusion_config or DiffusionConfig()
        self.scheduler = LinearNoiseScheduler(
            num_train_timesteps=self.diffusion_config.train_timesteps,
            beta_start=self.diffusion_config.beta_start,
            beta_end=self.diffusion_config.beta_end,
            beta_schedule=self.diffusion_config.beta_schedule,
        )
        self.backend = self.model_config.backend.lower()
        self.cross_attention_dim: int | None = None
        if self.backend == "tiny":
            self.vae = AnalyticAutoencoder(self.model_config.latent_downsample)
            self.unet = TinyConditionUNet(
                in_channels=self.model_config.in_channels,
                base=self.model_config.base_channels,
            )
            self.latent_scale = 1.0
        elif self.backend in {"stable-diffusion", "sd"}:
            try:
                from diffusers import AutoencoderKL, UNet2DConditionModel
                from transformers import CLIPTextModel, CLIPTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Stable Diffusion backend requires `pip install -r requirements-sd.txt`."
                ) from exc
            source = self.model_config.pretrained_model
            self.vae = AutoencoderKL.from_pretrained(source, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(source, subfolder="unet")
            expand_unet_conv_in(self.unet, self.model_config.in_channels)
            configure_unet_memory(
                self.unet,
                gradient_checkpointing=self.model_config.gradient_checkpointing,
                enable_xformers=self.model_config.enable_xformers,
                channels_last=self.model_config.channels_last,
            )
            cross_dim = self.unet.config.cross_attention_dim
            self.cross_attention_dim = int(cross_dim[0] if isinstance(cross_dim, (tuple, list)) else cross_dim)
            tokenizer = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(source, subfolder="text_encoder")
            tokens = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                empty_text_embedding = text_encoder(tokens.input_ids)[0].detach()
            if empty_text_embedding.shape[-1] != self.cross_attention_dim:
                raise RuntimeError(
                    "The text encoder hidden size does not match UNet cross_attention_dim."
                )
            self.register_buffer(
                "empty_text_embedding", empty_text_embedding, persistent=False
            )
            del text_encoder, tokenizer
            self.latent_scale = float(getattr(self.vae.config, "scaling_factor", 0.18215))
            self.vae.requires_grad_(False)
            self.vae.eval()
        else:
            raise ValueError(f"Unknown model backend: {self.model_config.backend!r}")

    def train(self, mode: bool = True) -> "ISSModel":
        """
        Set the model's training mode while keeping the VAE in evaluation mode for non-tiny backends.
        
        Args:
            mode: Whether to enable training mode.
        
        Returns:
            The model instance.
        """
        super().train(mode)
        if self.backend != "tiny":
            self.vae.eval()
        return self

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        weights: LossConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training losses for a batch.
        
        Parameters:
            batch (dict[str, torch.Tensor]): Input tensors required for loss computation.
            weights (LossConfig | None): Optional loss component weights.
        
        Returns:
            dict[str, torch.Tensor]: Total loss and detached component losses.
        """
        return self.training_losses(batch, weights)

    def encode(self, image: torch.Tensor, *, sample_posterior: bool = False) -> torch.Tensor:
        """
        Encode an image into the model's latent representation.
        
        Parameters:
            image (torch.Tensor): Input image tensor.
            sample_posterior (bool): Whether to sample from the posterior instead of using its mode.
        
        Returns:
            torch.Tensor: Encoded latent representation.
        """
        if self.backend == "tiny":
            return self.vae.encode(image)
        posterior = self.vae.encode(image).latent_dist
        latent = posterior.sample() if sample_posterior else posterior.mode()
        return latent * self.latent_scale

    def decode(self, latent: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if self.backend == "tiny":
            return self.vae.decode(latent, output_size)
        decoded = self.vae.decode(latent / self.latent_scale).sample
        if decoded.shape[-2:] != output_size:
            decoded = F.interpolate(decoded, size=output_size, mode="bilinear", align_corners=False)
        return decoded

    def _predict_noise(self, model_input: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if self.backend == "tiny":
            return self.unet(model_input, timesteps)
        hidden = self.empty_text_embedding.to(
            device=model_input.device, dtype=model_input.dtype
        ).expand(model_input.shape[0], -1, -1)
        return self.unet(model_input, timesteps, encoder_hidden_states=hidden).sample

    def _model_input(
        self,
        noisy_target: torch.Tensor,
        left_latent: torch.Tensor,
        right_latent: torch.Tensor,
        left_mask: torch.Tensor,
        right_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = [noisy_target, left_latent, right_latent]
        if self.model_config.use_masks:
            size = noisy_target.shape[-2:]
            values.extend(
                [
                    F.interpolate(left_mask, size=size, mode="nearest"),
                    F.interpolate(right_mask, size=size, mode="nearest"),
                ]
            )
        model_input = torch.cat(values, dim=1)
        if model_input.shape[1] != self.model_config.in_channels:
            raise RuntimeError(
                f"Expected {self.model_config.in_channels} UNet channels, got {model_input.shape[1]}."
            )
        return model_input

    def training_losses(
        self,
        batch: dict[str, torch.Tensor],
        weights: LossConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        weights = weights or LossConfig()
        left, right, target = batch["left"], batch["right"], batch["target"]
        left_mask, right_mask = batch["left_mask"], batch["right_mask"]
        seam_mask = batch["seam_mask"]
        with torch.no_grad():
            left_latent = self.encode(left)
            right_latent = self.encode(right)
            target_latent = self.encode(target, sample_posterior=True)
        noise = torch.randn_like(target_latent)
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (target.shape[0],),
            device=target.device,
        )
        noisy_target = self.scheduler.add_noise(target_latent, noise, timesteps)
        model_input = self._model_input(
            noisy_target, left_latent, right_latent, left_mask, right_mask
        )
        noise_prediction = self._predict_noise(model_input, timesteps)
        loss_diffusion = F.mse_loss(noise_prediction.float(), noise.float())

        predicted_x0 = self.scheduler.predict_x0(noisy_target, noise_prediction, timesteps)
        predicted_image = self.decode(predicted_x0.clamp(-4.0, 4.0), target.shape[-2:]).clamp(-1.0, 1.0)
        union = ((left_mask + right_mask) > 0).to(target.dtype)
        left_only = left_mask * (1.0 - right_mask)
        right_only = right_mask * (1.0 - left_mask)
        loss_reconstruction = _masked_l1(predicted_image, target, union)
        loss_seam = _masked_l1(predicted_image, target, seam_mask)
        loss_gradient = _gradient_l1(predicted_image, target)
        loss_preserve = _masked_l1(predicted_image, left, left_only) + _masked_l1(
            predicted_image, right, right_only
        )
        total = (
            weights.diffusion * loss_diffusion
            + weights.reconstruction * loss_reconstruction
            + weights.seam * loss_seam
            + weights.gradient * loss_gradient
            + weights.preserve * loss_preserve
        )
        return {
            "loss": total,
            "diffusion": loss_diffusion.detach(),
            "reconstruction": loss_reconstruction.detach(),
            "seam": loss_seam.detach(),
            "gradient": loss_gradient.detach(),
            "preserve": loss_preserve.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_mask: torch.Tensor,
        right_mask: torch.Tensor,
        *,
        initial_image: torch.Tensor | None = None,
        strength: float = 1.0,
        num_inference_steps: int | None = None,
        seed: int = 42,
    ) -> torch.Tensor:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be between 0 and 1.")
        left_latent = self.encode(left)
        right_latent = self.encode(right)
        if initial_image is not None and strength == 0.0:
            return initial_image.clamp(-1.0, 1.0)
        generator = torch.Generator(device=left.device).manual_seed(seed)
        noise = torch.randn(
            left_latent.shape,
            generator=generator,
            device=left.device,
            dtype=left_latent.dtype,
        )
        steps = self.scheduler.inference_timesteps(
            num_inference_steps or self.diffusion_config.inference_steps,
            left.device,
        )
        if initial_image is None or strength >= 1.0:
            sample = noise
        else:
            initial_latent = self.encode(initial_image)
            denoise_count = max(1, int(math.ceil(len(steps) * strength)))
            steps = steps[-denoise_count:]
            start = steps[0].expand(left.shape[0])
            sample = self.scheduler.add_noise(initial_latent, noise, start)
        for index, timestep in enumerate(steps):
            batch_timestep = timestep.expand(left.shape[0])
            model_input = self._model_input(
                sample, left_latent, right_latent, left_mask, right_mask
            )
            noise_prediction = self._predict_noise(model_input, batch_timestep)
            previous = steps[index + 1] if index + 1 < len(steps) else -1
            sample = self.scheduler.ddim_step(noise_prediction, timestep, previous, sample)
        return self.decode(sample, left.shape[-2:]).clamp(-1.0, 1.0)

    def trainable_parameters(self) -> Any:
        return self.unet.parameters()

    def save_model(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "backend": self.backend,
            "in_channels": self.model_config.in_channels,
            "channel_layout": (
                ["noisy_target:4", "left:4", "right:4", "left_mask:1", "right_mask:1"]
                if self.model_config.use_masks
                else ["noisy_target:4", "left:4", "right:4"]
            ),
            "pretrained_model": self.model_config.pretrained_model,
        }
        (output_dir / "model_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self.backend == "tiny":
            path = output_dir / "model.pt"
            torch.save(self.unet.state_dict(), path)
            return path
        self.unet.save_pretrained(output_dir / "unet", safe_serialization=True)
        return output_dir / "unet"

    def load_model(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint)
        if self.backend == "tiny":
            path = checkpoint / "model.pt" if checkpoint.is_dir() else checkpoint
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.unet.load_state_dict(state)
            return
        from diffusers import UNet2DConditionModel

        path = checkpoint / "unet" if (checkpoint / "unet").exists() else checkpoint
        loaded = UNet2DConditionModel.from_pretrained(path)
        self.unet.load_state_dict(loaded.state_dict(), strict=True)
        del loaded
