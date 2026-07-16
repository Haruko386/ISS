from __future__ import annotations

import torch
from torch import nn


class LinearNoiseScheduler(nn.Module):
    """Small dependency-free DDPM training scheduler with deterministic DDIM sampling."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        beta_schedule: str = "linear",
    ) -> None:
        super().__init__()
        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least 2.")
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_train_timesteps,
                dtype=torch.float32,
            ).square()
        else:
            raise ValueError("beta_schedule must be 'linear' or 'scaled_linear'.")
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.num_train_timesteps = num_train_timesteps
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alpha_cumprod, persistent=False)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        selected = values.to(sample.device)[timesteps.long()]
        return selected.reshape((-1,) + (1,) * (sample.ndim - 1)).to(sample.dtype)

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alpha = self._extract(self.alphas_cumprod, timesteps, clean)
        return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

    def predict_x0(
        self, noisy: torch.Tensor, noise_prediction: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alpha = self._extract(self.alphas_cumprod, timesteps, noisy)
        return (noisy - (1.0 - alpha).sqrt() * noise_prediction) / alpha.sqrt().clamp_min(1.0e-6)

    def inference_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        steps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
            device=device,
        )
        return steps.round().long().unique_consecutive()

    def ddim_step(
        self,
        noise_prediction: torch.Tensor,
        timestep: int | torch.Tensor,
        previous_timestep: int | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        t = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        previous = (
            int(previous_timestep.item())
            if isinstance(previous_timestep, torch.Tensor)
            else int(previous_timestep)
        )
        alpha_t = self.alphas_cumprod[t].to(device=sample.device, dtype=sample.dtype)
        alpha_previous = (
            self.alphas_cumprod[previous].to(device=sample.device, dtype=sample.dtype)
            if previous >= 0
            else torch.ones((), device=sample.device, dtype=sample.dtype)
        )
        predicted_x0 = (
            sample - (1.0 - alpha_t).sqrt() * noise_prediction
        ) / alpha_t.sqrt().clamp_min(1.0e-6)
        direction = (1.0 - alpha_previous).sqrt() * noise_prediction
        return alpha_previous.sqrt() * predicted_x0 + direction
