from __future__ import annotations

import torch
from torch import nn


SUPPORTED_PREDICTION_TYPES = {"epsilon", "sample", "v_prediction"}


def scheduler_prediction_type(scheduler: object) -> str:
    """
    Normalize and validate a scheduler's diffusion prediction type.
    
    The value is read from the scheduler configuration or scheduler attribute,
    defaulting to ``"epsilon"`` when unspecified. Supported values are
    ``"epsilon"``, ``"sample"``, and ``"v_prediction"``.
    
    Returns:
        str: The normalized prediction type.
    
    Raises:
        ValueError: If the prediction type is unsupported.
    """
    config = getattr(scheduler, "config", None)
    prediction_type = getattr(config, "prediction_type", None)
    if prediction_type is None and isinstance(config, dict):
        prediction_type = config.get("prediction_type")
    if prediction_type is None:
        prediction_type = getattr(scheduler, "prediction_type", "epsilon")
    prediction_type = str(prediction_type)
    if prediction_type not in SUPPORTED_PREDICTION_TYPES:
        raise ValueError(
            f"Unsupported diffusion prediction_type {prediction_type!r}; expected one of "
            f"{sorted(SUPPORTED_PREDICTION_TYPES)}."
        )
    return prediction_type


def scheduler_train_timesteps(scheduler: object) -> int:
    """
    Get the number of training timesteps configured for a scheduler.
    
    Parameters:
    	scheduler (object): Scheduler whose training timestep count is read.
    
    Returns:
    	int: Configured number of training timesteps.
    
    Raises:
    	AttributeError: If the scheduler does not expose a training timestep count.
    """
    config = getattr(scheduler, "config", None)
    value = getattr(config, "num_train_timesteps", None)
    if value is None and isinstance(config, dict):
        value = config.get("num_train_timesteps")
    if value is None:
        value = getattr(scheduler, "num_train_timesteps", None)
    if value is None:
        raise AttributeError("The scheduler does not expose num_train_timesteps.")
    return int(value)


def _extract_alpha(
    scheduler: object,
    timesteps: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    """Extract per-timestep cumulative alpha values reshaped for broadcasting with a sample.
    
    Parameters:
    	scheduler (object): Scheduler containing the cumulative alpha values.
    	timesteps (torch.Tensor): Timestep indices used to select alpha values.
    	sample (torch.Tensor): Tensor whose device, data type, and dimensions determine the output format.
    
    Returns:
    	(torch.Tensor): Selected cumulative alpha values on the sample's device and data type, with singleton dimensions for broadcasting.
    """
    alphas_cumprod = getattr(scheduler, "alphas_cumprod")
    selected = alphas_cumprod.to(device=sample.device)[timesteps.long()]
    return selected.reshape((-1,) + (1,) * (sample.ndim - 1)).to(sample.dtype)


def diffusion_training_target(
    scheduler: object,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Build the training target for the scheduler's prediction parameterization.
    
    Returns:
        A noise, clean-sample, or velocity target according to the scheduler's
        prediction type.
    """
    prediction_type = scheduler_prediction_type(scheduler)
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "sample":
        return clean
    get_velocity = getattr(scheduler, "get_velocity", None)
    if get_velocity is not None:
        return get_velocity(clean, noise, timesteps)
    alpha = _extract_alpha(scheduler, timesteps, clean)
    return alpha.sqrt() * noise - (1.0 - alpha).sqrt() * clean


def predict_clean_sample(
    scheduler: object,
    noisy: torch.Tensor,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Recover the clean sample from a noisy sample and model output.
    
    Parameters:
        scheduler (object): Scheduler providing the model prediction parameterization and cumulative alpha values.
        noisy (torch.Tensor): Noisy sample at the specified timesteps.
        model_output (torch.Tensor): Model output in the scheduler's configured parameterization.
        timesteps (torch.Tensor): Timesteps associated with the samples.
    
    Returns:
        torch.Tensor: Reconstructed clean sample.
    """
    prediction_type = scheduler_prediction_type(scheduler)
    if prediction_type == "sample":
        return model_output
    alpha = _extract_alpha(scheduler, timesteps, noisy)
    if prediction_type == "epsilon":
        return (
            noisy - (1.0 - alpha).sqrt() * model_output
        ) / alpha.sqrt().clamp_min(1.0e-6)
    return alpha.sqrt() * noisy - (1.0 - alpha).sqrt() * model_output


class LinearNoiseScheduler(nn.Module):
    """Small dependency-free DDPM training scheduler with deterministic DDIM sampling."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
    ) -> None:
        """
        Initialize a lightweight diffusion noise scheduler.
        
        Args:
            num_train_timesteps: Number of training diffusion steps; must be at least 2.
            beta_start: Initial noise variance.
            beta_end: Final noise variance.
            beta_schedule: Beta schedule, either ``"linear"`` or ``"scaled_linear"``.
            prediction_type: Model output parameterization: ``"epsilon"``, ``"sample"``,
                or ``"v_prediction"``.
        
        Raises:
            ValueError: If the number of timesteps, beta schedule, or prediction type is
                invalid.
        """
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
        if prediction_type not in SUPPORTED_PREDICTION_TYPES:
            raise ValueError(
                f"prediction_type must be one of {sorted(SUPPORTED_PREDICTION_TYPES)}."
            )
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alpha_cumprod, persistent=False)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        selected = values.to(sample.device)[timesteps.long()]
        return selected.reshape((-1,) + (1,) * (sample.ndim - 1)).to(sample.dtype)

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to clean samples according to the specified diffusion timesteps.
        
        Parameters:
            clean (torch.Tensor): Clean samples.
            noise (torch.Tensor): Noise to add.
            timesteps (torch.Tensor): Diffusion timestep for each sample.
        
        Returns:
            torch.Tensor: Noisy samples."""
        alpha = self._extract(self.alphas_cumprod, timesteps, clean)
        return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

    def predict_x0(
        self, noisy: torch.Tensor, model_output: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Recover the clean sample from a noisy sample and the model output.
        
        Parameters:
            noisy (torch.Tensor): The noisy sample.
            model_output (torch.Tensor): The model output for the configured prediction type.
            timesteps (torch.Tensor): The diffusion timestep for each sample.
        
        Returns:
            torch.Tensor: The predicted clean sample.
        """
        return predict_clean_sample(self, noisy, model_output, timesteps)

    def training_target(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Build the model training target for the configured prediction type.
        
        Parameters:
        	clean (torch.Tensor): Clean samples.
        	noise (torch.Tensor): Noise added to the clean samples.
        	timesteps (torch.Tensor): Diffusion timestep for each sample.
        
        Returns:
        	torch.Tensor: Training target corresponding to the scheduler's prediction type.
        """
        return diffusion_training_target(self, clean, noise, timesteps)

    def inference_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        """
        Create descending diffusion timesteps for inference.
        
        Parameters:
        	num_inference_steps (int): Number of inference steps to generate.
        	device (torch.device): Device on which to create the timestep tensor.
        
        Returns:
        	torch.Tensor: A one-dimensional tensor of unique, descending timestep indices.
        """
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
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        previous_timestep: int | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the deterministic DDIM denoising update for one timestep.
        
        Parameters:
        	model_output (torch.Tensor): Model prediction for the current noisy sample.
        	timestep (int | torch.Tensor): Current diffusion timestep.
        	previous_timestep (int | torch.Tensor): Timestep used for the updated sample.
        	sample (torch.Tensor): Current noisy sample.
        
        Returns:
        	torch.Tensor: Sample denoised to the previous timestep.
        """
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
        predicted_x0 = predict_clean_sample(
            self,
            sample,
            model_output,
            torch.full(
                (sample.shape[0],),
                t,
                device=sample.device,
                dtype=torch.long,
            ),
        )
        if self.prediction_type == "epsilon":
            predicted_noise = model_output
        else:
            predicted_noise = (
                sample - alpha_t.sqrt() * predicted_x0
            ) / (1.0 - alpha_t).sqrt().clamp_min(1.0e-6)
        direction = (1.0 - alpha_previous).sqrt() * predicted_noise
        return alpha_previous.sqrt() * predicted_x0 + direction
