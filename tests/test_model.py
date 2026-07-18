import sys
from types import SimpleNamespace

import torch
from torch import nn

from iss.config import DiffusionConfig, LossConfig, ModelConfig
from iss.diffusion import (
    LinearNoiseScheduler,
    diffusion_training_target,
    predict_clean_sample,
)
from iss.model import (
    ISSModel,
    configure_unet_memory,
    expand_unet_conv_in,
)


class DummyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 8, 3, padding=1)
        self.config = {"in_channels": 4}


def test_default_full_backend_uses_stable_diffusion_2():
    assert ModelConfig().pretrained_model == "sd2-community/stable-diffusion-2"


def test_expand_conv_preserves_noisy_channels_and_zeros_conditions():
    unet = DummyUNet()
    old_weight = unet.conv_in.weight.detach().clone()
    old_bias = unet.conv_in.bias.detach().clone()

    expand_unet_conv_in(unet, 14)

    assert unet.conv_in.in_channels == 14
    assert unet.config["in_channels"] == 14
    torch.testing.assert_close(unet.conv_in.weight[:, :4], old_weight)
    torch.testing.assert_close(unet.conv_in.weight[:, 4:], torch.zeros_like(unet.conv_in.weight[:, 4:]))
    torch.testing.assert_close(unet.conv_in.bias, old_bias)


def _batch(batch_size: int = 1):
    target = torch.rand(batch_size, 3, 32, 64) * 2 - 1
    left_mask = torch.zeros(batch_size, 1, 32, 64)
    right_mask = torch.zeros_like(left_mask)
    left_mask[..., :42] = 1
    right_mask[..., 22:] = 1
    left = target * left_mask
    right = target * right_mask
    seam = torch.zeros_like(left_mask)
    seam[..., 28:36] = 1
    return {
        "left": left,
        "right": right,
        "target": target,
        "left_mask": left_mask,
        "right_mask": right_mask,
        "seam_mask": seam,
    }


def test_tiny_training_loss_and_sampling():
    model = ISSModel(
        ModelConfig(backend="tiny", use_masks=True, base_channels=8, latent_downsample=4),
        DiffusionConfig(train_timesteps=10, inference_steps=2),
    )
    losses = model.training_losses(_batch(), LossConfig())
    losses["loss"].backward()
    sample = model.sample(
        _batch()["left"],
        _batch()["right"],
        _batch()["left_mask"],
        _batch()["right_mask"],
        num_inference_steps=2,
    )

    assert torch.isfinite(losses["loss"])
    assert model.unet.conv_in.weight.grad is not None
    assert sample.shape == (1, 3, 32, 64)
    assert torch.isfinite(sample).all()


def test_twelve_channel_ablation():
    model = ISSModel(
        ModelConfig(backend="tiny", use_masks=False, base_channels=8),
        DiffusionConfig(train_timesteps=10),
    )
    assert model.unet.conv_in.in_channels == 12
    assert torch.isfinite(model.training_losses(_batch())["loss"])


def test_scaled_linear_scheduler_endpoints():
    scheduler = LinearNoiseScheduler(
        num_train_timesteps=10,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
    )
    torch.testing.assert_close(scheduler.betas[0], torch.tensor(0.00085))
    torch.testing.assert_close(scheduler.betas[-1], torch.tensor(0.012))
    assert torch.all(scheduler.betas[1:] > scheduler.betas[:-1])


def test_velocity_prediction_target_recovers_clean_sample():
    scheduler = LinearNoiseScheduler(
        num_train_timesteps=10,
        prediction_type="v_prediction",
    )
    clean = torch.randn(2, 4, 4, 4)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([2, 7])
    noisy = scheduler.add_noise(clean, noise, timesteps)
    velocity = diffusion_training_target(scheduler, clean, noise, timesteps)

    recovered = predict_clean_sample(scheduler, noisy, velocity, timesteps)

    torch.testing.assert_close(recovered, clean)


def test_memory_options_are_applied():
    class MemoryUNet(DummyUNet):
        def __init__(self):
            super().__init__()
            self.checkpointing = False
            self.xformers = False

        def enable_gradient_checkpointing(self):
            self.checkpointing = True

        def enable_xformers_memory_efficient_attention(self):
            self.xformers = True

    unet = MemoryUNet()
    configure_unet_memory(
        unet,
        gradient_checkpointing=True,
        enable_xformers=True,
        channels_last=True,
    )
    assert unet.checkpointing is True
    assert unet.xformers is True
    assert unet.conv_in.weight.is_contiguous(memory_format=torch.channels_last)


def test_stable_diffusion_component_wiring_without_weights(monkeypatch):
    class FakeDistribution:
        def __init__(self, value):
            self.value = value

        def sample(self):
            return self.value

        def mode(self):
            return self.value

    class FakeVAE(nn.Module):
        config = SimpleNamespace(scaling_factor=1.0)

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def encode(self, image):
            latent = torch.cat([image, image.mean(dim=1, keepdim=True)], dim=1)
            return SimpleNamespace(latent_dist=FakeDistribution(latent))

        def decode(self, latent):
            return SimpleNamespace(sample=latent[:, :3])

    class FakeConfig(dict):
        cross_attention_dim = 6

    class FakeUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = nn.Conv2d(4, 4, 3, padding=1)
            self.config = FakeConfig(in_channels=4)

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def register_to_config(self, **values):
            self.config.update(values)

        def forward(self, sample, _timesteps, encoder_hidden_states=None):
            assert encoder_hidden_states.shape[-1] == 6
            return SimpleNamespace(sample=self.conv_in(sample))

    class FakeTokenizer:
        model_max_length = 5

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_ids=torch.zeros(1, 5, dtype=torch.long))

    class FakeTextEncoder(nn.Module):
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def forward(self, input_ids):
            return (torch.zeros(input_ids.shape[0], input_ids.shape[1], 6),)

    class FakeSchedulerConfig(SimpleNamespace):
        def __iter__(self):
            return iter(vars(self))

    class FakeDDPMScheduler:
        def __init__(self):
            self.config = FakeSchedulerConfig(
                num_train_timesteps=10,
                prediction_type="v_prediction",
            )
            self.alphas_cumprod = torch.linspace(0.99, 0.1, 10)

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def register_to_config(self, **values):
            for key, value in values.items():
                setattr(self.config, key, value)

        def add_noise(self, clean, noise, timesteps):
            alpha = self.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
            return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

        def get_velocity(self, clean, noise, timesteps):
            alpha = self.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
            return alpha.sqrt() * noise - (1.0 - alpha).sqrt() * clean

    class FakeDDIMScheduler:
        init_noise_sigma = 1.0

        @classmethod
        def from_config(cls, config):
            instance = cls()
            instance.config = config
            instance.alphas_cumprod = torch.linspace(0.99, 0.1, 10)
            return instance

        def set_timesteps(self, num_inference_steps, device=None):
            self.timesteps = torch.linspace(
                9, 0, num_inference_steps, device=device
            ).round().long()

        def scale_model_input(self, sample, _timestep):
            return sample

        def add_noise(self, clean, noise, timesteps):
            alpha = self.alphas_cumprod.to(clean.device)[timesteps].reshape(
                -1, 1, 1, 1
            )
            return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

        def step(self, _model_output, _timestep, sample, **_kwargs):
            return (sample,)

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            AutoencoderKL=FakeVAE,
            DDIMScheduler=FakeDDIMScheduler,
            DDPMScheduler=FakeDDPMScheduler,
            UNet2DConditionModel=FakeUNet,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(CLIPTextModel=FakeTextEncoder, CLIPTokenizer=FakeTokenizer),
    )
    model = ISSModel(
        ModelConfig(backend="stable-diffusion", use_masks=True),
        DiffusionConfig(train_timesteps=10),
    )
    losses = model.training_losses(_batch())
    losses["loss"].backward()
    batch = _batch()
    sample = model.sample(
        batch["left"],
        batch["right"],
        batch["left_mask"],
        batch["right_mask"],
        num_inference_steps=2,
    )

    assert model.unet.conv_in.in_channels == 14
    assert model.empty_text_embedding.shape == (1, 5, 6)
    assert model.prediction_type == "v_prediction"
    assert all(not parameter.requires_grad for parameter in model.vae.parameters())
    assert model.unet.conv_in.weight.grad is not None
    assert sample.shape == batch["target"].shape
    assert torch.isfinite(sample).all()
