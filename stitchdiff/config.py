from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    backend: str = "tiny"
    pretrained_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    use_masks: bool = True
    base_channels: int = 32
    latent_downsample: int = 4
    gradient_checkpointing: bool = False
    enable_xformers: bool = False
    channels_last: bool = False

    @property
    def in_channels(self) -> int:
        return 14 if self.use_masks else 12


@dataclass
class DataConfig:
    root: str = "data/demo"
    image_height: int = 64
    image_width: int = 128
    num_workers: int = 0


@dataclass
class LossConfig:
    diffusion: float = 1.0
    reconstruction: float = 0.10
    seam: float = 0.50
    gradient: float = 0.10
    preserve: float = 0.50


@dataclass
class TrainConfig:
    output_dir: str = "outputs/tiny"
    batch_size: int = 2
    max_steps: int = 1000
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-2
    grad_accumulation: int = 1
    grad_clip: float = 1.0
    checkpoint_every: int = 250
    validation_every: int = 250
    validation_batches: int = 2
    log_every: int = 10
    seed: int = 42
    device: str = "auto"
    mixed_precision: str = "no"


@dataclass
class DiffusionConfig:
    train_timesteps: int = 1000
    inference_steps: int = 30
    strength: float = 0.35
    beta_schedule: str = "linear"
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2


@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    known = {field.name for field in instance.__dataclass_fields__.values()}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"Unknown config keys for {type(instance).__name__}: {sorted(unknown)}"
        )
    for key, value in values.items():
        setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    allowed = {"model", "data", "loss", "train", "diffusion"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    return ProjectConfig(
        model=_merge_dataclass(ModelConfig(), raw.get("model", {})),
        data=_merge_dataclass(DataConfig(), raw.get("data", {})),
        loss=_merge_dataclass(LossConfig(), raw.get("loss", {})),
        train=_merge_dataclass(TrainConfig(), raw.get("train", {})),
        diffusion=_merge_dataclass(DiffusionConfig(), raw.get("diffusion", {})),
    )


def save_config(config: ProjectConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, allow_unicode=True, sort_keys=False)
