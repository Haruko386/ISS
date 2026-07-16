import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from iss.config import (
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    ProjectConfig,
    TrainConfig,
)
from iss.data import prepare_synthetic_dataset
from iss.trainer import ISSTrainer


def _config(data: Path, output: Path, steps: int) -> ProjectConfig:
    return ProjectConfig(
        model=ModelConfig(backend="tiny", base_channels=8, latent_downsample=4),
        data=DataConfig(root=str(data), image_height=32, image_width=64, num_workers=0),
        train=TrainConfig(
            output_dir=str(output),
            batch_size=1,
            max_steps=steps,
            checkpoint_every=0,
            validation_every=1,
            validation_batches=1,
            log_every=1,
            device="cpu",
        ),
        diffusion=DiffusionConfig(
            train_timesteps=10,
            inference_steps=1,
            strength=0.0,
        ),
    )


def test_training_resume_validation_and_best_checkpoint(tmp_path: Path):
    panorama = np.zeros((32, 64, 3), dtype=np.uint8)
    panorama[..., 0] = np.arange(64, dtype=np.uint8)[None] * 4
    source = tmp_path / "panorama.png"
    Image.fromarray(panorama).save(source)
    data = tmp_path / "data"
    prepare_synthetic_dataset(
        source,
        data,
        width=64,
        height=32,
        samples_per_image=2,
        validation_fraction=0.5,
        residual_shift=0,
    )
    output = tmp_path / "run"
    first = ISSTrainer(_config(data, output, 1)).train()

    fork_output = tmp_path / "forked-run"
    forked = ISSTrainer(_config(data, fork_output, 2), resume_from=first)
    assert math.isinf(forked.best_seam_mae)
    forked.train()
    assert (fork_output / "best" / "model.pt").exists()

    resumed = ISSTrainer(_config(data, output, 2), resume_from=first)
    assert resumed.start_step == 1
    final = resumed.train()
    summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    metric_records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert final == output / "final"
    assert (output / "best" / "model.pt").exists()
    assert (output / "validation" / "step-000002.png").exists()
    assert summary["start_step"] == 1
    assert summary["steps"] == 2
    assert summary["resumed_from"] == str(first)
    assert any(record["type"] == "validation" for record in metric_records)
