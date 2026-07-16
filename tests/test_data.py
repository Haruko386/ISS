from pathlib import Path

import numpy as np
from PIL import Image

from iss.data import StitchTripletDataset, prepare_synthetic_dataset


def test_prepare_and_load_dataset(tmp_path: Path):
    x = np.linspace(0, 255, 96, dtype=np.uint8)[None, :, None]
    panorama = np.tile(x, (48, 1, 3))
    source = tmp_path / "panorama.png"
    Image.fromarray(panorama).save(source)

    manifest = prepare_synthetic_dataset(
        source,
        tmp_path / "dataset",
        width=64,
        height=32,
        samples_per_image=2,
        validation_fraction=0.0,
        residual_shift=1,
    )
    dataset = StitchTripletDataset(manifest, width=64, height=32)
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["left"].shape == (3, 32, 64)
    assert sample["left_mask"].shape == (1, 32, 64)
    assert sample["target"].min() >= -1.0
    assert sample["target"].max() <= 1.0
