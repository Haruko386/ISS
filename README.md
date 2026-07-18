<h1 align="center"><strong>ISS: Image Stiching Super based on diffusion model</strong></h1>

ISS is a geometry-guided conditional diffusion project for stitching two overlapping images into one panorama.

The pipeline has two stages:

1. SIFT/ORB feature matching, RANSAC homography estimation, warping, masks, and a coarse feather-blended panorama.
2. A conditional latent diffusion UNet that uses the aligned left and right images to remove seams, exposure differences, and small residual misalignments.

The multi-latent input design is inspired by [Haruko386/ApDepth](https://github.com/Haruko386/ApDepth).

The full training backend is based on Stable Diffusion 2 (`sd2-community/stable-diffusion-2`).
ISS keeps its own geometry alignment, multi-image conditioning, losses, and sampling flow while
reusing the SD2 VAE, OpenCLIP text-conditioning space, U-Net, and scheduler configuration.

## Model architecture

The default model uses 14 input channels:

```text
noisy target latent    4 channels
left image latent      4 channels
right image latent     4 channels
left valid mask        1 channel
right valid mask       1 channel
---------------------------------
total                  14 channels
```

A 12-channel ablation is also supported:

```text
[noisy target latent (4), left latent (4), right latent (4)]
```

During training, the dataset ground-truth panorama is encoded and noised:

```text
GT panorama -> VAE -> z0 -> add random noise at timestep t -> zt

UNet input:  [zt, left_latent, right_latent, masks]
UNet target: scheduler target (velocity v for the default SD2 configuration)
```

The primary objective is scheduler-target MSE. ISS supports `epsilon`, `sample`, and `v_prediction`; the SD2 configuration uses velocity prediction and converts it correctly back to `x0` for the additional reconstruction, seam, gradient, and content-preservation losses.

During inference, two modes are available:

- Practical refinement mode: start from a noised coarse panorama and preserve pixels outside the seam region.
- Pure diffusion mode: start from random latent noise and generate the complete output from the left/right conditions.

## Requirements

- Python 3.10-3.12
- PyTorch 2.2 or newer
- A CUDA GPU is strongly recommended for full Stable Diffusion training
- CPU execution is supported by the tiny smoke-test configurations

## Installation

Create a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the base dependencies:

```bash
pip install -r requirements.txt
```

Install the Stable Diffusion dependencies when using `configs/sd-tiny.yaml` or `configs/sd2.yaml`:

```bash
pip install -r requirements-sd.txt
```

The project pins Diffusers to the tested 0.39.x range; use `requirements-sd.txt` so its scheduler and model APIs stay compatible with ISS.

Optionally install the project as a command-line package:

```bash
pip install -e .
iss doctor
```

Without editable installation, use `python main.py` for every command.

## Check the environment

```bash
python main.py doctor
```

This reports the Python, PyTorch, OpenCV, Diffusers, Transformers, and Accelerate versions, CUDA availability, and the selected device.

## Quick start

The repository includes `res/1.jpg` and `res/2.jpg`. Run the complete CPU pipeline with:

```bash
python main.py demo
```

The demo performs alignment, creates a synthetic training set, trains the tiny diffusion model for a few steps, and runs inference.

Generated files are written to:

```text
outputs/demo/
|-- alignment/             aligned canvases, masks, and coarse panorama
|-- data/                  generated training triplets and manifest
|-- training/
|   |-- best/              lowest validation seam MAE checkpoint
|   |-- final/             final checkpoint
|   |-- validation/        left/right/coarse/prediction/GT previews
|   `-- metrics.jsonl      training and validation metrics
|-- stitched.png           diffusion result
`-- stitched_coarse.png    geometry-only baseline
```

The tiny backend verifies the full data and diffusion pipeline. It is not a pretrained image generator and is not intended to produce high-quality results after only a few steps.

## Dataset download

The current supervised pipeline needs complete RGB panoramas as ground truth. The `prepare` command converts every source panorama into multiple samples containing an aligned left image, an aligned right image, validity masks, a seam mask, and the original panorama as the GT target. A dataset containing only unaligned left/right pairs and no GT panorama is not directly compatible with this training pipeline.

Downloaded and generated data should be stored under the repository's `dataset` directory:

```text
dataset/
|-- .placeholder
|-- raw/                         downloaded source panoramas
|   `-- PanoHK360/
`-- prepared/                    generated ISS manifests and images
    |-- panoramas/
    `-- sd-tiny/
```

The contents of `dataset/` are ignored by Git except for `.placeholder`.

### Recommended: PanoHK360 RGB panoramas

[PanoHK360](https://huggingface.co/datasets/adadai3132/PanoHK360) provides high-resolution equirectangular urban panoramas and is published under CC BY 4.0. ISS needs only the RGB images in a `pano_raw` directory; do not pass its depth maps, normal maps, or perspective ROI crops to `prepare`.

Install the current Hugging Face CLI:

```bash
pip install -U huggingface_hub
```

Download the filtered RGB panorama directory:

```bash
hf download adadai3132/PanoHK360 \
  --repo-type dataset \
  --include "R101 20230413--filter/pano_raw/**" \
  --local-dir dataset/raw/PanoHK360
```

Convert the downloaded panoramas into this project's supervised training format:

```bash
python main.py prepare --panoramas "dataset/raw/PanoHK360/R101 20230413--filter/pano_raw" --output dataset/prepared/panoramas --width 1024 --height 512 --samples-per-image 16 --validation-fraction 0.1 --seed 42
```

Re-running `hf download` updates or resumes the local download. Read the dataset card before use, preserve its attribution, and review images for privacy or licensing requirements relevant to the intended application.

### Smaller optional source: SUN360 mirror

For a smaller pipeline experiment, the [Everloom/SUN360 mirror](https://huggingface.co/datasets/Everloom/SUN360) contains train and test image folders. Download only its RGB training directory so that auxiliary labels are not treated as GT images:

```bash
hf download Everloom/SUN360 \
  "train/RGB/" \
  --repo-type dataset \
  --local-dir dataset/raw/SUN360
```

The mirror currently has no dataset card or declared license. Verify the original SUN360 usage terms before training or redistributing derived data. For a code-only smoke test with no external download, use `python main.py demo` instead.

### Custom panoramas

You can use your own `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, or `.webp` panoramas. Place them in `dataset/raw/custom/`, then run:

```bash
python main.py prepare --panoramas dataset/raw/custom --output dataset/prepared/panoramas --width 1024 --height 512 --samples-per-image 16
```

Use source images that you are allowed to process. Wider equirectangular images are preferred, but ordinary RGB images can be used to validate the software pipeline.

## Command overview

```text
doctor      inspect the runtime environment
align       align two input images onto a shared panorama canvas
prepare     generate left/right/GT training triplets from panoramas
prepare-udis copy/extract UDIS-D and convert pairs into ISS triplets
train       train or resume a 12/14-channel diffusion model
evaluate    evaluate a checkpoint on a train or validation split
infer       align and stitch a pair using a trained checkpoint
demo        run the complete tiny pipeline
```

Show all options with:

```bash
python main.py --help
python main.py train --help
python main.py infer --help
```

## 1. Align two images

```bash
python main.py align --left res/1.jpg --right res/2.jpg --output outputs/alignment
```

The alignment directory contains:

```text
left.png
right.png
left_mask.png
right_mask.png
seam_mask.png
coarse.png
alignment.json
```

`left.png` and `right.png` are independent images warped onto the same panorama canvas. Empty regions remain invalid according to their masks.

Alignment fails with an explicit error when there are not enough reliable matches or when the estimated canvas is unreasonably large. Large parallax scenes may require optical flow or a mesh-warp alignment module before diffusion refinement.

## 2. Prepare a training dataset

Create training triplets from one complete panorama or a directory of panoramas. The following example assumes the source images were placed or downloaded under `dataset/raw/custom`:

```bash
python main.py prepare --panoramas dataset/raw/custom --output dataset/prepared/panoramas --width 1024 --height 512 --samples-per-image 16
```

The generator creates overlapping left/right conditions, masks, a GT panorama, exposure and color changes, and small residual shifts.

The generated `manifest.jsonl` uses this format:

```json
{
  "left": "left/000000.png",
  "right": "right/000000.png",
  "target": "target/000000.png",
  "left_mask": "left_mask/000000.png",
  "right_mask": "right_mask/000000.png",
  "seam_mask": "seam_mask/000000.png",
  "split": "train"
}
```

Real datasets can use the same manifest format. Every left image, right image, mask, and target must use the same canvas size. Raw unaligned pairs should be geometrically aligned before being added to the manifest.

Useful preparation options:

```bash
python main.py prepare --panoramas dataset/raw/custom --output dataset/prepared/panoramas --width 512 --height 256 --samples-per-image 32 --validation-fraction 0.1 --residual-shift 2 --seed 42
```

Both image dimensions must be divisible by 8 for Stable Diffusion training.

### UDIS-D pairs

UDIS-D contains `input1/input2` image pairs but does not include a true GT
panorama. ISS can still use it for diffusion seam/refinement fine-tuning by
copying the UDIS archives into this project, extracting them, aligning each
pair, and using the feather-blended coarse panorama as a pseudo target:

```bash
python main.py prepare-udis \
  --source /root/lys2/udis_l/Data \
  --raw-output dataset/raw/UDIS-D \
  --output dataset/prepared/udis \
  --width 1024 \
  --height 512
```

The command expects `unrar` to be installed. It writes:

```text
dataset/raw/UDIS-D/          copied training/testing .rar files and extracted UDIS-D
dataset/prepared/udis/       ISS left/right/mask/target folders plus manifest.jsonl
```

For a quick pipeline check, limit conversion:

```bash
python main.py prepare-udis --max-samples 8 --width 256 --height 128
```

The resulting target is marked as `coarse_pseudo_gt` in the manifest. This is
useful for adapting SD2 to stitched-image appearance and seam cleanup, but it
does not provide stronger supervision than the geometry baseline itself.

## 3. Train the tiny backend

```bash
python main.py train --config configs/tiny.yaml --data dataset/prepared/panoramas --output outputs/tiny --steps 1000
```

This backend uses a compact UNet and deterministic four-channel image codec. It is useful for testing datasets, losses, checkpoints, and CLI workflows on CPU.

Run the 12-channel version without masks:

```bash
python main.py train --config configs/tiny.yaml --data dataset/prepared/panoramas --output outputs/tiny-12ch --steps 1000 --no-masks
```

## 4. Verify the real Diffusers backend

Before downloading the full Stable Diffusion 2 weights, run the small safetensors-based integration model:

```bash
python main.py prepare --panoramas res/output.jpg --output dataset/prepared/sd-tiny --width 64 --height 64 --samples-per-image 2 --validation-fraction 0.5
python main.py train --config configs/sd-tiny.yaml --data dataset/prepared/sd-tiny
```

This configuration downloads a small testing model and performs real `AutoencoderKL`, CLIP, and `UNet2DConditionModel` forward/backward passes. It verifies dependency compatibility and the 14-channel input layer, but its weights are not intended for visual-quality evaluation.

## 5. Train Stable Diffusion 2

Prepare the dataset at the resolution configured in `configs/sd2.yaml`, then run:

```bash
python main.py train --config configs/sd2.yaml --data dataset/prepared/panoramas
```

The default configuration uses:

- `/root/ApDepth/pretrained_checkpoint/stable-diffusion-2` (the local SD2 768-v model)
- 512 x 1024 training images
- scheduler-native `v_prediction`
- batch size 1 with four-step gradient accumulation
- FP16 CUDA training
- gradient checkpointing
- channels-last memory format
- 20,000 optimizer steps

SD2 is pretrained at 768 x 768. ISS uses 512 x 1024 by default to retain the panorama aspect ratio with a similar pixel budget. Start with 256 x 512 if GPU memory is limited, or use 768 x 1536 when memory permits; update both dataset preparation and `configs/sd2.yaml`.

The VAE and empty OpenCLIP conditioning are reused from Stable Diffusion 2. The VAE is frozen and only the U-Net is optimized. The original four noisy-latent input weights are preserved, while the left/right/mask condition channels are zero-initialized. Training uses the scheduler bundled with the base checkpoint, and inference constructs a matching deterministic DDIM scheduler from it.

SDXL is not currently supported because it requires additional text and size conditioning.

To fine-tune on the converted UDIS-D data:

```bash
python main.py train --config configs/sd2-udis.yaml
```

This keeps the SD2 checkpoint in ApDepth and saves only ISS training outputs
under `outputs/sd2-udis`.

## 6. Resume training

Resume from a periodic checkpoint, a `final` directory, or a run directory containing `final`:

```bash
python main.py train --resume outputs/sd2/checkpoint-010000 --steps 20000
```

`--steps` is the target total number of optimizer steps. It is not the number of extra steps to add after the checkpoint.

Override the output directory or validation interval if needed:

```bash
python main.py train --resume outputs/sd2/checkpoint-010000 --output outputs/sd2-resumed --steps 20000 --validation-every 250
```

Checkpoints include:

- model weights
- optimizer state
- GradScaler state
- Python, NumPy, PyTorch, and CUDA random states
- current optimizer step
- best validation seam MAE
- complete YAML configuration

## 7. Evaluate a checkpoint

```bash
python main.py evaluate --checkpoint outputs/sd2/best --data dataset/prepared/panoramas --split val --batches 8 --output outputs/sd2/evaluation.json --device cuda
```

Evaluation writes:

- `evaluation.json` with MAE, seam MAE, overlap MAE, and PSNR
- `evaluation.png` containing left, right, coarse, prediction, and GT images

`seam_mae` is the primary checkpoint-selection metric. Check `overlap_mae` as well to ensure that seam repair is not damaging the complete overlap region.

## 8. Run inference

### Practical geometry-guided refinement

This is the default and most stable mode. It starts from a noised coarse panorama and preserves coarse pixels outside the seam-repair band.

```bash
python main.py infer --checkpoint outputs/sd2/best --left path/to/left.jpg --right path/to/right.jpg --output outputs/result.png --device cuda
```

The configured diffusion `strength` controls how much noise is added to the coarse initialization:

- `0.0`: return the coarse panorama
- `0.1-0.4`: conservative seam refinement
- `1.0`: ignore the coarse initialization and start from pure random noise

Override it at runtime:

```bash
python main.py infer --checkpoint outputs/sd2/best --left path/to/left.jpg --right path/to/right.jpg --output outputs/result.png --strength 0.35 --steps 30 --device cuda
```

### Pure-noise conditional generation

To use exactly two reference images plus a pure random-noise target latent:

```bash
python main.py infer --checkpoint outputs/sd2/best --left path/to/left.jpg --right path/to/right.jpg --output outputs/pure-diffusion.png --strength 1.0 --no-preserve-known --device cuda
```

With these two flags:

- The denoising target starts from pure random latent noise.
- The saved image is the model's complete decoded output and is not blended with the coarse panorama.

### Reuse an existing alignment

Avoid recomputing feature matching by passing a directory created by `align`:

```bash
python main.py infer --checkpoint outputs/sd2/best --left path/to/left.jpg --right path/to/right.jpg --aligned-dir outputs/alignment --output outputs/result.png --device cuda
```

The left and right paths are still required by the CLI, but the aligned canvases and masks are loaded from `--aligned-dir`.

## Configuration files

Three ready-to-use configurations are included:

| File | Purpose | Expected hardware |
|---|---|---|
| `configs/tiny.yaml` | Fast end-to-end pipeline and dataset debugging | CPU or GPU |
| `configs/sd-tiny.yaml` | Real Diffusers/VAE/CLIP/UNet integration smoke test | CPU or GPU |
| `configs/sd2.yaml` | Full Stable Diffusion 2 fine-tuning | CUDA GPU |
| `configs/sd2-udis.yaml` | Full SD2 fine-tuning on converted UDIS-D pseudo targets | CUDA GPU |

Important fields:

```yaml
model:
  backend: stable-diffusion
  pretrained_model: /root/ApDepth/pretrained_checkpoint/stable-diffusion-2
  use_masks: true
  gradient_checkpointing: true
  enable_xformers: false
  channels_last: true

train:
  max_steps: 20000
  grad_accumulation: 4
  validation_every: 500
  mixed_precision: fp16

diffusion:
  inference_steps: 30
  strength: 0.35
  prediction_type: v_prediction
```

Enable xFormers only after installing a version compatible with the local PyTorch and CUDA versions:

```yaml
model:
  enable_xformers: true
```

## Losses

```text
L = lambda_diff     * L_diffusion
  + lambda_rec      * L_reconstruction
  + lambda_seam     * L_seam
  + lambda_grad     * L_gradient
  + lambda_preserve * L_content_preservation
```

Loss weights are configured in the YAML `loss` section. Pixel-space losses require decoding the predicted clean latent and increase memory usage. If full-resolution training runs out of memory, first reduce resolution or set the pixel loss weights to zero for a diffusion-only warm-up stage.

## Continuous integration and package releases

The repository includes two GitHub Actions workflows:

- `.github/workflows/ci.yml` runs on every branch push and pull request. It tests Python 3.10, 3.11, and 3.12, then builds and validates the wheel and source distribution.
- `.github/workflows/release.yml` runs when a semantic version tag such as `v0.1.0` is pushed. It runs the tests again, creates a GitHub Release with the wheel and source distribution attached, and publishes a CPU container to GitHub Container Registry.

GitHub Packages does not provide a PyPI-compatible registry. The installable Python distributions are therefore attached to GitHub Releases, while the GHCR container is the package shown in the repository's **Packages** section.

### Create a release

The package version has a single source of truth in `iss/_version.py`. Update it before creating the tag:

```python
__version__ = "0.2.0"
```

Commit the version change and push the matching tag:

```bash
git add iss/_version.py
git commit -m "Release ISS 0.2.0"
git tag v0.2.0
git push origin HEAD
git push origin v0.2.0
```

The release workflow rejects a tag when its version does not exactly match `iss/_version.py`. No custom publishing secret is required: the workflow uses the repository-scoped `GITHUB_TOKEN` with explicit `contents: write` and `packages: write` permissions.

After the workflow succeeds:

- `iss-0.2.0-py3-none-any.whl` and the source archive are available from the GitHub Release.
- `ghcr.io/<owner>/<repository>:0.2.0` and `:latest` are available from GitHub Packages.

Install a downloaded wheel with:

```bash
pip install iss-0.2.0-py3-none-any.whl
```

Run the published CPU image with:

```bash
docker pull ghcr.io/<owner>/<repository>:0.2.0
docker run --rm ghcr.io/<owner>/<repository>:0.2.0 doctor
```

The container contains the base ISS dependencies and CPU PyTorch. Stable Diffusion extras and CUDA training environments should be installed separately on suitable GPU systems.

If an organization policy restricts workflow tokens, allow GitHub Actions to create releases and write packages in the repository or organization Actions settings. The first GHCR package may also need its visibility changed to public if anonymous pulls are desired.

Dependabot checks GitHub Actions, Python, and Docker dependencies weekly using `.github/dependabot.yml`.

## Tests

Install the development dependencies and run:

```bash
pip install -r requirements-dev.txt
pytest
```

When the Stable Diffusion dependencies are installed, the tests also create a real Diffusers conditional UNet, expand it to 14 channels, and run forward/backward propagation.

## Troubleshooting

### CUDA was requested but is unavailable

Run `python main.py doctor`. Install a CUDA-enabled PyTorch build or change the configuration to `device: cpu` for smoke testing.

### Out of GPU memory

- Lower `image_height` and `image_width`.
- Keep `batch_size: 1` and increase `grad_accumulation`.
- Keep `gradient_checkpointing: true`.
- Use FP16 on CUDA.
- Set reconstruction, seam, gradient, and preservation loss weights to zero during an initial diffusion-only stage.
- Enable xFormers only if a compatible build is available.

### Alignment cannot find enough matches

Use input images with more overlap or texture. Homography alignment is not sufficient for large parallax, repeated patterns, moving objects, or very small overlap regions.

### Model download problems

Set `model.pretrained_model` to a complete local Diffusers SD2 model directory. Prefer safetensors checkpoints. The directory must contain `vae`, `unet`, `tokenizer`, `text_encoder`, and `scheduler` subdirectories; ISS loads the scheduler configuration as part of the model rather than assuming an SD1.5 epsilon schedule.

### Windows Hugging Face cache warning

Hugging Face may warn that symlinks are unavailable. Downloads still work but may consume more disk space. Enabling Windows Developer Mode allows the cache to use symlinks.

## Current limitations

- Homography alignment cannot fully solve large parallax.
- Diffusion output may look plausible while changing real scene content.
- Text, building edges, and moving objects remain difficult.
- Synthetic panorama crops do not replace real multi-view training pairs.
- The project does not yet include a trainable pixel-space refinement network.
