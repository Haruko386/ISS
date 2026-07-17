from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .alignment import AlignmentError, load_and_align, save_alignment
from .config import ProjectConfig, load_config, save_config
from .data import StitchTripletDataset, prepare_synthetic_dataset
from .metrics import evaluate_model
from .model import ISSModel
from .trainer import ISSTrainer, resolve_device, tensor_to_image


def _rgb_tensor(image_bgr: np.ndarray, width: int, height: int) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    array = np.asarray(rgb, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)[None]


def _mask_tensor(mask: np.ndarray, width: int, height: int) -> torch.Tensor:
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((resized >= 128).astype(np.float32))[None, None]


def _load_checkpoint_config(checkpoint: Path, explicit: str | None) -> tuple[ProjectConfig, Path]:
    if explicit:
        config = load_config(explicit)
    else:
        candidates = []
        if checkpoint.is_dir():
            candidates.extend([checkpoint / "config.yaml", checkpoint.parent / "config.yaml"])
        else:
            candidates.extend([checkpoint.parent / "config.yaml", checkpoint.parent.parent / "config.yaml"])
        config_path = next((path for path in candidates if path.exists()), None)
        if config_path is None:
            raise FileNotFoundError("No config.yaml found near the checkpoint; pass --config.")
        config = load_config(config_path)
    model_path = checkpoint
    if checkpoint.is_dir() and (checkpoint / "final").exists():
        model_path = checkpoint / "final"
    return config, model_path


def command_align(args: argparse.Namespace) -> int:
    result = load_and_align(
        args.left,
        args.right,
        ratio=args.ratio,
        ransac_threshold=args.ransac_threshold,
        min_matches=args.min_matches,
    )
    coarse = save_alignment(result, args.output)
    print(
        f"alignment saved: {coarse} | matches={result.matches} "
        f"inliers={result.inliers} canvas={result.shape[1]}x{result.shape[0]}"
    )
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    manifest = prepare_synthetic_dataset(
        args.panoramas,
        args.output,
        width=args.width,
        height=args.height,
        samples_per_image=args.samples_per_image,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        residual_shift=args.residual_shift,
    )
    count = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line)
    print(f"dataset ready: {manifest} ({count} samples)")
    return 0


def _apply_train_overrides(config: ProjectConfig, args: argparse.Namespace) -> None:
    if args.data:
        config.data.root = args.data
    if args.output:
        config.train.output_dir = args.output
    if args.steps is not None:
        config.train.max_steps = args.steps
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.device:
        config.train.device = args.device
    if args.no_masks:
        config.model.use_masks = False


def command_train(args: argparse.Namespace) -> int:
    """
    Train the stitching model using the selected configuration and command-line overrides.
    
    Parameters:
    	args (argparse.Namespace): Command-line arguments containing configuration, resume, override, and validation settings.
    
    Returns:
    	int: Exit status code, always 0 after training completes.
    """
    if args.config:
        config = load_config(args.config)
    elif args.resume:
        config, _ = _load_checkpoint_config(Path(args.resume), None)
    else:
        config = load_config("configs/tiny.yaml")
    _apply_train_overrides(config, args)
    if args.validation_every is not None:
        config.train.validation_every = args.validation_every
    trainer = ISSTrainer(config, resume_from=args.resume)
    checkpoint = trainer.train()
    if trainer.is_main_process:
        print(f"training complete: {checkpoint}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint)
    config, model_path = _load_checkpoint_config(checkpoint, args.config)
    if args.data:
        config.data.root = args.data
    device = resolve_device(args.device)
    model = ISSModel(config.model, config.diffusion)
    model.load_model(model_path)
    model.to(device).eval()
    dataset = StitchTripletDataset(
        config.data.root,
        width=config.data.image_width,
        height=config.data.image_height,
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics, preview = evaluate_model(
        model,
        loader,
        device=device,
        max_batches=args.batches,
        inference_steps=args.steps or config.diffusion.inference_steps,
        strength=config.diffusion.strength if args.strength is None else args.strength,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(model_path),
        "split": args.split,
        "batches": args.batches,
        **metrics,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if preview is not None:
        grid = torch.cat([image for image in preview], dim=2)
        tensor_to_image(grid).save(output.with_suffix(".png"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _alignment_from_directory(path: Path):
    required = ["left.png", "right.png", "left_mask.png", "right_mask.png", "coarse.png"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Aligned directory {path} is missing: {missing}")
    images = {name: cv2.imread(str(path / name), cv2.IMREAD_UNCHANGED) for name in required}
    seam_path = path / "seam_mask.png"
    if seam_path.exists():
        images["seam_mask.png"] = cv2.imread(str(seam_path), cv2.IMREAD_GRAYSCALE)
    if any(image is None for image in images.values()):
        raise OSError(f"Failed to read one or more aligned images from {path}")
    return images


def run_inference(
    config: ProjectConfig,
    checkpoint: Path,
    *,
    left_path: str,
    right_path: str,
    output_path: str | Path,
    aligned_dir: str | None = None,
    steps: int | None = None,
    strength: float | None = None,
    seed: int = 42,
    device_name: str = "auto",
    preserve_known: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_dir = output_path.parent / f"{output_path.stem}_alignment"
    if aligned_dir:
        images = _alignment_from_directory(Path(aligned_dir))
        left_bgr, right_bgr = images["left.png"], images["right.png"]
        left_mask, right_mask = images["left_mask.png"], images["right_mask.png"]
        coarse = images["coarse.png"]
        seam_mask = images.get("seam_mask.png")
    else:
        alignment = load_and_align(left_path, right_path)
        save_alignment(alignment, geometry_dir)
        left_bgr, right_bgr = alignment.left, alignment.right
        left_mask, right_mask, coarse = alignment.left_mask, alignment.right_mask, alignment.coarse
        seam_mask = alignment.seam_mask

    device = resolve_device(device_name)
    model = ISSModel(config.model, config.diffusion)
    model.load_model(checkpoint)
    model.to(device).eval()
    width, height = config.data.image_width, config.data.image_height
    left = _rgb_tensor(left_bgr, width, height).to(device)
    right = _rgb_tensor(right_bgr, width, height).to(device)
    mask_left = _mask_tensor(left_mask, width, height).to(device)
    mask_right = _mask_tensor(right_mask, width, height).to(device)
    coarse_tensor = _rgb_tensor(coarse, width, height).to(device)
    generated = model.sample(
        left,
        right,
        mask_left,
        mask_right,
        initial_image=coarse_tensor,
        strength=config.diffusion.strength if strength is None else strength,
        num_inference_steps=steps,
        seed=seed,
    )[0]
    generated_rgb = np.asarray(tensor_to_image(generated)).copy()
    if preserve_known:
        coarse_rgb = cv2.cvtColor(
            cv2.resize(coarse, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
        )
        if seam_mask is None:
            seam_mask = ((left_mask > 0) & (right_mask > 0)).astype(np.uint8) * 255
        alpha = cv2.resize(seam_mask, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0)[..., None]
        generated_rgb = np.uint8(
            np.clip(
                generated_rgb.astype(np.float32) * alpha
                + coarse_rgb.astype(np.float32) * (1.0 - alpha),
                0,
                255,
            )
        )
    Image.fromarray(generated_rgb).save(output_path)
    coarse_path = output_path.with_name(output_path.stem + "_coarse.png")
    cv2.imwrite(str(coarse_path), cv2.resize(coarse, (width, height), interpolation=cv2.INTER_AREA))
    metadata = {
        "checkpoint": str(checkpoint),
        "backend": config.model.backend,
        "in_channels": config.model.in_channels,
        "inference_steps": steps or config.diffusion.inference_steps,
        "strength": config.diffusion.strength if strength is None else strength,
        "seed": seed,
        "preserve_known": preserve_known,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def command_infer(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint)
    config, model_path = _load_checkpoint_config(checkpoint, args.config)
    result = run_inference(
        config,
        model_path,
        left_path=args.left,
        right_path=args.right,
        output_path=args.output,
        aligned_dir=args.aligned_dir,
        steps=args.steps,
        strength=args.strength,
        seed=args.seed,
        device_name=args.device,
        preserve_known=args.preserve_known,
    )
    print(f"stitched image saved: {result}")
    return 0


def command_demo(args: argparse.Namespace) -> int:
    root = Path(args.output)
    geometry_dir = root / "alignment"
    dataset_dir = root / "data"
    training_dir = root / "training"
    result = load_and_align(args.left, args.right)
    coarse_path = save_alignment(result, geometry_dir)
    prepare_synthetic_dataset(
        coarse_path,
        dataset_dir,
        width=args.width,
        height=args.height,
        samples_per_image=args.samples,
        seed=args.seed,
        validation_fraction=0.0,
        residual_shift=max(1, args.width // 128),
    )
    config = load_config(args.config)
    config.data.root = str(dataset_dir)
    config.data.image_width = args.width
    config.data.image_height = args.height
    config.train.output_dir = str(training_dir)
    config.train.max_steps = args.steps
    config.train.batch_size = min(config.train.batch_size, args.samples)
    config.train.device = args.device
    config.train.checkpoint_every = 0
    save_config(config, root / "demo_config.yaml")
    checkpoint = ISSTrainer(config).train()
    final = run_inference(
        config,
        checkpoint,
        left_path=args.left,
        right_path=args.right,
        aligned_dir=str(geometry_dir),
        output_path=root / "stitched.png",
        steps=args.inference_steps,
        strength=args.strength,
        seed=args.seed,
        device_name=args.device,
        preserve_known=True,
    )
    print(f"demo complete: {final}")
    return 0


def command_doctor(_: argparse.Namespace) -> int:
    """
    Report Python, library, and CUDA environment information as formatted JSON.
    
    Returns:
    	int: Exit status 0.
    """
    def installed(distribution: str) -> str | None:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return None

    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "diffusers": installed("diffusers"),
        "transformers": installed("transformers"),
        "accelerate": installed("accelerate"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "recommended_device": str(resolve_device("auto")),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser for the ISS image-stitching workflow.
    
    Returns:
    	argparse.ArgumentParser: Parser configured with all supported subcommands and their arguments.
    """
    parser = argparse.ArgumentParser(
        prog="iss",
        description="ISS geometry-guided conditional diffusion image stitching",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check the runtime environment")
    doctor.set_defaults(func=command_doctor)

    align = subparsers.add_parser("align", help="geometrically align a pair of images")
    align.add_argument("--left", required=True)
    align.add_argument("--right", required=True)
    align.add_argument("--output", default="outputs/alignment")
    align.add_argument("--ratio", type=float, default=0.75)
    align.add_argument("--ransac-threshold", type=float, default=4.0)
    align.add_argument("--min-matches", type=int, default=8)
    align.set_defaults(func=command_align)

    prepare = subparsers.add_parser("prepare", help="make triplets from complete panoramas")
    prepare.add_argument("--panoramas", required=True)
    prepare.add_argument("--output", default="data/demo")
    prepare.add_argument("--width", type=int, default=128)
    prepare.add_argument("--height", type=int, default=64)
    prepare.add_argument("--samples-per-image", type=int, default=16)
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--residual-shift", type=int, default=2)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=command_prepare)

    train = subparsers.add_parser("train", help="fine-tune a 12/14-channel diffusion UNet")
    train.add_argument("--config")
    train.add_argument("--data")
    train.add_argument("--output")
    train.add_argument("--steps", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument(
        "--device",
        help="training device (use cuda/auto when launching multi-GPU with torchrun)",
    )
    train.add_argument("--resume", help="checkpoint directory or run directory")
    train.add_argument("--validation-every", type=int)
    train.add_argument("--no-masks", action="store_true", help="use the 12-channel ablation")
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on train/val data")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--config")
    evaluate.add_argument("--data")
    evaluate.add_argument("--split", choices=["train", "val"], default="val")
    evaluate.add_argument("--output", default="outputs/evaluation.json")
    evaluate.add_argument("--batches", type=int, default=4)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--steps", type=int)
    evaluate.add_argument("--strength", type=float)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(func=command_evaluate)

    infer = subparsers.add_parser("infer", help="align and stitch a pair with a checkpoint")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--config")
    infer.add_argument("--left", required=True)
    infer.add_argument("--right", required=True)
    infer.add_argument("--aligned-dir")
    infer.add_argument("--output", default="outputs/stitched.png")
    infer.add_argument("--steps", type=int)
    infer.add_argument(
        "--strength",
        type=float,
        help="0 keeps the coarse panorama; 1 starts from pure random noise",
    )
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--device", default="auto")
    infer.add_argument(
        "--preserve-known",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep coarse pixels outside the seam repair band",
    )
    infer.set_defaults(func=command_infer)

    demo = subparsers.add_parser("demo", help="run alignment, data, training, and inference")
    demo.add_argument("--left", default="res/1.jpg")
    demo.add_argument("--right", default="res/2.jpg")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--config", default="configs/tiny.yaml")
    demo.add_argument("--width", type=int, default=128)
    demo.add_argument("--height", type=int, default=88)
    demo.add_argument("--samples", type=int, default=4)
    demo.add_argument("--steps", type=int, default=4)
    demo.add_argument("--inference-steps", type=int, default=8)
    demo.add_argument("--strength", type=float, default=0.05)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--device", default="auto")
    demo.set_defaults(func=command_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (AlignmentError, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
