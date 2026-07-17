from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from .config import ProjectConfig, save_config
from .data import StitchTripletDataset
from .metrics import evaluate_model
from .model import ISSModel


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach().float().cpu().clamp(-1.0, 1.0).add(1.0).mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array)


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch 2.2 compatibility
        return torch.load(path, map_location="cpu")


class ISSTrainer:
    def __init__(
        self,
        config: ProjectConfig,
        *,
        resume_from: str | Path | None = None,
    ) -> None:
        """
        Initialize the trainer, configure the execution environment, and prepare model and optimizer state.
        
        Parameters:
            config (ProjectConfig): Training, model, diffusion, and data configuration.
            resume_from (str | Path | None): Optional checkpoint path from which to resume training.
        
        Raises:
            ValueError: If fp16 mixed-precision training is requested on a non-CUDA device.
        """
        self.config = config
        self._validate_config()
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        self.is_main_process = self.rank == 0
        self._owns_process_group = False
        self.device = self._setup_distributed(config.train.device)
        precision = config.train.mixed_precision.lower()
        if precision == "fp16" and self.device.type != "cuda":
            self.close()
            raise ValueError("fp16 training is only supported on CUDA; use mixed_precision: no.")
        seed_everything(config.train.seed + self.rank)
        self.output_dir = Path(config.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        if self.is_main_process:
            save_config(config, self.output_dir / "config.yaml")
        if self.distributed:
            dist.barrier()
        try:
            self.model = ISSModel(config.model, config.diffusion).to(self.device)
        except Exception:
            self.close()
            raise
        if self.distributed:
            ddp_kwargs: dict[str, Any] = {"gradient_as_bucket_view": True}
            if self.device.type == "cuda":
                ddp_kwargs.update(
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )
            self.training_model: ISSModel | DistributedDataParallel = (
                DistributedDataParallel(self.model, **ddp_kwargs)
            )
        else:
            self.training_model = self.model
        self.optimizer = AdamW(
            self.model.trainable_parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        self.use_autocast = precision != "no" and self.device.type in {"cuda", "cpu"}
        scaler_enabled = precision == "fp16" and self.device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):  # torch 2.2 compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.start_step = 0
        self.best_seam_mae = float("inf")
        self.resumed_from: str | None = None
        if resume_from is not None:
            self.resume(resume_from)

    def _setup_distributed(self, requested_device: str) -> torch.device:
        """
        Configure the distributed process group and select the device for this trainer.
        
        Parameters:
        	requested_device (str): Requested device name, such as `"auto"`, `"cuda"`, or `"cpu"`.
        
        Returns:
        	torch.device: The device assigned to this process.
        """
        if not self.distributed:
            return resolve_device(requested_device)
        if self.rank < 0 or self.rank >= self.world_size:
            raise RuntimeError(
                f"Invalid distributed rank {self.rank} for world size {self.world_size}."
            )
        requested = requested_device.lower()
        wants_cuda = requested == "auto" or requested.startswith("cuda")
        if wants_cuda and torch.cuda.is_available():
            if self.local_rank < 0 or self.local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={self.local_rank} is invalid for "
                    f"{torch.cuda.device_count()} visible CUDA devices."
                )
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
            backend = "nccl"
        else:
            device = resolve_device(requested_device)
            if device.type == "mps":
                raise RuntimeError("Distributed training is not supported on MPS.")
            backend = "gloo"
        if not dist.is_available():
            raise RuntimeError("This PyTorch build does not support distributed training.")
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(hours=1),
            )
            self._owns_process_group = True
        return device

    def close(self) -> None:
        """Release resources owned by the trainer's distributed process group."""
        if self._owns_process_group and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_process_group = False

    def _validate_config(self) -> None:
        """
        Validate training, model, and image-dimension settings required by the trainer.
        
        Raises:
            ValueError: If a training value is invalid, mixed precision is unsupported,
                or Stable Diffusion image dimensions are not divisible by 8.
        """
        train = self.config.train
        if train.max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if train.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if train.grad_accumulation < 1:
            raise ValueError("grad_accumulation must be positive.")
        if train.log_every < 1:
            raise ValueError("log_every must be positive.")
        if train.validation_every > 0 and train.validation_batches < 1:
            raise ValueError("validation_batches must be positive when validation is enabled.")
        if train.mixed_precision.lower() not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: no, fp16, bf16")
        if self.config.model.backend.lower() in {"stable-diffusion", "sd"}:
            if self.config.data.image_height % 8 or self.config.data.image_width % 8:
                raise ValueError("Stable Diffusion image dimensions must be divisible by 8.")

    def _autocast(self):
        """Provide the configured automatic mixed-precision context for model operations.
        
        Returns:
        	context_manager: A no-op context when automatic mixed precision is disabled; otherwise, a context configured for the trainer's device and precision.
        """
        if not self.use_autocast:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _loader(
        self,
        split: str,
        *,
        shuffle: bool,
        distributed: bool = True,
    ) -> DataLoader:
        """
        Create a data loader for the specified dataset split.
        
        Parameters:
        	split (str): Dataset split to load.
        	shuffle (bool): Whether to shuffle the samples.
        	distributed (bool): Whether to use distributed sampling when distributed training is enabled.
        
        Returns:
        	DataLoader: Configured data loader for the requested split.
        """
        dataset = StitchTripletDataset(
            self.config.data.root,
            width=self.config.data.image_width,
            height=self.config.data.image_height,
            split=split,
        )
        sampler = None
        if self.distributed and distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                seed=self.config.train.seed,
                drop_last=False,
            )
        generator = torch.Generator().manual_seed(self.config.train.seed + self.rank)
        return DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.config.data.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            generator=generator,
        )

    def _append_metrics(self, payload: dict[str, Any]) -> None:
        """
        Append a metrics record to the metrics file on the main process.
        
        Parameters:
        	payload (dict[str, Any]): Metrics data to serialize as a JSON line.
        """
        if not self.is_main_process:
            return
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _average_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """
        Average metric values across distributed processes.
        
        Parameters:
        	metrics (dict[str, float]): Metric names and local values to average.
        
        Returns:
        	dict[str, float]: The input metrics unchanged for non-distributed execution; otherwise, metrics averaged across all processes.
        """
        if not self.distributed:
            return metrics
        keys = list(metrics)
        values = torch.tensor(
            [metrics[key] for key in keys],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= self.world_size
        return {key: float(value) for key, value in zip(keys, values.tolist())}

    def _rng_state(self) -> dict[str, Any]:
        """
        Capture the current random number generator states.
        
        Returns:
        	dict[str, Any]: Python, NumPy, and Torch CPU RNG states, plus CUDA RNG states when CUDA is available.
        """
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict[str, Any]) -> None:
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def save_checkpoint(self, step: int, label: str | None = None) -> Path:
        """
        Save the model, optimizer, training state, and random number generator states to a checkpoint.
        
        Parameters:
            step (int): Training step associated with the checkpoint.
            label (str | None): Optional directory label; defaults to a zero-padded step name.
        
        Returns:
            Path: Directory containing the saved checkpoint.
        """
        checkpoint_dir = self.output_dir / (label or f"checkpoint-{step:06d}")
        local_rng_state = self._rng_state()
        if self.distributed:
            rng_states: list[dict[str, Any] | None] | None = (
                [None] * self.world_size if self.is_main_process else None
            )
            dist.gather_object(local_rng_state, rng_states, dst=0)
        else:
            rng_states = [local_rng_state]
        if self.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_model(checkpoint_dir)
            torch.save(
                {
                    "format_version": 3,
                    "step": step,
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "best_seam_mae": self.best_seam_mae,
                    # Keep the old field for backwards consumers and save every
                    # rank separately for deterministic same-world-size resumes.
                    "rng_state": local_rng_state,
                    "rng_states": rng_states,
                    "world_size": self.world_size,
                },
                checkpoint_dir / "trainer_state.pt",
            )
            save_config(self.config, checkpoint_dir / "config.yaml")
        if self.distributed:
            dist.barrier()
        return checkpoint_dir

    def resume(self, checkpoint: str | Path) -> None:
        """
        Resume training from a checkpoint and restore the trainer state.
        
        Parameters:
            checkpoint (str | Path): Path to a checkpoint directory or an output directory containing a ``final`` checkpoint.
        
        Raises:
            FileNotFoundError: If the checkpoint does not contain ``trainer_state.pt``.
        """
        checkpoint = Path(checkpoint)
        if (checkpoint / "final" / "trainer_state.pt").exists():
            checkpoint = checkpoint / "final"
        state_path = checkpoint / "trainer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Trainer state not found: {state_path}")
        self.model.load_model(checkpoint)
        state = _safe_torch_load(state_path)
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        self.start_step = int(state["step"])
        self.best_seam_mae = float(state.get("best_seam_mae", float("inf")))
        # A different output directory must own a self-contained best checkpoint.
        # Reset the score so the first validation writes one locally.
        if checkpoint.parent.resolve() != self.output_dir.resolve():
            self.best_seam_mae = float("inf")
        rng_states = state.get("rng_states")
        if rng_states and self.rank < len(rng_states) and rng_states[self.rank] is not None:
            self._restore_rng_state(rng_states[self.rank])
        elif not self.distributed:
            self._restore_rng_state(state.get("rng_state", {}))
        else:
            # Resuming with a different number of workers cannot reproduce all
            # old RNG streams; start a deterministic stream for the new rank.
            seed_everything(self.config.train.seed + self.rank)
        self.resumed_from = str(checkpoint)

    def validate(
        self,
        step: int,
        loader: DataLoader,
        *,
        save_best: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate the model on a validation dataset and optionally save a checkpoint when seam MAE improves.
        
        Parameters:
            step (int): Training step associated with the validation.
            loader (DataLoader): Validation data loader.
            save_best (bool): Whether to save a checkpoint when the seam MAE reaches a new best value.
        
        Returns:
            dict[str, float]: Validation metrics.
        """
        with self._autocast():
            metrics, preview = evaluate_model(
                self.model,
                loader,
                device=self.device,
                max_batches=self.config.train.validation_batches,
                inference_steps=self.config.diffusion.inference_steps,
                strength=self.config.diffusion.strength,
                seed=self.config.train.seed,
            )
        record = {"type": "validation", "step": step, **metrics}
        self._append_metrics(record)
        details = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"validation step={step} {details}", flush=True)
        if preview is not None:
            preview_dir = self.output_dir / "validation"
            preview_dir.mkdir(parents=True, exist_ok=True)
            grid = torch.cat([image for image in preview], dim=2)
            tensor_to_image(grid).save(preview_dir / f"step-{step:06d}.png")
        if metrics["seam_mae"] < self.best_seam_mae:
            self.best_seam_mae = metrics["seam_mae"]
            if save_best:
                self.save_checkpoint(step, label="best")
        return metrics

    def train(self) -> Path:
        """
        Run the training process and release distributed resources when it finishes.
        
        Returns:
        	Path: Path to the final training checkpoint.
        """
        try:
            return self._train()
        finally:
            self.close()

    def _train(self) -> Path:
        """
        Run the training loop through the configured maximum step and save the final checkpoint.
        
        Returns:
        	Path: Path to the final training checkpoint.
        
        Raises:
        	ValueError: If the checkpoint's starting step is already at or beyond the configured maximum.
        """
        config = self.config.train
        if self.start_step >= config.max_steps:
            raise ValueError(
                f"Checkpoint is already at step {self.start_step}, but max_steps={config.max_steps}."
            )
        loader = self._loader("train", shuffle=True)
        validation_loader = (
            self._loader("val", shuffle=False, distributed=False)
            if config.validation_every > 0 and self.is_main_process
            else None
        )
        train_sampler = (
            loader.sampler if isinstance(loader.sampler, DistributedSampler) else None
        )
        epoch = 0
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iterator = iter(loader)
        self.model.train()
        started = time.perf_counter()
        last_metrics: dict[str, float] = {}
        last_validation: dict[str, float] = {}
        effective_batch_size = (
            config.batch_size * config.grad_accumulation * self.world_size
        )
        self._append_metrics(
            {
                "type": "run",
                "start_step": self.start_step,
                "max_steps": config.max_steps,
                "resumed_from": self.resumed_from,
                "world_size": self.world_size,
                "per_device_batch_size": config.batch_size,
                "effective_batch_size": effective_batch_size,
            }
        )
        if self.is_main_process and self.distributed:
            print(
                f"distributed training: world_size={self.world_size} "
                f"per_device_batch_size={config.batch_size} "
                f"effective_batch_size={effective_batch_size}",
                flush=True,
            )

        for step in range(self.start_step + 1, config.max_steps + 1):
            self.optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, float] = {}
            for micro_step in range(config.grad_accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = _to_device(batch, self.device)
                should_sync = micro_step == config.grad_accumulation - 1
                sync_context = (
                    self.training_model.no_sync()
                    if self.distributed and not should_sync
                    else nullcontext()
                )
                with sync_context:
                    with self._autocast():
                        losses = self.training_model(batch, self.config.loss)
                        scaled_loss = losses["loss"] / config.grad_accumulation
                    self.scaler.scale(scaled_loss).backward()
                for key, value in losses.items():
                    accumulated[key] = accumulated.get(key, 0.0) + float(
                        value.detach().float().item()
                    ) / config.grad_accumulation

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), config.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            last_metrics = self._average_metrics(accumulated)
            self._append_metrics({"type": "train", "step": step, **last_metrics})

            if self.is_main_process and (
                step == self.start_step + 1
                or step % config.log_every == 0
                or step == config.max_steps
            ):
                elapsed = time.perf_counter() - started
                details = " ".join(f"{key}={value:.4f}" for key, value in last_metrics.items())
                print(f"step={step}/{config.max_steps} {details} elapsed={elapsed:.1f}s", flush=True)
            if config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
                self.save_checkpoint(step)
            should_validate = config.validation_every > 0 and (
                step % config.validation_every == 0 or step == config.max_steps
            )
            if should_validate:
                improved = False
                if self.is_main_process:
                    previous_best = self.best_seam_mae
                    if validation_loader is None:  # pragma: no cover - defensive
                        raise RuntimeError("Main process has no validation loader.")
                    last_validation = self.validate(
                        step,
                        validation_loader,
                        save_best=not self.distributed,
                    )
                    improved = self.best_seam_mae < previous_best
                if self.distributed:
                    validation_state = [
                        last_validation,
                        self.best_seam_mae,
                        improved,
                    ]
                    dist.broadcast_object_list(validation_state, src=0)
                    last_validation = validation_state[0]
                    self.best_seam_mae = float(validation_state[1])
                    improved = bool(validation_state[2])
                    if improved:
                        self.save_checkpoint(step, label="best")

        checkpoint = self.save_checkpoint(config.max_steps, label="final")
        summary = {
            "steps": config.max_steps,
            "start_step": self.start_step,
            "resumed_from": self.resumed_from,
            "device": str(self.device),
            "world_size": self.world_size,
            "per_device_batch_size": config.batch_size,
            "effective_batch_size": effective_batch_size,
            "backend": self.model.backend,
            "in_channels": self.config.model.in_channels,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "final_metrics": last_metrics,
            "validation_metrics": last_validation,
            "best_seam_mae": self.best_seam_mae,
            "checkpoint": str(checkpoint),
        }
        if self.is_main_process:
            with (self.output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
        if self.distributed:
            dist.barrier()
        return checkpoint
