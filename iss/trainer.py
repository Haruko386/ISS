from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from .config import ProjectConfig, save_config
from .data import StitchTripletDataset
from .memory import cuda_empty_cache
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


def _distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_fsdp_module(module: torch.nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        return False
    return isinstance(module, FSDP)


class ISSTrainer:
    def __init__(
        self,
        config: ProjectConfig,
        *,
        resume_from: str | Path | None = None,
    ) -> None:
        self.config = config
        self._validate_config()
        seed_everything(config.train.seed)
        self.distributed_mode = config.train.distributed.lower()
        self.world_size = _distributed_world_size()
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_distributed = self.world_size > 1 and self.distributed_mode != "none"
        if self.is_distributed:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = resolve_device(config.train.device)
        self.is_main_process = self.rank == 0
        cuda_empty_cache(self.device)
        self.output_dir = Path(config.train.output_dir)
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()
        self.metrics_path = self.output_dir / "metrics.jsonl"
        if self.is_main_process:
            save_config(config, self.output_dir / "config.yaml")
        self.model = ISSModel(config.model, config.diffusion).to(self.device)
        self._wrap_distributed_model()
        cuda_empty_cache(self.device)
        self.optimizer = AdamW(
            self.model.trainable_parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        precision = config.train.mixed_precision.lower()
        if precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: no, fp16, bf16")
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training is only supported on CUDA; use mixed_precision: no.")
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

    def _validate_config(self) -> None:
        train = self.config.train
        if train.distributed.lower() not in {"none", "fsdp"}:
            raise ValueError("distributed must be one of: none, fsdp")
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
        if self.config.model.backend.lower() in {"stable-diffusion", "sd"}:
            if self.config.data.image_height % 8 or self.config.data.image_width % 8:
                raise ValueError("Stable Diffusion image dimensions must be divisible by 8.")

    def _barrier(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.barrier()

    def _wrap_distributed_model(self) -> None:
        if not self.is_distributed:
            return
        if self.distributed_mode == "fsdp":
            if self.device.type != "cuda":
                raise RuntimeError("FSDP training requires CUDA devices.")
            try:
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,
                    ShardingStrategy,
                )
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            except ImportError as exc:
                raise RuntimeError("FSDP requires a PyTorch build with torch.distributed.fsdp.") from exc
            from functools import partial

            policy = partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
            self.model.unet = FSDP(
                self.model.unet,
                auto_wrap_policy=policy,
                device_id=self.device.index,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=True,
            )
            return
        raise ValueError(f"Unsupported distributed mode: {self.distributed_mode}")

    def _autocast(self):
        if not self.use_autocast:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        dataset = StitchTripletDataset(
            self.config.data.root,
            width=self.config.data.image_width,
            height=self.config.data.image_height,
            split=split,
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                seed=self.config.train.seed,
                drop_last=False,
            )
            if self.is_distributed
            else None
        )
        generator = torch.Generator().manual_seed(self.config.train.seed)
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
        if not self.is_main_process:
            return
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _rng_state(self) -> dict[str, Any]:
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
        checkpoint_dir = self.output_dir / (label or f"checkpoint-{step:06d}")
        if self.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()
        self.model.save_model(checkpoint_dir)
        optimizer_state = self._optimizer_state_dict_for_save()
        if self.is_main_process:
            torch.save(
                {
                    "format_version": 2,
                    "step": step,
                    "optimizer": optimizer_state,
                    "scaler": self.scaler.state_dict(),
                    "best_seam_mae": self.best_seam_mae,
                    "rng_state": self._rng_state(),
                },
                checkpoint_dir / "trainer_state.pt",
            )
            save_config(self.config, checkpoint_dir / "config.yaml")
        self._barrier()
        del optimizer_state
        cuda_empty_cache(self.device)
        return checkpoint_dir

    def _optimizer_state_dict_for_save(self) -> dict[str, Any]:
        if self.is_distributed and _is_fsdp_module(self.model.unet):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            return FSDP.full_optim_state_dict(self.model.unet, self.optimizer)
        return self.optimizer.state_dict()

    def _load_optimizer_state_dict(self, state: dict[str, Any]) -> None:
        optimizer_state = state["optimizer"]
        if self.is_distributed and _is_fsdp_module(self.model.unet):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            optimizer_state = FSDP.scatter_full_optim_state_dict(
                optimizer_state if self.is_main_process else None,
                self.model.unet,
            )
        self.optimizer.load_state_dict(optimizer_state)

    def _average_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if not self.is_distributed:
            return metrics
        keys = list(metrics)
        values = torch.tensor(
            [metrics[key] for key in keys],
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= self.world_size
        return {key: float(value.item()) for key, value in zip(keys, values)}

    def resume(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint)
        if (checkpoint / "final" / "trainer_state.pt").exists():
            checkpoint = checkpoint / "final"
        state_path = checkpoint / "trainer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Trainer state not found: {state_path}")
        self.model.load_model(checkpoint)
        state = _safe_torch_load(state_path)
        self._load_optimizer_state_dict(state)
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        self.start_step = int(state["step"])
        self.best_seam_mae = float(state.get("best_seam_mae", float("inf")))
        # A different output directory must own a self-contained best checkpoint.
        # Reset the score so the first validation writes one locally.
        if checkpoint.parent.resolve() != self.output_dir.resolve():
            self.best_seam_mae = float("inf")
        self._restore_rng_state(state.get("rng_state", {}))
        self.resumed_from = str(checkpoint)
        del state
        cuda_empty_cache(self.device)

    def validate(self, step: int, loader: DataLoader) -> dict[str, float]:
        cuda_empty_cache(self.device)
        try:
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
            metrics = self._average_metrics(metrics)
            record = {"type": "validation", "step": step, **metrics}
            self._append_metrics(record)
            details = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print(f"validation step={step} {details}", flush=True)
            if preview is not None:
                preview_dir = self.output_dir / "validation"
                if self.is_main_process:
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    grid = torch.cat([image for image in preview], dim=2)
                    tensor_to_image(grid).save(preview_dir / f"step-{step:06d}.png")
                    del grid
            if metrics["seam_mae"] < self.best_seam_mae:
                self.best_seam_mae = metrics["seam_mae"]
                self.save_checkpoint(step, label="best")
            return metrics
        finally:
            cuda_empty_cache(self.device)

    def train(self) -> Path:
        """
        Train the model to the configured maximum step and save the final checkpoint.
        
        Returns:
            Path: Path to the final checkpoint directory.
        
        Raises:
            ValueError: If the checkpoint has already reached or exceeded the configured maximum step.
        """
        config = self.config.train
        if self.start_step >= config.max_steps:
            raise ValueError(
                f"Checkpoint is already at step {self.start_step}, but max_steps={config.max_steps}."
            )
        loader = self._loader("train", shuffle=True)
        validation_loader = (
            self._loader("val", shuffle=False) if config.validation_every > 0 else None
        )
        iterator = iter(loader)
        self.model.train()
        started = time.perf_counter()
        last_metrics: dict[str, float] = {}
        last_validation: dict[str, float] = {}
        self._append_metrics(
            {
                "type": "run",
                "start_step": self.start_step,
                "max_steps": config.max_steps,
                "resumed_from": self.resumed_from,
            }
        )

        for step in range(self.start_step + 1, config.max_steps + 1):
            if isinstance(getattr(loader, "sampler", None), DistributedSampler):
                loader.sampler.set_epoch(step)
            self.optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, float] = {}
            for _ in range(config.grad_accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = _to_device(batch, self.device)
                with self._autocast():
                    losses = self.model.training_losses(batch, self.config.loss)
                    scaled_loss = losses["loss"] / config.grad_accumulation
                self.scaler.scale(scaled_loss).backward()
                for key, value in losses.items():
                    accumulated[key] = accumulated.get(key, 0.0) + float(
                        value.detach().float().item()
                    ) / config.grad_accumulation
                del batch, losses, scaled_loss
                cuda_empty_cache(self.device)

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            cuda_empty_cache(self.device)
            last_metrics = accumulated
            self._append_metrics({"type": "train", "step": step, **last_metrics})

            if step == self.start_step + 1 or step % config.log_every == 0 or step == config.max_steps:
                elapsed = time.perf_counter() - started
                details = " ".join(f"{key}={value:.4f}" for key, value in last_metrics.items())
                if self.is_main_process:
                    print(f"step={step}/{config.max_steps} {details} elapsed={elapsed:.1f}s", flush=True)
            if config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
                self.save_checkpoint(step)
                cuda_empty_cache(self.device)
            should_validate = validation_loader is not None and (
                step % config.validation_every == 0 or step == config.max_steps
            )
            if should_validate:
                last_validation = self.validate(step, validation_loader)

        cuda_empty_cache(self.device)
        checkpoint = self.save_checkpoint(config.max_steps, label="final")
        summary = {
            "steps": config.max_steps,
            "start_step": self.start_step,
            "resumed_from": self.resumed_from,
            "device": str(self.device),
            "backend": self.model.backend,
            "pretrained_model": self.config.model.pretrained_model,
            "prediction_type": self.model.prediction_type,
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
        self._barrier()
        return checkpoint
