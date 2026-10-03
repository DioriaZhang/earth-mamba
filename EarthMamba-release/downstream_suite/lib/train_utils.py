"""Self-contained training helpers + UPerNet head (no v3_loveda dependency)."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_OUTPUT_DIR = os.environ.get("DEEPGLOBE_OUTPUT", "/hy-tmp/downstream_results")


def setup_perf() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except RuntimeError as exc:
        print(f"  [setup_perf] skip CUDA tuning: {exc}", flush=True)


def add_perf_args(parser) -> None:
    g = parser.add_argument_group("performance")
    g.add_argument("--amp_dtype", default="auto", choices=["auto", "none", "bf16", "fp16"])
    g.add_argument("--prefetch_factor", type=int, default=4)
    g.add_argument("--persistent_workers", action="store_true", default=True)
    g.add_argument("--no_persistent_workers", action="store_false", dest="persistent_workers")
    g.add_argument("--compile", action="store_true")


def add_warmup_args(parser, default_warmup: int = 5) -> None:
    parser.add_argument("--warmup_epochs", type=int, default=default_warmup)


def loader_kwargs(num_workers: int, prefetch_factor: int = 4, persistent_workers: bool = True) -> dict:
    if num_workers <= 0:
        return {}
    kw = {"prefetch_factor": prefetch_factor}
    if persistent_workers:
        kw["persistent_workers"] = True
    return kw


def args_to_dict(args) -> dict:
    out = vars(args).copy()
    out.pop("amp_helper", None)
    return out


class AmpHelper:
    def __init__(self, amp_dtype: str = "auto"):
        if amp_dtype == "auto":
            if not torch.cuda.is_available():
                amp_dtype = "none"
            else:
                major, _ = torch.cuda.get_device_capability()
                amp_dtype = "bf16" if major >= 8 else "fp16"
        self.amp_dtype = amp_dtype
        self.enabled = torch.cuda.is_available() and amp_dtype != "none"
        self.use_scaler = self.enabled and amp_dtype == "fp16"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield

    def backward_step(self, loss: torch.Tensor, optimizer, model: nn.Module, clip_grad: float = 0.0) -> None:
        if self.use_scaler:
            self.scaler.scale(loss).backward()
            if clip_grad > 0:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            self.scaler.step(optimizer)
            self.scaler.update()
            return
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()


def maybe_compile(model: nn.Module, do_compile: bool, device: torch.device) -> nn.Module:
    if not do_compile:
        return model
    print("  [skip] torch.compile is disabled for these remote-sensing backbones.")
    return model


def param_groups_weight_decay(module: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if total_epochs <= warmup_epochs:
            return 1.0
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_out_dir(output_dir: Optional[str], *, data_dir: str, dataset_name: str = "DeepGlobe_v3") -> Path:
    if output_dir:
        out = Path(output_dir)
    else:
        out = Path(DEFAULT_OUTPUT_DIR) / datetime.now().strftime("%Y%m%d_%H%M%S") / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_resume_checkpoint(resume: str | Path) -> Path:
    path = Path(resume).expanduser().resolve()
    if path.is_dir():
        for name in ("last.pth", "best.pth"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no last.pth or best.pth under {path}")
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def _align_scheduler_to_epoch(scheduler, completed_epochs: int) -> None:
    for _ in range(completed_epochs):
        scheduler.step()


def checkpoint_payload(
    model: nn.Module,
    optimizer,
    scheduler,
    *,
    epoch: int,
    best_miou: float,
    best_epoch: int,
    stale_epochs: int,
    history: list,
    args,
    backbone_name: str,
    val_miou: float,
    val_miou6: float,
    val_class_iou,
    protocol: str,
) -> dict:
    return {
        "epoch": epoch,
        "val_miou": val_miou,
        "val_miou_6": val_miou6,
        "val_class_iou": val_class_iou,
        "best_val_miou": best_miou,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "epochs_total": args.epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
        "args": args_to_dict(args),
        "backbone": backbone_name,
        "protocol": protocol,
    }


def load_training_resume(
    resume: str | Path,
    model: nn.Module,
    optimizer,
    scheduler,
    *,
    target_epochs: int,
) -> tuple[int, float, int, int, list, Path]:
    ckpt_path = resolve_resume_checkpoint(resume)
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(state["model"])
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    else:
        print("  [resume] checkpoint has no optimizer state; optimizer re-initialized", flush=True)

    completed = int(state.get("epoch", 0))
    saved_epochs = int(state.get("epochs_total", target_epochs))
    if "scheduler" in state and saved_epochs == target_epochs:
        scheduler.load_state_dict(state["scheduler"])
    else:
        if saved_epochs != target_epochs:
            print(
                f"  [resume] epochs changed ({saved_epochs} -> {target_epochs}); "
                f"scheduler realigned to epoch {completed}",
                flush=True,
            )
        _align_scheduler_to_epoch(scheduler, completed)

    best_miou = float(state.get("best_val_miou", state.get("val_miou_6", state.get("val_miou", 0.0))))
    best_epoch = int(state.get("best_epoch", completed))
    stale_epochs = int(state.get("stale_epochs", 0))
    history = list(state.get("history", []))
    out_dir = ckpt_path.parent
    print(
        f"  [resume] from {ckpt_path.name} dir={out_dir} "
        f"next_epoch={completed + 1} best_val_mIoU-6={best_miou:.4f} (epoch {best_epoch})",
        flush=True,
    )
    return completed, best_miou, best_epoch, stale_epochs, history, out_dir


def print_final_summary(
    *,
    dataset: str,
    task: str,
    script: str,
    ckpt: Optional[str],
    metrics: dict,
    out_dir: Path,
    split: str = "",
    epoch: Optional[int] = None,
    extra_lines: Optional[list[str]] = None,
) -> None:
    print("\n" + "=" * 72)
    print(f"  [{dataset}] {task}  |  {script}")
    print(f"  ckpt: {ckpt or '(none) random init'}")
    if split:
        print(f"  split: {split}")
    if epoch is not None:
        print(f"  best_epoch: {epoch}")
    print("-" * 72)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if extra_lines:
        for line in extra_lines:
            print(f"  {line}")
    print("-" * 72)
    print(f"  输出目录: {out_dir}")
    print(f"  结果文件: {out_dir / 'results.json'}")
    print("=" * 72 + "\n")


def _gn(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if groups <= channels and channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class PPM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                _gn(out_dim),
                nn.ReLU(True),
            )
            for b in bins
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim + out_dim * len(bins), out_dim, 3, padding=1, bias=False),
            _gn(out_dim),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        parts = [x] + [
            F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        return self.bottleneck(torch.cat(parts, dim=1))


class UPerNetDecoder(nn.Module):
    def __init__(
        self,
        in_dims: List[int],
        fpn_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ppm = PPM(in_dims[-1], fpn_dim)
        self.lateral = nn.ModuleList([nn.Conv2d(d, fpn_dim, 1) for d in in_dims[:-1]])
        self.fpn_conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False), _gn(fpn_dim), nn.ReLU(True))
            for _ in in_dims[:-1]
        ])
        self.fuse = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(fpn_dim * len(in_dims), fpn_dim, 3, padding=1, bias=False),
            _gn(fpn_dim),
            nn.ReLU(True),
        )
        self.head = nn.Conv2d(fpn_dim, num_classes, 1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        out_hw = feats[0].shape[-2:]
        x = self.ppm(feats[-1])
        outs = [x]
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.fpn_conv[i](x + self.lateral[i](feats[i]))
            outs.insert(0, x)
        outs = [F.interpolate(o, size=out_hw, mode="bilinear", align_corners=False) for o in outs]
        return self.head(self.fuse(torch.cat(outs, dim=1)))


class UPerNetSeg(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_classes: int = 7,
        fpn_dim: int = 256,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        in_dims = list(getattr(encoder, "out_dims"))
        self.decoder = UPerNetDecoder(in_dims, fpn_dim=fpn_dim, num_classes=num_classes)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
