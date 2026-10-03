from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CLASSES, DiorRDataset, collate_batch
from .geometry import mean_average_precision
from .model import decode_predictions, detection_loss

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(records, image_size, batch_size, workers, training, *, persistent=None):
    if persistent is None:
        # 验证集 loader 复用单实例；勿对每次 val 新建 persistent worker（会泄漏 fd）
        persistent = bool(training and workers > 0)
    kwargs = dict(
        dataset=DiorRDataset(records, image_size, training),
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=True,
        drop_last=training,
        collate_fn=collate_batch,
        persistent_workers=persistent,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 4 if training else 2
    return DataLoader(**kwargs)


def _amp_context(dtype: str):
    if dtype == "none":
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _log_training_setup(device: torch.device, args) -> None:
    amp = args.amp_dtype
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"[train] gpu={gpu} amp={amp}", flush=True)
    if amp == "bf16":
        print("[train] torch.autocast(bfloat16) ON — A100 原生 bf16，无需 GradScaler", flush=True)
    elif amp == "fp16":
        print("[train] torch.autocast(float16) ON — GradScaler enabled", flush=True)
    else:
        print("[train] mixed precision OFF (amp=none)", flush=True)
    if device.type == "cuda":
        print(
            f"[train] cudnn.benchmark={torch.backends.cudnn.benchmark} "
            f"tf32 matmul={torch.backends.cuda.matmul.allow_tf32}",
            flush=True,
        )


def _validation_epochs(total_epochs: int, eval_interval: int) -> list[int]:
    epochs = {1, total_epochs}
    if eval_interval > 0:
        epochs.update(range(eval_interval, total_epochs + 1, eval_interval))
    return sorted(epochs)


def _progress_enabled() -> bool:
    """tqdm 默认开启；`2>&1 | tee` 时 stderr 不是 TTY，但仍应显示进度（日志里逐行输出）。"""
    if tqdm is None:
        return False
    flag = os.environ.get("DIOR_R_PROGRESS", "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def _iter_with_progress(loader, *, desc: str, leave: bool):
    if not _progress_enabled():
        return loader
    # 非 TTY（tee / 重定向）时用固定宽度，避免 tqdm 完全禁用
    is_tty = sys.stderr.isatty() or sys.stdout.isatty()
    return tqdm(
        loader,
        desc=desc,
        unit="batch",
        leave=leave,
        dynamic_ncols=is_tty,
        ncols=100 if not is_tty else None,
        mininterval=0.5 if not is_tty else 0.1,
    )


def _set_learning_rates(optimizer, step, total_steps, warmup_steps, base_lrs):
    if step < warmup_steps:
        factor = (step + 1) / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        factor = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * factor


@torch.no_grad()
def validate(model, loader, device, amp_dtype, image_size, score_threshold, nms_threshold):
    model.eval()
    predictions, ground_truth = [], []
    start = time.time()
    for images, targets in _iter_with_progress(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with _amp_context(amp_dtype):
            outputs = model(images)
        predictions.extend(decode_predictions(
            outputs, image_size=image_size, score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        ))
        # targets are CPU tensors from DataLoader; .tolist() works directly.
        for target in targets:
            ground_truth.append({
                "polygons": target["polygons"].tolist(),
                "labels": target["labels"].tolist(),
                "difficult": target["difficult"].tolist(),
            })
    metrics = mean_average_precision(predictions, ground_truth, len(CLASSES), 0.5)
    metrics["elapsed_seconds"] = time.time() - start
    return metrics


def train(model, train_records, val_records, args):
    device = torch.device("cuda")
    model = model.to(device, memory_format=torch.channels_last)
    _log_training_setup(device, args)
    if getattr(args, "val_subset", 0) > 0:
        print(
            f"[val] mid-run subset cap={args.val_subset}; "
            f"final epoch uses full val ({len(val_records)} images)",
            flush=True,
        )
    val_epochs = _validation_epochs(args.epochs, args.eval_interval)
    subset_note = f"subset={args.val_subset}" if getattr(args, "val_subset", 0) > 0 else "full val"
    print(
        f"[val] eval_interval={args.eval_interval} ({subset_note}) → "
        f"mAP@0.5 at epochs {val_epochs}",
        flush=True,
    )
    train_loader = build_loader(
        train_records, args.image_size, args.batch_size, args.workers, True
    )
    val_workers = min(args.workers, 4)
    val_loader = build_loader(
        val_records, args.image_size, args.val_batch_size, val_workers, False
    )
    val_subset_loader = None
    if getattr(args, "val_subset", 0) > 0:
        val_subset_loader = build_loader(
            val_records[: args.val_subset],
            args.image_size,
            args.val_batch_size,
            val_workers,
            False,
        )
        print(f"[val] subset loader ready ({args.val_subset} images, workers={val_workers})", flush=True)

    from .earth_adapter import build_earth_mamba_param_groups, uses_earth_adapter

    earth_adapter_mode = getattr(args, "earth_adapter", "gated_pyramid")
    if uses_earth_adapter(args.backbone, earth_adapter_mode):
        param_groups, base_lrs = build_earth_mamba_param_groups(
            model,
            lr=args.lr,
            encoder_lr=args.encoder_lr,
            adapter_lr=getattr(args, "adapter_lr", args.lr),
            weight_decay=args.weight_decay,
        )
        optimizer = torch.optim.AdamW(param_groups)
    else:
        encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
        encoder_parameters, detector_parameters = [], []
        for parameter in model.parameters():
            (encoder_parameters if id(parameter) in encoder_ids else detector_parameters).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": detector_parameters, "lr": args.lr},
                {"params": encoder_parameters, "lr": args.encoder_lr},
            ],
            weight_decay=args.weight_decay,
        )
        base_lrs = [args.lr, args.encoder_lr]
    # GradScaler is only meaningful for fp16; bf16 accumulates natively.
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp_dtype == "fp16")
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    best_map, global_step, start_epoch = -1.0, 0, 1

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", (start_epoch - 1) * len(train_loader)))
        best_map = float(state.get("best_map50", -1.0))
        print(f"[resume] epoch={start_epoch} best_mAP50={best_map:.4f}")

    history = []
    wall_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "class_loss": 0.0, "quad_loss": 0.0}
        epoch_start = time.time()
        batch_iter = _iter_with_progress(
            train_loader,
            desc=f"train {epoch}/{args.epochs}",
            leave=(epoch == args.epochs),
        )
        use_pbar = _progress_enabled()
        for batch_index, (images, targets) in enumerate(batch_iter, 1):
            _set_learning_rates(optimizer, global_step, total_steps, warmup_steps, base_lrs)
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            # targets remain on CPU; build_dense_targets handles the GPU transfer
            # internally via a single non-blocking host→device copy per level.
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(args.amp_dtype):
                outputs = model(images)
                losses = detection_loss(outputs, targets, len(CLASSES))
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch} batch={batch_index}: {losses}"
                )
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            for key in sums:
                sums[key] += float(losses[key])
            if use_pbar and (
                batch_index % args.log_interval == 0 or batch_index == len(train_loader)
            ):
                elapsed = time.time() - epoch_start
                speed = batch_index * args.batch_size / max(elapsed, 1e-6)
                batch_iter.set_postfix(
                    loss=f"{sums['loss']/batch_index:.3f}",
                    cls=f"{sums['class_loss']/batch_index:.3f}",
                    quad=f"{sums['quad_loss']/batch_index:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    img_s=f"{speed:.0f}",
                    refresh=False,
                )
            elif not use_pbar and (
                batch_index % args.log_interval == 0 or batch_index == len(train_loader)
            ):
                elapsed = time.time() - epoch_start
                speed = batch_index * args.batch_size / max(elapsed, 1e-6)
                print(
                    f"Epoch {epoch}/{args.epochs} [{batch_index}/{len(train_loader)}] "
                    f"loss={sums['loss']/batch_index:.4f} "
                    f"cls={sums['class_loss']/batch_index:.4f} "
                    f"quad={sums['quad_loss']/batch_index:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} images/s={speed:.1f}",
                    flush=True,
                )
        if use_pbar and hasattr(batch_iter, "close"):
            batch_iter.close()

        record = {
            "epoch": epoch,
            "train_loss": sums["loss"] / len(train_loader),
            "epoch_seconds": time.time() - epoch_start,
        }
        should_validate = epoch in val_epochs
        if should_validate:
            using_subset = getattr(args, "val_subset", 0) > 0 and epoch != args.epochs
            if using_subset:
                print(f"[val] epoch={epoch} subset={args.val_subset}/{len(val_records)} images", flush=True)
                eval_loader = val_subset_loader
            else:
                print(f"[val] epoch={epoch} full val ({len(val_records)} images)", flush=True)
                eval_loader = val_loader
            metrics = validate(
                model, eval_loader, device, args.amp_dtype, args.image_size,
                args.score_threshold, args.nms_threshold,
            )
            record.update(metrics)
            run_best = max(best_map, metrics["mAP50"])
            tag = "subset" if using_subset else "full"
            print(
                f"[DIOR-R] epoch={epoch}/{args.epochs} OBB_mAP50={metrics['mAP50']:.4f} "
                f"best={run_best:.4f} ({tag}) val_time={metrics['elapsed_seconds']/60:.1f}min",
                flush=True,
            )
            state = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "global_step": global_step,
                "best_map50": max(best_map, metrics["mAP50"]), "args": vars(args),
            }
            torch.save(state, work_dir / "last.pt")
            if metrics["mAP50"] > best_map:
                best_map = metrics["mAP50"]
                torch.save(state, work_dir / "best.pt")
        else:
            print(
                f"[train] epoch={epoch}/{args.epochs} done "
                f"loss={record['train_loss']:.4f} "
                f"(val skipped; next mAP at epoch "
                f"{next(e for e in val_epochs if e > epoch)})",
                flush=True,
            )
        history.append(record)
        (work_dir / "results.json").write_text(
            json.dumps({"best_mAP50": best_map, "history": history}, indent=2),
            encoding="utf-8",
        )
    print(
        f"[done] best OBB mAP@0.5={best_map:.4f} "
        f"elapsed={(time.time()-wall_start)/3600:.2f}h output={work_dir}"
    )


def load_trained_model(checkpoint_path: str | Path, args):
    """Build detector and load ``best.pt`` / ``last.pt`` weights."""
    from .backbones import build_backbone
    from .model import FastOrientedDetector

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = state.get("args", {})
    backbone = saved_args.get("backbone", args.backbone)
    ckpt = saved_args.get("ckpt", args.ckpt)
    image_size = int(saved_args.get("image_size", args.image_size))
    ssm_version = saved_args.get("ssm_version", getattr(args, "ssm_version", "mamba3"))
    encoder, channels = build_backbone(
        backbone,
        ckpt,
        args.project_root,
        image_size,
        ssm_version,
        earth_adapter=saved_args.get("earth_adapter", getattr(args, "earth_adapter", "gated_pyramid")),
        debug_shapes=getattr(args, "debug_shapes", False),
    )
    model = FastOrientedDetector(encoder, channels, len(CLASSES))
    model.load_state_dict(state["model"])
    return model, state


def run_eval(model, val_records, args) -> dict:
    """Run OBB mAP@0.5 on validation set (optionally sweep decode thresholds)."""
    device = torch.device("cuda")
    model = model.to(device, memory_format=torch.channels_last).eval()
    eval_records = val_records
    if getattr(args, "val_subset", 0) > 0:
        eval_records = val_records[: args.val_subset]
        print(f"[eval] subset={len(eval_records)}/{len(val_records)} images", flush=True)
    else:
        print(f"[eval] full val ({len(val_records)} images)", flush=True)

    val_workers = min(args.workers, 4)
    loader = build_loader(
        eval_records, args.image_size, args.val_batch_size, val_workers, False
    )

    if getattr(args, "sweep_thresholds", False):
        score_grid = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15)
        nms_grid = (0.05, 0.1, 0.15, 0.2, 0.3)
        best_map, best_score, best_nms = -1.0, args.score_threshold, args.nms_threshold
        print(f"[eval] threshold sweep: {len(score_grid)}×{len(nms_grid)} combos", flush=True)
        for score_threshold in score_grid:
            for nms_threshold in nms_grid:
                metrics = validate(
                    model, loader, device, args.amp_dtype, args.image_size,
                    score_threshold, nms_threshold,
                )
                tag = "★" if metrics["mAP50"] > best_map else " "
                print(
                    f"  {tag} score={score_threshold:g} nms={nms_threshold:g} "
                    f"mAP50={metrics['mAP50']:.4f}",
                    flush=True,
                )
                if metrics["mAP50"] > best_map:
                    best_map, best_score, best_nms = metrics["mAP50"], score_threshold, nms_threshold
        print(
            f"[eval] best sweep mAP50={best_map:.4f} "
            f"score={best_score:g} nms={best_nms:g}",
            flush=True,
        )
        return {"mAP50": best_map, "score_threshold": best_score, "nms_threshold": best_nms}

    metrics = validate(
        model, loader, device, args.amp_dtype, args.image_size,
        args.score_threshold, args.nms_threshold,
    )
    print(
        f"[eval] mAP50={metrics['mAP50']:.4f} "
        f"score={args.score_threshold:g} nms={args.nms_threshold:g} "
        f"time={metrics['elapsed_seconds']/60:.1f}min",
        flush=True,
    )
    return metrics
