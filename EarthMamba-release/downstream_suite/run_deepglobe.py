#!/usr/bin/env python3
"""DeepGlobe land-cover segmentation — unified ``--backbone`` (six models)."""

from __future__ import annotations

import argparse

import bootstrap

bootstrap.setup()

from lib._paths import add_project_paths, project_root
from lib.deepglobe_core import (
    DEEPGLOBE_NUM_CLASSES,
    UPerNetSeg,
    add_cache_args,
    add_perf_args,
    add_warmup_args,
    run_deepglobe_training,
)
from lib.earthmamba_adapter import ensure_registered
import backbone as bb


def _build_model(args, root):
    ensure_registered()
    ckpt = bb.resolve_ckpt(args.backbone, args.ckpt)
    args.ckpt = ckpt
    encoder = bb.build_segmentation_encoder(
        args.backbone,
        args.img_size,
        ckpt,
        ssm_version=args.ssm_version,
        dense_adapter=args.dense_adapter,
        model_size=args.model_size,
        debug_shapes=args.debug_shapes,
    )
    return UPerNetSeg(
        encoder,
        num_classes=DEEPGLOBE_NUM_CLASSES,
        fpn_dim=args.fpn_dim,
        freeze_encoder=args.freeze_encoder,
    )


def main() -> None:
    root = add_project_paths(project_root())
    ensure_registered()
    choices = bb.BACKBONE_CHOICES

    p = argparse.ArgumentParser("downstream_suite — DeepGlobe")
    p.add_argument("--backbone", required=True, choices=choices)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--fpn_dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--adapter_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--samples_per_tile", type=int, default=8)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--color_jitter", type=float, default=0.05)
    p.add_argument("--dice_weight", type=float, default=0.5)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--inspect_data", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ssm_version", default="mamba3")
    p.add_argument("--model_size", default="auto", choices=("auto", "small", "base"))
    p.add_argument("--early_stop_patience", type=int, default=12)
    p.add_argument("--dense_adapter", default="gated_pyramid",
                   choices=("gated_pyramid", "detail_pyramid", "semantic_pyramid", "none"))
    p.add_argument("--debug_shapes", action="store_true")
    p.add_argument("--nan_guard", action="store_true", default=True)
    p.add_argument("--no_nan_guard", action="store_false", dest="nan_guard")
    p.add_argument("--max_loss", type=float, default=10.0)
    p.add_argument("--max_nan_batches", type=int, default=8)
    add_perf_args(p)
    add_cache_args(p)
    p.set_defaults(amp_dtype="bf16", prefetch_factor=4)
    add_warmup_args(p, default_warmup=5)
    args = p.parse_args()

    if args.patch_size is None:
        args.patch_size = args.img_size
    if args.backbone in ("earth-mamba", "earth-mamba-b"):
        args.nan_guard = True
        if args.dice_weight <= 0:
            args.dice_weight = 0.5

    script = f"run_deepglobe.py --backbone {args.backbone}"
    if args.dry_run:
        run_deepglobe_training(args, None, script_name=script, backbone_name=args.backbone)
        return
    model = _build_model(args, root)
    run_deepglobe_training(args, model, script_name=script, backbone_name=args.backbone)


if __name__ == "__main__":
    main()
