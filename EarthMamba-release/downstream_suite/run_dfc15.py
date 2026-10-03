#!/usr/bin/env python3
"""DFC15 multilabel classification — unified ``--backbone`` for all six models."""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings

import bootstrap

bootstrap.setup()

import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

import backbone as bb
from common import (
    AmpHelper,
    add_output_args,
    add_perf_args,
    add_warmup_args,
    args_to_dict,
    build_warmup_cosine_scheduler,
    loader_kwargs,
    maybe_compile,
    param_groups_weight_decay,
    print_final_summary,
    resolve_out_dir,
    setup_perf,
)
from tasks.dfc15 import (
    DFC15Dataset,
    MultiBackboneClassifier,
    evaluate,
    load_multilabel_csv,
    resolve_dfc15_base,
    resolve_dfc15_split,
    train_one_epoch,
)

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    t_start = time.time()
    p = argparse.ArgumentParser("downstream_suite — DFC15")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--backbone", required=True, choices=bb.BACKBONE_CHOICES)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--ckpt_dir", default=None)
    add_output_args(p)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    add_perf_args(p)
    add_warmup_args(p, default_warmup=3)
    args = p.parse_args()
    setup_perf()
    args.amp_helper = AmpHelper(args.amp_dtype)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.data_dir).expanduser().resolve()
    base = resolve_dfc15_base(root)
    csv_path = base / "multilabel.csv"
    if csv_path.is_file() and (base / "images_tr").is_dir():
        label_map = load_multilabel_csv(csv_path)
        train_img, test_img = base / "images_tr", base / "images_test"
        train_lbl, test_lbl = None, None
        layout = "DFC15_multilabel (csv)"
    else:
        label_map = None
        train_img, train_lbl = resolve_dfc15_split(root, "train")
        test_img, test_lbl = resolve_dfc15_split(root, "test")
        layout = "legacy txt"

    print(f"[DFC15] backbone={args.backbone} layout={layout}")
    if args.dry_run:
        print(f"  train samples: {len(DFC15Dataset(train_img, label_map=label_map, label_file=train_lbl, img_size=args.img_size))}")
        return

    ckpt = bb.resolve_ckpt(args.backbone, args.ckpt, args.ckpt_dir)
    encoder = bb.build_encoder(args.backbone, args.img_size, ckpt)
    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lk = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)

    train_loader = DataLoader(
        DFC15Dataset(train_img, label_map=label_map, label_file=train_lbl, img_size=args.img_size, augment=True, strong_augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, **lk,
    )
    test_loader = DataLoader(
        DFC15Dataset(test_img, label_map=label_map, label_file=test_lbl, img_size=args.img_size, augment=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, **lk,
    )

    model = MultiBackboneClassifier(encoder, encoder.out_dims).to(device)
    model = maybe_compile(model, args.compile, device)

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    else:
        enc_groups = param_groups_weight_decay(model.encoder, args.weight_decay)
        for g in enc_groups:
            g["lr"] = args.encoder_lr
        optimizer = optim.AdamW(enc_groups + [{"params": list(model.head.parameters()), "lr": args.lr, "weight_decay": 0.0}])

    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)
    best_map, best_epoch, best_metrics, history = 0.0, 0, {}, []

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        scheduler.step()
        metrics = evaluate(model, test_loader, device, args.amp_helper)
        history.append({"epoch": epoch + 1, "train_loss": loss, **metrics})
        print(f"  test macro_mAP={metrics['macro_mAP']:.4f} micro_mAP={metrics['micro_mAP']:.4f}")
        if metrics["macro_mAP"] > best_map:
            best_map = metrics["macro_mAP"]
            best_epoch = epoch + 1
            best_metrics = metrics
            torch.save({"epoch": best_epoch, "metrics": metrics, "model": model.state_dict(), "args": args_to_dict(args)}, out_dir / "best.pth")

    elapsed = time.time() - t_start
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"backbone": args.backbone, "ckpt": ckpt, "args": args_to_dict(args), "best_epoch": best_epoch,
                   "best_metrics": best_metrics, "history": history, "time_seconds": elapsed}, f, indent=2, ensure_ascii=False)
    print_final_summary(dataset="DFC15", task="multilabel_cls", script="run_dfc15.py", ckpt=ckpt,
                        metrics=best_metrics, out_dir=out_dir, split="test", epoch=best_epoch,
                        extra_lines=[f"backbone={args.backbone}", f"elapsed={elapsed/60:.1f}min"])


if __name__ == "__main__":
    main()
