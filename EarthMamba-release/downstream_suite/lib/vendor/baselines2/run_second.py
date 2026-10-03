"""
baselines2 — SECOND 语义变化检测

支持 2 个 backbone: rvsa | satlas_aerial
流水线：Siamese Backbone → 三元差异融合 → UPerNet → CE Loss
主指标：test mIoU（7 类，含未变化）

用法:
  cd /hy-tmp/baselines2
  python run_second.py --backbone rvsa         --data_dir /hy-tmp/task/second_dataset --epochs 50
  python run_second.py --backbone satlas_aerial --data_dir /hy-tmp/task/second_dataset --epochs 50
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import argparse
import json
import random
import time
import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tasks.second import (
    SECONDDataset,
    SECONDSeg,
    inspect_stats,
    resolve_second_split,
    train_epoch,
    validate,
)
from downstream_common import (
    AmpHelper,
    add_eval_interval_args,
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
from backbone_registry import BACKBONE_CHOICES, build_encoder, resolve_ckpt

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    t_start = time.time()
    p = argparse.ArgumentParser("baselines2 — SECOND 语义变化检测")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--backbone", default="rvsa", choices=BACKBONE_CHOICES)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--ckpt_dir", default=None)
    add_output_args(p)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--eval_split", default="test", choices=["val", "test"])
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--fpn_dim", type=int, default=256)
    add_perf_args(p)
    add_warmup_args(p, default_warmup=5)
    add_eval_interval_args(p, default=1)
    args = p.parse_args()
    setup_perf()
    args.amp_helper = AmpHelper(args.amp_dtype)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.data_dir).expanduser().resolve()
    train_dir = resolve_second_split(root, "train")
    try:
        eval_dir = resolve_second_split(root, args.eval_split)
    except FileNotFoundError:
        eval_dir = resolve_second_split(root, "test")
        args.eval_split = "test"

    print(f"[baselines2/SECOND] backbone={args.backbone} img_size={args.img_size}")
    print(f"  train={train_dir} eval={eval_dir}")
    inspect_stats(train_dir, args.img_size)
    inspect_stats(eval_dir, args.img_size)

    if args.dry_run:
        print("  dry_run 完成。")
        return

    ckpt = resolve_ckpt(args.backbone, args.ckpt, args.ckpt_dir)
    print(f"  ckpt={ckpt or 'random_init'}")
    encoder = build_encoder(args.backbone, args.img_size, ckpt)
    print(f"  out_dims={encoder.out_dims}")

    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _lk = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)

    train_loader = DataLoader(
        SECONDDataset(train_dir, args.img_size, True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, **_lk,
    )
    eval_loader = DataLoader(
        SECONDDataset(eval_dir, args.img_size, False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, **_lk,
    )

    model = SECONDSeg(encoder, freeze=args.freeze_encoder, fpn_dim=args.fpn_dim).to(device)
    model = maybe_compile(model, args.compile, device)

    if args.freeze_encoder:
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        enc_groups = param_groups_weight_decay(model.enc, args.weight_decay)
        for g in enc_groups:
            g["lr"] = args.encoder_lr
        fuse_dec = list(model.fuse_proj.parameters()) + list(model.dec.parameters())
        fuse_groups = [
            {"params": [p for p in fuse_dec if p.ndim > 1],
             "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": [p for p in fuse_dec if p.ndim <= 1],
             "lr": args.lr, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(enc_groups + fuse_groups)

    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)
    best_miou, best_epoch, history = 0.0, 0, []

    for epoch in range(args.epochs):
        tr_loss = train_epoch(model, train_loader, optimizer, device, epoch, args)
        do_eval = (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs
        ev_miou = float("nan")
        if do_eval:
            ev_miou = validate(model, eval_loader, device, args.amp_helper,
                               desc=f"Valid ep{epoch+1}/{args.epochs}")
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": tr_loss,
                         f"{args.eval_split}_miou": ev_miou if do_eval else None})
        if do_eval:
            print(f"  ep{epoch+1:03d} loss={tr_loss:.4f} {args.eval_split}_miou={ev_miou:.4f} best={best_miou:.4f}")
            if ev_miou > best_miou:
                best_miou, best_epoch = ev_miou, epoch + 1
                torch.save({"epoch": best_epoch, "miou": best_miou, "model": model.state_dict(),
                            "args": args_to_dict(args)}, out_dir / "best.pth")
        else:
            print(f"  ep{epoch+1:03d} loss={tr_loss:.4f} (skip val)")

    elapsed = time.time() - t_start
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"backbone": args.backbone, "ckpt": ckpt, "args": args_to_dict(args),
                   f"best_{args.eval_split}_miou": best_miou, "best_epoch": best_epoch,
                   "history": history, "time_seconds": elapsed}, f, indent=2, ensure_ascii=False)

    print_final_summary(dataset="SECOND", task="semantic_change_detection",
                        script=f"baselines2/run_second.py [{args.backbone}]",
                        ckpt=ckpt, metrics={f"{args.eval_split}_mIoU": best_miou},
                        out_dir=out_dir, split=args.eval_split, epoch=best_epoch,
                        extra_lines=[f"backbone={args.backbone}",
                                     f"elapsed={elapsed/60:.1f}min"])


if __name__ == "__main__":
    main()
