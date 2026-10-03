"""
baselines2 — INRIA 建筑物语义分割

支持 2 个 backbone: rvsa | satlas_aerial
流水线：Image → Backbone → UPerNet Decoder → CE Loss
主指标：val mIoU（建筑 / 背景二分类）

用法:
  cd /hy-tmp/baselines2
  python run_inria.py --backbone rvsa        --data_dir /hy-tmp/task/INRIA --patch_size 512 --epochs 50
  python run_inria.py --backbone satlas_aerial --data_dir /hy-tmp/task/INRIA --patch_size 512 --epochs 50
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

from tasks.inria import (
    INRIAPatchDataset,
    INRIASeg,
    resolve_inria_base,
    resolve_inria_train,
    train_epoch,
    validate,
)
from downstream_common import (
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
from backbone_registry import BACKBONE_CHOICES, build_encoder, resolve_ckpt

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    t_start = time.time()
    p = argparse.ArgumentParser("baselines2 — INRIA 建筑分割")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--backbone", default="rvsa", choices=BACKBONE_CHOICES)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--ckpt_dir", default=None)
    add_output_args(p)
    p.add_argument("--patch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    add_perf_args(p)
    add_warmup_args(p, default_warmup=5)
    args = p.parse_args()
    setup_perf()
    args.amp_helper = AmpHelper(args.amp_dtype)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.data_dir).expanduser().resolve()
    img_dir, gt_dir, test_img = resolve_inria_train(root)
    stems = sorted({p.stem for p in img_dir.glob("*.tif")})
    pairs = [(s, s) for s in stems if (gt_dir / f"{s}.tif").is_file()]
    rng = random.Random(args.seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_ratio))
    val_pairs, train_pairs = shuffled[:n_val], shuffled[n_val:]

    print(f"[baselines2/INRIA] backbone={args.backbone} patch_size={args.patch_size}")
    print(f"  base={resolve_inria_base(root)} train={len(train_pairs)} val={len(val_pairs)}")

    if args.dry_run:
        print("  dry_run 完成。")
        return

    ckpt = resolve_ckpt(args.backbone, args.ckpt, args.ckpt_dir)
    print(f"  ckpt={ckpt or 'random_init'}")
    encoder = build_encoder(args.backbone, args.patch_size, ckpt)

    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _lk = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)

    train_loader = DataLoader(
        INRIAPatchDataset(img_dir, gt_dir, args.patch_size, train_pairs, True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, **_lk,
    )
    val_loader = DataLoader(
        INRIAPatchDataset(img_dir, gt_dir, args.patch_size, val_pairs, False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, **_lk,
    )

    model = INRIASeg(encoder, encoder.out_dims, freeze=args.freeze_encoder).to(device)
    model = maybe_compile(model, args.compile, device)

    if args.freeze_encoder:
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        enc_groups = param_groups_weight_decay(model.encoder, args.weight_decay)
        for g in enc_groups:
            g["lr"] = args.encoder_lr
        dec_groups = param_groups_weight_decay(model.decoder, args.weight_decay)
        for g in dec_groups:
            g["lr"] = args.lr
        optimizer = optim.AdamW(enc_groups + dec_groups)

    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)
    best_miou, best_epoch, history = 0.0, 0, []

    for epoch in range(args.epochs):
        tr_loss, tr_miou = train_epoch(model, train_loader, optimizer, device, epoch, args)
        val_miou = validate(model, val_loader, device, args.amp_helper)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": tr_loss,
                         "train_miou": tr_miou, "val_miou": val_miou})
        print(f"  train_miou={tr_miou:.4f} val_miou={val_miou:.4f} best={best_miou:.4f}")
        if val_miou > best_miou:
            best_miou, best_epoch = val_miou, epoch + 1
            torch.save({"epoch": best_epoch, "val_miou": val_miou, "model": model.state_dict(),
                        "args": args_to_dict(args)}, out_dir / "best.pth")

    elapsed = time.time() - t_start
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"backbone": args.backbone, "ckpt": ckpt, "args": args_to_dict(args),
                   "best_val_miou": best_miou, "best_epoch": best_epoch,
                   "history": history, "time_seconds": elapsed}, f, indent=2, ensure_ascii=False)

    print_final_summary(dataset="INRIA", task="building_seg",
                        script=f"baselines2/run_inria.py [{args.backbone}]",
                        ckpt=ckpt, metrics={"val_mIoU": best_miou}, out_dir=out_dir,
                        split=f"val({len(val_pairs)} tiles)", epoch=best_epoch,
                        extra_lines=[f"backbone={args.backbone}",
                                     f"elapsed={elapsed/60:.1f}min"])


if __name__ == "__main__":
    main()
