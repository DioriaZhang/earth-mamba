"""
baselines2 — DIOR 水平框目标检测

支持 2 个 backbone: rvsa | satlas_aerial
流水线：Backbone → CanonicalPyramid → torchvision RetinaNet（FP32 loss）
主指标：val mAP@0.5

用法:
  cd /hy-tmp/baselines2
  python run_dior.py --backbone rvsa --data_dir /hy-tmp/task/DIOR --batch_size 4 --grad_accum 2
  # 等效 batch=8；显存够可继续加大 --batch_size 或 --grad_accum
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

from tasks.dior import (
    DIORDataset,
    auto_split_ids,
    build_detector,
    collate_fn,
    evaluate,
    find_image,
    labeled_stems_in_trainval,
    load_split_ids,
    parse_voc_xml,
    resolve_ann_dir,
    resolve_trainval_img_dir,
    train_one_epoch,
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
    param_groups_weight_decay,
    print_final_summary,
    resolve_out_dir,
    setup_perf,
)
from backbone_registry import BACKBONE_CHOICES, build_encoder, resolve_ckpt

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    t_start = time.time()
    p = argparse.ArgumentParser("baselines2 — DIOR 水平框检测")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--backbone", default="rvsa", choices=BACKBONE_CHOICES)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--fpn_out", type=int, default=256)
    p.add_argument("--nms_thresh", type=float, default=0.5)
    p.add_argument("--detector", choices=["retinanet", "fasterrcnn"], default="retinanet",
                   help="默认 retinanet；fasterrcnn 为旧协议")
    add_output_args(p)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=4,
                   help="每卡 batch；显存不足可 2 + --grad_accum 加大等效 batch")
    p.add_argument("--lr", type=float, default=2e-4, help="检测头学习率")
    p.add_argument("--encoder_lr", type=float, default=1e-5, help="encoder 学习率")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--score_thresh", type=float, default=0.05)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--grad_accum", type=int, default=2,
                   help="梯度累积；等效 batch = batch_size × grad_accum")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    add_perf_args(p)
    add_warmup_args(p, default_warmup=2)
    add_eval_interval_args(p, default=5)
    # RetinaNet focal loss 在 fp16 下易 NaN；DIOR 默认 FP32
    p.set_defaults(amp_dtype="none")
    args = p.parse_args()
    setup_perf()
    args.amp_helper = AmpHelper(args.amp_dtype)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    ann_dir = resolve_ann_dir(data_dir)
    img_dir = resolve_trainval_img_dir(data_dir)
    all_labeled = labeled_stems_in_trainval(ann_dir, img_dir)
    split_dir = data_dir / "ImageSets" / "Main"
    train_ids = load_split_ids(split_dir, "train") if split_dir.is_dir() else None
    val_ids = load_split_ids(split_dir, "val") if split_dir.is_dir() else None
    split_source = "ImageSets/Main"
    if train_ids is None or val_ids is None:
        train_ids, val_ids = auto_split_ids(all_labeled, args.val_ratio, args.seed)
        split_source = f"auto {1 - args.val_ratio:.0%}/{args.val_ratio:.0%} from trainval"
    else:
        train_ids = [s for s in train_ids if s in all_labeled]
        val_ids = [s for s in val_ids if s in all_labeled]

    eff_bs = args.batch_size * args.grad_accum
    print(f"[baselines2/DIOR] backbone={args.backbone} img_size={args.img_size} "
          f"detector={args.detector} amp={args.amp_dtype}")
    print(f"  train={len(train_ids)} val={len(val_ids)} split={split_source}")
    print(f"  batch={args.batch_size} grad_accum={args.grad_accum} eff_batch={eff_bs}")
    print(f"  img={img_dir} ann={ann_dir}")

    if args.dry_run:
        for sid in train_ids[:3]:
            xml = ann_dir / f"{sid}.xml"
            stem, boxes, names, _, _ = parse_voc_xml(xml)
            img = find_image(img_dir, stem)
            print(f"  [{sid}] img={img.name} boxes={len(boxes)} classes={set(names)}")
        print("  dry_run 完成。")
        return

    ckpt = resolve_ckpt(args.backbone, args.ckpt, args.ckpt_dir)
    encoder = build_encoder(args.backbone, args.img_size, ckpt)
    n_enc = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"  ckpt={ckpt or 'random_init'} params={n_enc:.1f}M out_dims={encoder.out_dims}")

    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _lk = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)

    train_loader = DataLoader(
        DIORDataset(img_dir, ann_dir, train_ids, args.img_size, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True, drop_last=True, **_lk,
    )
    val_loader = DataLoader(
        DIORDataset(img_dir, ann_dir, val_ids, args.img_size, augment=False),
        batch_size=max(1, args.batch_size),
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True, **_lk,
    )

    raw_model, wrapper = build_detector(
        encoder,
        fpn_out=args.fpn_out,
        freeze_encoder=args.freeze_encoder,
        detector=args.detector,
        score_thresh=args.score_thresh,
        img_size=args.img_size,
        nms_thresh=args.nms_thresh,
    )
    raw_model = raw_model.to(device)
    n_total = sum(p.numel() for p in raw_model.parameters()) / 1e6
    print(f"  总参数量: {n_total:.1f}M backend=torchvision")

    if args.freeze_encoder:
        optimizer = optim.AdamW(
            [p for p in raw_model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        enc_body = raw_model.backbone.body
        enc_ids = {id(p) for p in enc_body.parameters()}
        enc_groups = param_groups_weight_decay(enc_body, args.weight_decay)
        for g in enc_groups:
            g["lr"] = args.encoder_lr
        head_params = [
            p for p in raw_model.parameters()
            if id(p) not in enc_ids and p.requires_grad
        ]
        head_groups = [
            {"params": [p for p in head_params if p.ndim > 1],
             "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": [p for p in head_params if p.ndim <= 1],
             "lr": args.lr, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(enc_groups + head_groups)

    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)
    best_map, best_epoch, history = 0.0, 0, []

    for epoch in range(args.epochs):
        t_ep = time.time()
        loss = train_one_epoch(
            wrapper, train_loader, optimizer, device, epoch, args,
            grad_accum=args.grad_accum,
        )
        do_eval = (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs
        if do_eval:
            metrics = evaluate(
                wrapper, val_loader, device, args.score_thresh, args.amp_helper,
            )
            map50 = metrics["mAP@0.5"]
        else:
            metrics = {}
            map50 = float("nan")

        scheduler.step()
        ep_sec = time.time() - t_ep
        row = {
            "epoch": epoch + 1,
            "train_loss": loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(ep_sec, 1),
        }
        if do_eval:
            row.update(metrics)
        history.append(row)

        if do_eval:
            print(f"  ep{epoch+1} loss={loss:.4f} mAP@0.5={map50:.4f} best={best_map:.4f} "
                  f"({ep_sec/60:.1f}min)")
            if map50 > best_map:
                best_map, best_epoch = map50, epoch + 1
                torch.save(
                    {
                        "epoch": best_epoch,
                        "metrics": metrics,
                        "model": raw_model.state_dict(),
                        "args": args_to_dict(args),
                    },
                    out_dir / "best.pth",
                )
        else:
            print(f"  ep{epoch+1} loss={loss:.4f} (skip val) ({ep_sec/60:.1f}min)")

    elapsed = time.time() - t_start
    summary = {
        "backbone": args.backbone,
        "ckpt": ckpt,
        "args": args_to_dict(args),
        "split_source": split_source,
        "best_mAP@0.5": best_map,
        "best_epoch": best_epoch,
        "backend": "torchvision",
        "detector": args.detector,
        "eff_batch": eff_bs,
        "history": history,
        "time_seconds": elapsed,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_final_summary(
        dataset="DIOR",
        task="horizontal_detection",
        script=f"baselines2/run_dior.py [{args.backbone}]",
        ckpt=ckpt,
        metrics={"mAP@0.5": best_map},
        out_dir=out_dir,
        split=f"val({len(val_ids)})",
        epoch=best_epoch,
        extra_lines=[
            f"backbone={args.backbone}",
            f"split_source={split_source}",
            f"backend=torchvision",
            f"detector={args.detector}",
            f"eff_batch={eff_bs}",
            f"elapsed={elapsed/60:.1f}min",
        ],
    )


if __name__ == "__main__":
    main()
