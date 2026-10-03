"""DIOR 检测 — EarthMamba 消融。"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.common import (
    AmpHelper,
    add_output_args,
    add_perf_args,
    add_warmup_args,
    args_to_dict,
    build_warmup_cosine_scheduler,
    loader_kwargs,
    maybe_compile,
    print_final_summary,
    resolve_out_dir,
    setup_perf,
)
from lib.dior_core import (
    DIORDataset,
    auto_split_ids,
    build_detector,
    collate_fn,
    evaluate,
    labeled_stems_in_trainval,
    resolve_ann_dir,
    resolve_trainval_img_dir,
    train_one_epoch,
)
from lib.earthmamba import build_dense_encoder, build_earthmamba_param_groups
from lib.linux_env import clean_str
from variant_flags import resolve_flags


def main(argv=None):
    start = time.time()
    p = argparse.ArgumentParser("ablation DIOR")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", default=None)
    add_output_args(p)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--adapter_lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--dense_adapter", default="semantic_pyramid",
                   choices=["semantic_pyramid", "gated_pyramid", "detail_pyramid", "none"])
    p.add_argument("--fpn_out", type=int, default=256)
    p.add_argument("--detector", default="retinanet", choices=["retinanet", "fasterrcnn"])
    p.add_argument("--score_thresh", type=float, default=0.05)
    p.add_argument("--nms_thresh", type=float, default=0.5)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ssm_version", default="mamba3")
    p.add_argument("--ablation_variant", default=None,
                   choices=["ID1", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8", "ID9"])
    p.add_argument("--no_sparse", action="store_true")
    p.add_argument("--no_graph", action="store_true")
    p.add_argument("--no_armg", action="store_true")
    add_perf_args(p)
    add_warmup_args(p, 2)
    args = p.parse_args(argv)
    args.data_dir = clean_str(args.data_dir)
    if args.ckpt:
        args.ckpt = clean_str(args.ckpt)
    if args.output_dir:
        args.output_dir = clean_str(args.output_dir)
    setup_perf()
    args.amp_helper = AmpHelper(args.amp_dtype)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_sparse, use_graph, use_armg = resolve_flags(
        args.ablation_variant, no_sparse=args.no_sparse, no_graph=args.no_graph, no_armg=args.no_armg,
    )
    if args.ablation_variant:
        print(f"  [ablation] {args.ablation_variant} A={use_sparse} B={use_graph} C={use_armg}")

    root = Path(args.data_dir).resolve()
    ann_dir = resolve_ann_dir(root)
    img_dir = resolve_trainval_img_dir(root)
    ids = labeled_stems_in_trainval(ann_dir, img_dir)
    train_ids, val_ids = auto_split_ids(ids, args.val_ratio, args.seed)
    print(f"DIOR train={len(train_ids)} val={len(val_ids)}")
    if args.dry_run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = build_dense_encoder(
            args.ckpt, args.img_size, ssm_version=args.ssm_version,
            dense_adapter=args.dense_adapter,
            use_sparse_ssm=use_sparse, use_graph=use_graph, use_armg=use_armg,
        )
        raw_model, _wrapper = build_detector(
            encoder, fpn_out=args.fpn_out, freeze_encoder=args.freeze_encoder,
            detector=args.detector, score_thresh=args.score_thresh,
            img_size=args.img_size, nms_thresh=args.nms_thresh,
        )
        raw_model.to(device)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
            _ = encoder(dummy)
        print(f"  [dry_run] encoder+detector OK on {device}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)
    train_loader = DataLoader(
        DIORDataset(img_dir, ann_dir, train_ids, args.img_size, True),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, **kw,
    )
    val_loader = DataLoader(
        DIORDataset(img_dir, ann_dir, val_ids, args.img_size, False),
        batch_size=max(1, min(args.batch_size, 4)), shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True, **kw,
    )

    encoder = build_dense_encoder(
        args.ckpt, args.img_size, ssm_version=args.ssm_version,
        dense_adapter=args.dense_adapter,
        use_sparse_ssm=use_sparse, use_graph=use_graph, use_armg=use_armg,
    )
    raw_model, wrapper = build_detector(
        encoder, fpn_out=args.fpn_out, freeze_encoder=args.freeze_encoder,
        detector=args.detector, score_thresh=args.score_thresh,
        img_size=args.img_size, nms_thresh=args.nms_thresh,
    )
    raw_model = maybe_compile(raw_model.to(device), args.compile, device)
    opt = optim.AdamW(build_earthmamba_param_groups(
        raw_model, lr=args.lr, encoder_lr=args.encoder_lr,
        adapter_lr=args.adapter_lr, weight_decay=args.weight_decay,
    ))
    sched = build_warmup_cosine_scheduler(opt, args.warmup_epochs, args.epochs)
    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir, dataset_name="DIOR_ablation")

    best, best_ep, hist = 0.0, 0, []
    for ep in range(args.epochs):
        loss = train_one_epoch(wrapper, train_loader, opt, device, ep, args, args.grad_accum)
        do_eval = (ep + 1) % args.eval_interval == 0 or ep + 1 == args.epochs
        metrics = evaluate(wrapper, val_loader, device, args.score_thresh, args.amp_helper) if do_eval else {}
        sched.step()
        row = {"epoch": ep + 1, "loss": loss, "evaluated": do_eval, **metrics}
        hist.append(row)
        if do_eval:
            score = float(metrics.get("mAP@0.5", 0.0))
            print(f"  loss={loss:.4f} mAP@0.5={score:.4f} best={best:.4f}")
            if score > best:
                best, best_ep = score, ep + 1
                torch.save({"epoch": best_ep, "model": raw_model.state_dict()}, out_dir / "best.pth")

    elapsed = time.time() - start
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": args_to_dict(args), "best_mAP@0.5": best, "best_epoch": best_ep, "history": hist}, f, indent=2)
    print_final_summary(dataset="DIOR", task="det_ablation", script="ablation_study/run/task_dior.py",
                        ckpt=args.ckpt, metrics={"mAP@0.5": best}, out_dir=out_dir,
                        split=f"val({len(val_ids)})", epoch=best_ep, extra_lines=[f"elapsed={elapsed/60:.1f}min"])
