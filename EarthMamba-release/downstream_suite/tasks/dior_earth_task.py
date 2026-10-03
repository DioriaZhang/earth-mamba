from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
BASELINES2 = ROOT / "baselines2"
if str(BASELINES2) not in sys.path:
    sys.path.insert(0, str(BASELINES2))

from lib.downstream_common_v1 import (  # noqa: E402
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
from lib.earthmamba_inria import build_earthmamba_param_groups  # noqa: E402


def _load_baselines2_dior():
    path = BASELINES2 / "tasks" / "dior.py"
    spec = importlib.util.spec_from_file_location("_baselines2_dior_task", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_b2 = _load_baselines2_dior()

DIORDataset = _b2.DIORDataset
CanonicalPyramidBackbone = _b2.CanonicalPyramidBackbone
DetectorWrapper = _b2.DetectorWrapper
RetinaNet = "torchvision RetinaNet"
collate_fn = _b2.collate_fn
resolve_ann_dir = _b2.resolve_ann_dir
resolve_trainval_img_dir = _b2.resolve_trainval_img_dir
labeled_stems_in_trainval = _b2.labeled_stems_in_trainval
auto_split_ids = _b2.auto_split_ids
build_detector = _b2.build_detector
train_one_epoch = _b2.train_one_epoch
evaluate = _b2.evaluate


def _fp32_protocol_marker() -> str:
    return "autocast(enabled=False)"


def add_dior_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", default=None)
    add_output_args(p)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--adapter_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--fpn_out", type=int, default=256)
    p.add_argument("--detector", choices=["retinanet", "fasterrcnn"], default="retinanet")
    p.add_argument("--score_thresh", type=float, default=0.05)
    p.add_argument("--nms_thresh", type=float, default=0.5)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--eval_debug", action="store_true",
                   help="print prediction/GT count and score stats after evaluation")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    add_perf_args(p)
    add_warmup_args(p, default_warmup=1)


@torch.no_grad()
def diagnose_detector_outputs(wrapper: nn.Module, loader: DataLoader, device, max_batches: int = 2) -> None:
    wrapper.model.eval()
    n_img = 0
    n_gt = 0
    n_pred = 0
    score_max = 0.0
    score_sum = 0.0
    score_n = 0
    label_hist: dict[int, int] = {}
    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images = [im.to(device, non_blocking=True) for im in images]
        outs = wrapper.forward_eval(images)
        for out, gt in zip(outs, targets):
            n_img += 1
            n_gt += int(gt["boxes"].shape[0])
            scores = out.get("scores", torch.empty(0))
            labels = out.get("labels", torch.empty(0, dtype=torch.long))
            n_pred += int(scores.numel())
            if scores.numel() > 0:
                score_max = max(score_max, float(scores.max().item()))
                score_sum += float(scores.sum().item())
                score_n += int(scores.numel())
            for lab in labels.tolist():
                label_hist[int(lab)] = label_hist.get(int(lab), 0) + 1
    wrapper.model.train()
    score_mean = score_sum / max(1, score_n)
    top_labels = sorted(label_hist.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print(
        "  [DIOR eval-debug] "
        f"images={n_img} gt_boxes={n_gt} preds_after_model_thresh={n_pred} "
        f"score_max={score_max:.4f} score_mean={score_mean:.4f} top_labels={top_labels}"
    )


def run_dior_training(
    args,
    build_encoder_fn: Callable[[object], nn.Module],
    *,
    task_name: str,
    earthmamba_param_groups: bool = False,
) -> None:
    _ = _fp32_protocol_marker()
    start = time.time()
    setup_perf()
    args.amp_helper = AmpHelper(args.amp_dtype)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.data_dir).expanduser().resolve()
    ann_dir = resolve_ann_dir(root)
    img_dir = resolve_trainval_img_dir(root)
    ids = labeled_stems_in_trainval(ann_dir, img_dir)
    train_ids, val_ids = auto_split_ids(ids, args.val_ratio, args.seed)
    print(f"DIOR train={len(train_ids)} val={len(val_ids)} labeled_in_trainval={len(ids)} task={task_name}")
    print(f"  img={img_dir} ann={ann_dir}")
    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)
    train_loader = DataLoader(
        DIORDataset(img_dir, ann_dir, train_ids, img_size=args.img_size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        **kw,
    )
    val_loader = DataLoader(
        DIORDataset(img_dir, ann_dir, val_ids, img_size=args.img_size, augment=False),
        batch_size=max(1, min(args.batch_size, 4)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        **kw,
    )

    encoder = build_encoder_fn(args)
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
    raw_model = maybe_compile(raw_model, args.compile, device)
    if earthmamba_param_groups:
        params = build_earthmamba_param_groups(
            raw_model,
            lr=args.lr,
            encoder_lr=args.encoder_lr,
            adapter_lr=args.adapter_lr,
            weight_decay=args.weight_decay,
        )
    else:
        enc_body = raw_model.backbone.body
        enc_ids = {id(p) for p in enc_body.parameters()}
        enc_groups = param_groups_weight_decay(enc_body, args.weight_decay)
        for group in enc_groups:
            group["lr"] = args.encoder_lr
        head_params = [p for p in raw_model.parameters() if id(p) not in enc_ids and p.requires_grad]
        params = enc_groups + [
            {"params": [p for p in head_params if p.ndim > 1], "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": [p for p in head_params if p.ndim <= 1], "lr": args.lr, "weight_decay": 0.0},
        ]
        print(f"  lr groups: baseline_encoder={args.encoder_lr:g} detector={args.lr:g}")
    opt = optim.AdamW(params)
    scheduler = build_warmup_cosine_scheduler(opt, args.warmup_epochs, args.epochs)
    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir, dataset_name=f"DIOR_{task_name}")

    best, best_epoch, history = 0.0, 0, []
    for epoch in range(args.epochs):
        loss = train_one_epoch(wrapper, train_loader, opt, device, epoch, args, grad_accum=args.grad_accum)
        do_eval = ((epoch + 1) % args.eval_interval == 0) or (epoch + 1 == args.epochs)
        metrics = evaluate(wrapper, val_loader, device, args.score_thresh, args.amp_helper) if do_eval else {}
        scheduler.step()
        row = {"epoch": epoch + 1, "loss": loss, "evaluated": do_eval}
        row.update(metrics)
        history.append(row)
        if do_eval:
            score = float(metrics.get("mAP@0.5", 0.0))
            print(f"  loss={loss:.4f} mAP@0.5={score:.4f} best={best:.4f}")
            if args.eval_debug or score == 0.0:
                diagnose_detector_outputs(wrapper, val_loader, device)
            if score > best:
                best, best_epoch = score, epoch + 1
                torch.save({"epoch": best_epoch, "metrics": metrics, "model": raw_model.state_dict(), "args": args_to_dict(args)}, out_dir / "best.pth")
        else:
            print(f"  loss={loss:.4f} mAP@0.5=(not evaluated; eval_interval={args.eval_interval}) best={best:.4f}")

    elapsed = time.time() - start
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": args_to_dict(args), "best_mAP@0.5": best, "best_epoch": best_epoch, "history": history}, f, indent=2)
    print_final_summary(
        dataset="DIOR",
        task=f"det_{task_name}",
        script=f"run_dior.py",
        ckpt=args.ckpt,
        metrics={"mAP@0.5": best},
        out_dir=out_dir,
        split=f"val({len(val_ids)} images)",
        epoch=best_epoch,
        extra_lines=[f"elapsed={elapsed/60:.1f}min"],
    )
