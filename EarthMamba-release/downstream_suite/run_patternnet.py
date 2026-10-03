#!/usr/bin/env python3
"""PatternNet linear probe — unified ``--backbone`` (frozen encoder by default)."""

from __future__ import annotations

import argparse
import json
import time

import bootstrap

bootstrap.setup()

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

import backbone as bb
from common import AmpHelper, add_perf_args, loader_kwargs, resolve_out_dir, setup_perf
from lib.verify_patternnet import get_patternnet_loaders


class PatternNetClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        out_dims = getattr(encoder, "out_dims", None)
        if not out_dims:
            raise ValueError("encoder must expose out_dims")
        self.head = nn.Linear(int(out_dims[-1]), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        x = feats[-1] if isinstance(feats, (list, tuple)) else feats
        if x.ndim == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        elif x.ndim == 3:
            x = x.mean(dim=1)
        return self.head(x)


@torch.no_grad()
def evaluate(model, loader, device, amp) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp.autocast():
            pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


def main() -> None:
    p = argparse.ArgumentParser("downstream_suite — PatternNet")
    p.add_argument("--backbone", required=True, choices=bb.BACKBONE_CHOICES)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--freeze_encoder", action="store_true", default=True)
    p.add_argument("--no_freeze_encoder", action="store_false", dest="freeze_encoder")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--max_per_class", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", default=None)
    add_perf_args(p)
    args = p.parse_args()

    setup_perf()
    amp = AmpHelper(args.amp_dtype)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes, classes, meta = get_patternnet_loaders(
        data_dir=args.data_dir, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, val_ratio=args.val_ratio, seed=args.seed,
        max_per_class=args.max_per_class, prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    print(f"[PatternNet] backbone={args.backbone} classes={num_classes} layout={meta.get('layout')}")

    ckpt = bb.resolve_ckpt(args.backbone, args.ckpt, args.ckpt_dir)
    encoder = bb.build_encoder(args.backbone, args.img_size, ckpt)
    if args.freeze_encoder:
        for p_enc in encoder.parameters():
            p_enc.requires_grad = False

    model = PatternNetClassifier(encoder, len(classes)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr if args.freeze_encoder else args.encoder_lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing if not args.freeze_encoder else 0.0)

    out_dir = Path(resolve_out_dir(args.output_dir, data_dir=args.data_dir, dataset_name="PatternNet"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, best_epoch, start = 0.0, 0, time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with amp.autocast():
                loss = criterion(model(x), y)
            if torch.isfinite(loss):
                amp.backward_step(loss, optim, model, args.clip_grad)
        sched.step()
        val_acc = evaluate(model, val_loader, device, amp)
        print(f"  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes}, out_dir / "best.pth")

    elapsed = time.time() - start
    result = {"task": "PatternNet", "backbone": args.backbone, "ckpt": ckpt, "best_epoch": best_epoch,
              "val_acc": best_acc, "elapsed_sec": elapsed, "args": vars(args)}
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[PatternNet] best_epoch={best_epoch} val_acc={best_acc:.4f} → {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
