"""DFC15 多标签分类 — 仅 EarthMamba 消融。"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from lib.common import (
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
from lib.earthmamba import build_cls_encoder
from lib.linux_env import clean_str
from variant_flags import resolve_flags

warnings.filterwarnings("ignore", category=UserWarning)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DFC15_CLASSES = ("impervious", "water", "clutter", "vegetation", "building", "tree", "boat", "car")
NUM_LABELS = len(DFC15_CLASSES)


def _parse_label_line(line: str) -> Tuple[str, List[int]]:
    parts = line.replace(",", " ").split()
    vals = [int(float(x)) for x in parts[1:]]
    return parts[0], vals[:NUM_LABELS]


def resolve_dfc15_base(root: Path) -> Path:
    for base in (root, root / "DFC15_multilabel"):
        if (base / "multilabel.csv").is_file() or (base / "images_tr").is_dir():
            return base
    raise FileNotFoundError(f"未找到 DFC15: {root}")


def load_multilabel_csv(csv_path: Path) -> Dict[str, List[int]]:
    label_map: Dict[str, List[int]] = {}
    with csv_path.open(encoding="utf-8", errors="ignore", newline="") as f:
        rows = list(csv.reader(f))
    start = 0
    if rows and rows[0] and not rows[0][0].lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png")):
        try:
            [int(float(x)) for x in rows[0][1:1 + NUM_LABELS]]
        except ValueError:
            start = 1
    for row in rows[start:]:
        if len(row) < NUM_LABELS + 1:
            continue
        name = row[0].strip()
        if name:
            vals = [int(float(x)) for x in row[1:1 + NUM_LABELS]]
            label_map[name] = vals
            label_map[Path(name).stem] = vals
    return label_map


def resolve_dfc15_split(root: Path, split: str) -> Tuple[Path, Path]:
    for img_dir, lbl in (
        (root / split / "images", root / split / "multilabels.txt"),
        (root / "images", root / f"{split}_multilabels.txt"),
    ):
        if img_dir.is_dir() and lbl.is_file():
            return img_dir, lbl
    raise FileNotFoundError(f"未找到 DFC15 {split}: {root}")


class DFC15Dataset(Dataset):
    def __init__(self, img_dir: Path, label_map=None, label_file=None, img_size=224, augment=False, strong_augment=False):
        self.img_size = img_size
        self.samples = []
        self._to_tensor = transforms.ToTensor()
        self._normalize = transforms.Normalize(MEAN, STD)
        self._jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        if label_map:
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                    continue
                labels = label_map.get(p.name) or label_map.get(p.stem)
                if labels:
                    self.samples.append((p, torch.tensor(labels, dtype=torch.float32)))
        elif label_file:
            for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip():
                    name, labels = _parse_label_line(line.strip())
                    for ext in (".tif", ".tiff", ".jpg", ".png"):
                        p = img_dir / f"{Path(name).stem}{ext}"
                        if p.is_file():
                            self.samples.append((p, torch.tensor(labels, dtype=torch.float32)))
                            break
        self.augment = augment
        self.strong_augment = strong_augment
        if not self.samples:
            raise FileNotFoundError(f"DFC15 {img_dir} 未匹配到带标签图片（label_map={label_map is not None}）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.strong_augment:
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, (0.2, 1.0), (3 / 4, 4 / 3))
            img = transforms.functional.resized_crop(img, i, j, h, w, (self.img_size, self.img_size))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            if self.augment and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self._normalize(self._to_tensor(img)), label


class DFC15Model(nn.Module):
    def __init__(self, encoder, out_dims):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(out_dims[-1], NUM_LABELS)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.encoder(x)[-1].mean(dim=(2, 3)))


def _binary_ap(y_true, y_score):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(1, y_true.sum())
    ap, prev = 0.0, 0.0
    for i in range(len(y_true)):
        if y_true[i]:
            ap += prec[i] * (rec[i] - prev)
            prev = rec[i]
    return float(ap)


def multilabel_metrics(logits, targets, thresh=0.5):
    prob = logits.float().sigmoid().numpy()
    y_true = targets.numpy().astype(np.int32)
    y_pred = (prob >= thresh).astype(np.int32)
    per_class = [_binary_ap(y_true[:, c], prob[:, c]) for c in range(NUM_LABELS)]
    macro = float(np.mean(per_class))
    return {"macro_mAP": macro, "mAP": macro, "micro_mAP": _binary_ap(y_true.ravel(), prob.ravel())}


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    logits_all, targets_all = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(device, non_blocking=True)
        with amp.autocast():
            logits_all.append(model(imgs).cpu())
        targets_all.append(lbl)
    return multilabel_metrics(torch.cat(logits_all), torch.cat(targets_all))


def train_epoch(model, loader, opt, device, epoch, args):
    model.train()
    total, n = 0.0, 0
    for imgs, lbl in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        imgs, lbl = imgs.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with args.amp_helper.autocast():
            loss = F.binary_cross_entropy_with_logits(model(imgs), lbl)
        args.amp_helper.backward_step(loss, opt, model, args.clip_grad)
        total += loss.item()
        n += 1
    return total / max(1, n)


def main(argv=None):
    t0 = time.time()
    p = argparse.ArgumentParser("ablation DFC15")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", default=None)
    add_output_args(p)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--ssm_version", default="mamba3")
    p.add_argument("--ablation_variant", default=None,
                   choices=["ID1", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8", "ID9"])
    p.add_argument("--no_sparse", action="store_true")
    p.add_argument("--no_graph", action="store_true")
    p.add_argument("--no_armg", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    add_perf_args(p)
    add_warmup_args(p, 3)
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
    base = resolve_dfc15_base(root)
    label_map = None
    train_lbl = None
    test_lbl = None
    if (base / "multilabel.csv").is_file():
        label_map = load_multilabel_csv(base / "multilabel.csv")
        train_img, test_img = base / "images_tr", base / "images_test"
    else:
        train_img, train_lbl = resolve_dfc15_split(root, "train")
        test_img, test_lbl = resolve_dfc15_split(root, "test")

    if args.dry_run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = build_cls_encoder(
            args.ckpt, args.img_size, ssm_version=args.ssm_version,
            use_sparse_ssm=use_sparse, use_graph=use_graph, use_armg=use_armg,
        )
        DFC15Model(encoder, encoder.out_dims).to(device)
        print(f"DFC15 dry_run OK: {base}, encoder+head on {device}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)
    train_loader = DataLoader(
        DFC15Dataset(train_img, label_map, train_lbl, args.img_size, True, True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, **kw,
    )
    test_loader = DataLoader(
        DFC15Dataset(test_img, label_map, test_lbl, args.img_size, False, False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, **kw,
    )

    encoder = build_cls_encoder(
        args.ckpt, args.img_size, ssm_version=args.ssm_version,
        use_sparse_ssm=use_sparse, use_graph=use_graph, use_armg=use_armg,
    )
    model = maybe_compile(DFC15Model(encoder, encoder.out_dims).to(device), args.compile, device)
    enc_groups = param_groups_weight_decay(model.encoder, args.weight_decay)
    for g in enc_groups:
        g["lr"] = args.encoder_lr
    opt = optim.AdamW(enc_groups + [{"params": model.head.parameters(), "lr": args.lr, "weight_decay": 0.0}])
    sched = build_warmup_cosine_scheduler(opt, args.warmup_epochs, args.epochs)
    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir)

    best, best_ep, best_m, hist = 0.0, 0, {}, []
    for ep in range(args.epochs):
        loss = train_epoch(model, train_loader, opt, device, ep, args)
        sched.step()
        m = evaluate(model, test_loader, device, args.amp_helper)
        hist.append({"epoch": ep + 1, "train_loss": loss, **m})
        print(f"  macro_mAP={m['macro_mAP']:.4f}")
        if m["macro_mAP"] > best:
            best, best_ep, best_m = m["macro_mAP"], ep + 1, m
            torch.save({"epoch": best_ep, "model": model.state_dict()}, out_dir / "best.pth")

    elapsed = time.time() - t0
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": args_to_dict(args), "best_epoch": best_ep, "best_metrics": best_m, "history": hist}, f, indent=2)
    print_final_summary(dataset="DFC15", task="multilabel_cls", script="ablation_study/run/task_dfc15.py",
                        ckpt=args.ckpt, metrics=best_m, out_dir=out_dir, split="test", epoch=best_ep,
                        extra_lines=[f"elapsed={elapsed/60:.1f}min"])
