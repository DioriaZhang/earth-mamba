"""
baselines_dfc15.py — 多模型 DFC15 多标签场景分类基线对比脚本

支持 7 个 backbone（通过 --backbone 参数切换）：
  earthmamba | skysense | satmae | dofa | clay | roma | rsmamba

流水线：Image → Backbone（全量微调）→ GAP → Linear → Logits → BCEWithLogitsLoss
主指标：macro-mAP；补充 micro-mAP / F1 / exact match

用法示例（在 baselines/ 目录或项目根运行）：
  cd /hy-tmp/baselines
  python baselines_dfc15.py --backbone earthmamba --data_dir /hy-tmp/task/DFC15 \\
      --ckpt /hy-tmp/CKPT/PN-log13-ep14/backbone.pth --epochs 30
  python baselines_dfc15.py --backbone satmae --data_dir /hy-tmp/task/DFC15 --epochs 30
  python baselines_dfc15.py --backbone skysense --data_dir /hy-tmp/task/DFC15 \\
      --backbone_ckpt_dir SkySense/weights --epochs 30
"""

from __future__ import annotations

import sys
from pathlib import Path

# baselines/ 与 downstream_code/ 分离：注入 import 路径
_BASELINES_ROOT = Path(__file__).resolve().parent
if str(_BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINES_ROOT))
from shared.bootstrap import setup_import_paths

setup_import_paths()

import argparse
import csv
import json
import random
import time
import warnings
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

from downstream_common import (
    DEFAULT_OUTPUT_DIR,
    AmpHelper,
    add_output_args,
    add_perf_args,
    add_warmup_args,
    args_to_dict,
    build_warmup_cosine_scheduler,
    loader_kwargs,
    maybe_compile,
    param_groups_weight_decay,
    resolve_out_dir,
    print_final_summary,
    setup_perf,
)
from shared.backbone_registry import build_encoder
from shared.paths import resolve_backbone_ckpt

warnings.filterwarnings("ignore", category=UserWarning)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DFC15_CLASSES = (
    "impervious", "water", "clutter", "vegetation",
    "building", "tree", "boat", "car",
)
NUM_LABELS = len(DFC15_CLASSES)


# ─────────────────────────── 数据集工具 ──────────────────────────────────────

def _parse_label_line(line: str) -> Tuple[str, List[int]]:
    parts = line.replace(",", " ").split()
    name = parts[0]
    vals = [int(float(x)) for x in parts[1:]]
    if len(vals) < NUM_LABELS:
        raise ValueError(f"标签维度不足 {NUM_LABELS}: {line[:80]}")
    return name, vals[:NUM_LABELS]


def resolve_dfc15_base(root: Path) -> Path:
    for base in (root, root / "DFC15_multilabel"):
        if (base / "multilabel.csv").is_file() or (base / "images_tr").is_dir():
            return base
    raise FileNotFoundError(f"未找到 DFC15_multilabel 布局: {root}")


def load_multilabel_csv(csv_path: Path) -> Dict[str, List[int]]:
    """解析 multilabel.csv：首列文件名，后 8 列 0/1。"""
    label_map: Dict[str, List[int]] = {}
    with csv_path.open(encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise FileNotFoundError(f"空 CSV: {csv_path}")
    start = 0
    first = rows[0]
    if first and not first[0].lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png")):
        try:
            [int(float(x)) for x in first[1:1 + NUM_LABELS]]
        except ValueError:
            start = 1
    for row in rows[start:]:
        if len(row) < NUM_LABELS + 1:
            continue
        name = row[0].strip()
        if not name:
            continue
        vals = [int(float(x)) for x in row[1:1 + NUM_LABELS]]
        label_map[name] = vals
        label_map[Path(name).stem] = vals
    if not label_map:
        raise FileNotFoundError(f"{csv_path} 未解析到有效标签行")
    return label_map


def _match_image(img_dir: Path, name: str) -> Optional[Path]:
    stem = Path(name).stem
    for cand in (img_dir / name, img_dir / f"{stem}.tif", img_dir / f"{stem}.tiff",
                 img_dir / f"{stem}.jpg", img_dir / f"{stem}.png"):
        if cand.is_file():
            return cand
    return None


def resolve_dfc15_split(root: Path, split: str) -> Tuple[Path, Path]:
    """旧布局：返回 (img_dir, label_file)。"""
    candidates = [
        (root / split / "images", root / split / "multilabels.txt"),
        (root / "images", root / f"{split}_multilabels.txt"),
        (root / split, root / split / "labels.txt"),
    ]
    for img_dir, lbl_file in candidates:
        if img_dir.is_dir() and lbl_file.is_file():
            return img_dir, lbl_file
    raise FileNotFoundError(f"未找到 DFC15 {split} 旧布局: {root}")


def build_dfc15_samples(
    img_dir: Path, label_map: Dict[str, List[int]]
) -> List[Tuple[Path, torch.Tensor]]:
    samples: List[Tuple[Path, torch.Tensor]] = []
    seen: set = set()
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue
        key = img_path.name
        labels = label_map.get(key) or label_map.get(img_path.stem)
        if labels is None:
            continue
        if key in seen:
            continue
        seen.add(key)
        samples.append((img_path, torch.tensor(labels, dtype=torch.float32)))
    if not samples:
        raise FileNotFoundError(f"{img_dir} 与 multilabel.csv 无匹配样本")
    return samples


class DFC15Dataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        label_file: Optional[Path] = None,
        label_map: Optional[Dict[str, List[int]]] = None,
        img_size: int = 224,
        augment: bool = False,
        strong_augment: bool = False,
    ):
        self.img_size = img_size
        self.augment = augment
        self.strong_augment = strong_augment
        self.samples: List[Tuple[Path, torch.Tensor]] = []
        self._to_tensor = transforms.ToTensor()
        self._normalize = transforms.Normalize(MEAN, STD)
        self._jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
        )
        if label_map is not None:
            self.samples = build_dfc15_samples(img_dir, label_map)
        elif label_file is not None:
            for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                name, labels = _parse_label_line(line)
                p = _match_image(img_dir, name)
                if p is not None:
                    self.samples.append((p, torch.tensor(labels, dtype=torch.float32)))
        if not self.samples:
            raise FileNotFoundError(f"DFC15 {img_dir} 未匹配到带标签图片")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(str(path)).convert("RGB")
        if self.strong_augment:
            scale = (0.2, 1.0)
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=scale, ratio=(3 / 4, 4 / 3)
            )
            img = transforms.functional.resized_crop(
                img, i, j, h, w, (self.img_size, self.img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.3:
                img = self._jitter(img)
        elif self.augment:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = self._normalize(self._to_tensor(img))
        return img, label


# ─────────────────────────── 分类模型 ────────────────────────────────────────

class MultiBackboneClassifier(nn.Module):
    """多 backbone 多标签分类器（DFC15，全量微调）。

    流水线：Image → Backbone → GAP → Linear(embed_dim_last, NUM_LABELS) → Logits
    """

    def __init__(self, encoder: nn.Module, out_dims: List[int], num_labels: int = NUM_LABELS):
        super().__init__()
        self.encoder = encoder
        embed_dim_last = out_dims[-1]
        self.head = nn.Linear(embed_dim_last, num_labels)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"  全量微调：总参数 {n_total:,}（backbone + GAP + Linear）")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        # 取最深特征，GAP → Linear
        feat = feats[-1]  # (B, C, H, W)
        feat = feat.mean(dim=[2, 3])  # GAP → (B, C)
        return self.head(feat)


# ─────────────────────────── 指标计算 ────────────────────────────────────────

def _binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """单标签 AP（与 sklearn average_precision_score 逐步积分一致）。"""
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    ap = 0.0
    prev_recall = 0.0
    for i in range(len(y_true)):
        if y_true[i]:
            ap += precision[i] * (recall[i] - prev_recall)
            prev_recall = recall[i]
    return float(ap)


def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float((y_pred & y_true).sum())
    fp = float((y_pred & ~y_true).sum())
    fn = float((~y_pred & y_true).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2.0 * p * r / (p + r)


def multilabel_metrics(
    logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5
) -> Dict[str, float]:
    """遥感多标签指标（DFC15 / ML-Mamba 口径，纯 numpy）。

    主指标 macro-mAP（mAP 键）：各类 AP 简单平均，顶会主报口径。
    """
    prob = logits.float().sigmoid().numpy()
    y_true = targets.float().numpy().astype(np.int32)
    y_pred = (prob >= thresh).astype(np.int32)
    n_classes = y_true.shape[1]

    per_class_ap = []
    for c in range(n_classes):
        ap = _binary_average_precision(y_true[:, c], prob[:, c])
        per_class_ap.append(ap)
    per_class_ap_dict = {cls: float(ap) for cls, ap in zip(DFC15_CLASSES, per_class_ap)}
    macro_map = float(np.mean(per_class_ap))
    micro_map = _binary_average_precision(y_true.ravel(), prob.ravel())

    micro_f1 = _f1_binary(y_true.ravel(), y_pred.ravel())
    macro_f1 = float(np.mean([
        _f1_binary(y_true[:, c].astype(bool), y_pred[:, c].astype(bool))
        for c in range(n_classes)
    ]))

    tp = float((y_pred & y_true).sum())
    fp = float((y_pred & ~y_true).sum())
    fn = float((~y_pred & y_true).sum())
    micro_prec = tp / (tp + fp + 1e-8)
    micro_rec = tp / (tp + fn + 1e-8)
    exact = float((y_pred == y_true).all(axis=1).mean())

    return {
        "mAP": macro_map,
        "macro_mAP": macro_map,
        "micro_mAP": micro_map,
        "micro_F1": micro_f1,
        "macro_F1": macro_f1,
        "f1": micro_f1,
        "precision": micro_prec,
        "recall": micro_rec,
        "exact_match": exact,
        "per_class_AP": per_class_ap_dict,
    }


# ─────────────────────────── 训练 / 评估循环 ─────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             amp: Optional[AmpHelper] = None) -> Dict[str, float]:
    model.eval()
    logits_all, targets_all = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(device, non_blocking=True)
        if amp is not None and amp.enabled:
            with amp.autocast():
                logits_all.append(model(imgs).cpu())
        else:
            logits_all.append(model(imgs).cpu())
        targets_all.append(lbl)
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    return multilabel_metrics(logits, targets)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args) -> float:
    model.train()
    amp: AmpHelper = args.amp_helper
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d}/{args.epochs}", dynamic_ncols=True)
    for imgs, lbl in pbar:
        imgs = imgs.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast():
            loss = F.binary_cross_entropy_with_logits(model(imgs), lbl)
        amp.backward_step(loss, optimizer, model, args.clip_grad)
        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, n)


# ─────────────────────────── 入口 ────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    p = argparse.ArgumentParser("baselines_dfc15 — 多模型 DFC15 多标签分类")
    p.add_argument("--data_dir", required=True, help="DFC15 数据根目录")
    p.add_argument("--ckpt", default=None, help="backbone 预训练权重路径")
    p.add_argument(
        "--backbone",
        default="earthmamba",
        choices=["earthmamba", "skysense", "satmae", "dofa", "clay", "roma", "rsmamba"],
        help="backbone 名称",
    )
    p.add_argument(
        "--backbone_ckpt_dir",
        default=None,
        help="backbone 权重目录（未指定 --ckpt 时在此目录下自动寻找 .pth 文件）",
    )
    p.add_argument("--ssm_version", default="mamba3", choices=["mamba1", "mamba3"],
                   help="仅 earthmamba 使用")
    p.add_argument("--encoder_embed_dim", type=int, default=768,
                   help="ViT 系列 embed_dim（satmae=768, dofa=768, clay=1024）")
    p.add_argument("--encoder_depth", type=int, default=None,
                   help="ViT 系列 depth（不填则使用 backbone 默认值）")
    p.add_argument("--backbone_model_size", default="huge",
                   help="仅 skysense 使用，huge/large/base")
    add_output_args(p)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="分类 head 学习率")
    p.add_argument("--encoder_lr", type=float, default=1e-5, help="backbone 学习率")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--freeze_encoder", action="store_true",
                   help="冻结 backbone（线性探测模式）")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ablation_variant",
        default=None,
        choices=["BASE", "WO_ARMG", "WO_SPARSE", "WO_GRAPH", "FULL"],
        help="leave-one-out 消融变体（仅 earthmamba）",
    )
    p.add_argument("--no_armg", action="store_true", help="关闭 ARMG（消融）")
    p.add_argument("--no_graph", action="store_true", help="关闭 Latent Graph（消融）")
    add_perf_args(p)
    add_warmup_args(p, default_warmup=3)
    args = p.parse_args()

    use_armg, use_graph = True, True
    if args.backbone == "earthmamba" and (
        args.ablation_variant or args.no_armg or args.no_graph
    ):
        _proj = Path(__file__).resolve().parents[1]
        if str(_proj) not in sys.path:
            sys.path.insert(0, str(_proj))
        try:
            from ablation_study.run.variant_flags import resolve_flags
        except ImportError:
            from ablation_study.variant_flags import resolve_flags
        use_armg, use_graph = resolve_flags(
            args.ablation_variant, no_armg=args.no_armg, no_graph=args.no_graph,
        )
        args.use_armg = use_armg
        args.use_graph = use_graph
        print(f"  [ablation] variant={args.ablation_variant} use_armg={use_armg} use_graph={use_graph}")
    setup_perf()
    args.amp_helper = AmpHelper(args.amp_dtype)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 数据集布局探测 ──────────────────────────────────────────────────────
    root = Path(args.data_dir).expanduser().resolve()
    base = resolve_dfc15_base(root)
    csv_path = base / "multilabel.csv"
    if csv_path.is_file() and (base / "images_tr").is_dir():
        label_map = load_multilabel_csv(csv_path)
        train_img, test_img = base / "images_tr", base / "images_test"
        train_lbl, test_lbl = None, None
        def train_ds_ctor(aug):
            return DFC15Dataset(train_img, label_map=label_map,
                                img_size=args.img_size, augment=aug)
        def test_ds_ctor(aug):
            return DFC15Dataset(test_img, label_map=label_map,
                                img_size=args.img_size, augment=aug)
        layout = "DFC15_multilabel (csv)"
    else:
        label_map = None
        train_img, train_lbl = resolve_dfc15_split(root, "train")
        test_img, test_lbl = resolve_dfc15_split(root, "test")
        def train_ds_ctor(aug):
            return DFC15Dataset(train_img, label_file=train_lbl,
                                img_size=args.img_size, augment=aug)
        def test_ds_ctor(aug):
            return DFC15Dataset(test_img, label_file=test_lbl,
                                img_size=args.img_size, augment=aug)
        layout = "legacy txt"

    print(f"DFC15 base={base} layout={layout}")
    print(f"  train={train_img} | test={test_img}")

    if args.dry_run:
        print(f"  train 样本: {len(train_ds_ctor(False))}")
        print(f"  test  样本: {len(test_ds_ctor(False))}")
        print("  dry_run 完成。")
        return

    # ── 自动查找 ckpt ───────────────────────────────────────────────────────
    ckpt = resolve_backbone_ckpt(
        args.backbone,
        ckpt=args.ckpt,
        ckpt_dir=args.backbone_ckpt_dir,
    )

    # ── 构建 backbone ───────────────────────────────────────────────────────
    enc_kwargs: dict = {}
    if args.backbone == "earthmamba":
        enc_kwargs["ssm_version"] = args.ssm_version
        if hasattr(args, "use_armg"):
            enc_kwargs["use_armg"] = args.use_armg
            enc_kwargs["use_graph"] = args.use_graph
    elif args.backbone == "skysense":
        enc_kwargs["swin_size"] = args.backbone_model_size
    elif args.backbone in ("satmae", "dofa", "clay"):
        enc_kwargs["embed_dim"] = args.encoder_embed_dim
        if args.encoder_depth is not None:
            enc_kwargs["depth"] = args.encoder_depth

    print(f"\n[baselines_dfc15] backbone={args.backbone}")
    encoder = build_encoder(
        backbone=args.backbone,
        img_size=args.img_size,
        ckpt=ckpt,
        **enc_kwargs,
    )

    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _lk = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)
    train_ds = DFC15Dataset(
        train_img,
        label_map=label_map,
        label_file=train_lbl,
        img_size=args.img_size, augment=True, strong_augment=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, **_lk,
    )
    test_loader = DataLoader(
        test_ds_ctor(False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, **_lk,
    )

    model = MultiBackboneClassifier(encoder, encoder.out_dims, NUM_LABELS).to(device)
    model = maybe_compile(model, args.compile, device)

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        # backbone lr 小，head lr 大，bias/norm 不加 weight_decay
        enc_groups = param_groups_weight_decay(model.encoder, args.weight_decay)
        for g in enc_groups:
            g["lr"] = args.encoder_lr
        head_params = list(model.head.parameters())
        optimizer = optim.AdamW(
            enc_groups + [{"params": head_params, "lr": args.lr, "weight_decay": 0.0}],
        )

    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)

    best_map, best_epoch, best_metrics, history = 0.0, 0, {}, []
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        scheduler.step()
        metrics = evaluate(model, test_loader, device, args.amp_helper)
        history.append({
            "epoch": epoch + 1,
            "train_loss": loss,
            "lr": optimizer.param_groups[-1]["lr"],
            **metrics,
        })
        print(
            f"  test macro_mAP={metrics['macro_mAP']:.4f} "
            f"micro_mAP={metrics['micro_mAP']:.4f} "
            f"micro_F1={metrics['micro_F1']:.4f} "
            f"exact={metrics['exact_match']:.4f}"
        )
        if metrics["macro_mAP"] > best_map:
            best_map = metrics["macro_mAP"]
            best_epoch = epoch + 1
            best_metrics = metrics
            torch.save(
                {
                    "epoch": best_epoch,
                    "metrics": metrics,
                    "model": model.state_dict(),
                    "args": args_to_dict(args),
                },
                out_dir / "best.pth",
            )

    elapsed = time.time() - t_start
    print(
        f"\n[DFC15] 总耗时: {elapsed/60:.1f} min ({elapsed:.0f}s) "
        f"| Backbone: {args.backbone} | ckpt: {ckpt or 'random_init'}"
    )

    summary = {
        "args": args_to_dict(args),
        "backbone": args.backbone,
        "classes": list(DFC15_CLASSES),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
        "time_seconds": elapsed,
        "backbone_path": ckpt or "random_init",
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_final_summary(
        dataset="DFC15",
        task="multilabel_cls",
        script=f"baselines_dfc15.py [{args.backbone}]",
        ckpt=ckpt,
        metrics=best_metrics,
        out_dir=out_dir,
        split="test",
        epoch=best_epoch,
        extra_lines=[
            f"backbone={args.backbone}",
            f"elapsed={elapsed/60:.1f}min",
        ],
    )


if __name__ == "__main__":
    main()
