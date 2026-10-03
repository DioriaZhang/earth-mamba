"""
PatternNet 下游验证 — EarthMamba / Swin / ViT 多模型对比

数据目录（上传完成后任选其一）::

  布局 A（推荐，TorchGeo / 官方 zip 解压）::

    <data_dir>/
      images/
        airplane/*.jpg
        beach/*.jpg
        ...（38 类，每类约 800 张）

  布局 B（已手动划分）::

    <data_dir>/
      train/<class>/*
      val/<class>/*

  布局 C（类文件夹直接在根目录）::

    <data_dir>/
      airplane/*.jpg
      ...

用法示例::

  # 1) EarthMamba 线性探测（冻结编码器，只训分类头）
  python verify_patternnet.py \
    --data_dir J:/datasets/PatternNet \
    --model_arch earth_mamba \
    --ckpt /hy-tmp/output/pretrain_xxx/epochs/ep30/backbone.pth \
    --freeze_encoder --epochs 30 --img_size 224

  # 2) Swin-B 线性探测（ImageNet 预训练，自动下载权重）
  python verify_patternnet.py \
    --data_dir J:/datasets/PatternNet \
    --model_arch swin_b --freeze_encoder --epochs 30

  # 3) ViT-B/16 线性探测
  python verify_patternnet.py \
    --data_dir J:/datasets/PatternNet \
    --model_arch vit_b_16 --freeze_encoder --epochs 30

  # 4) 全量微调（小 lr）
  python verify_patternnet.py \
    --data_dir J:/datasets/PatternNet \
    --model_arch earth_mamba --ckpt .../backbone.pth \
    --epochs 50 --lr 5e-4

  # 5) 随机初始化 baseline（不传 --ckpt）
  python verify_patternnet.py \
    --data_dir J:/datasets/PatternNet \
    --model_arch earth_mamba --freeze_encoder --epochs 30

  # 6) 批量对比（bash 脚本）
  for arch in swin_t swin_b vit_b_16 earth_mamba; do
    python verify_patternnet.py \
      --data_dir J:/datasets/PatternNet \
      --model_arch $arch \
      --ckpt .../backbone.pth \
      --freeze_encoder --epochs 50 \
      --output_dir ./results/$arch
  done

判读建议:
  - Swin-B / ViT-B ImageNet 预训练 ≈ 70-85% (224px linear probe)
  - EarthMamba 随机初始化 ≈ 2-5%
  - EarthMamba + 有效预训练 ≈ 60-80% (追平或超过 ImageNet 预训练)
  - 新 backbone（no_norm_pix）应高于旧 norm_pix 塌缩 run 的 backbone
"""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import lib.fine_tune_cls as ft
from common import (
    AmpHelper, add_output_args, add_perf_args, add_warmup_args,
    args_to_dict, build_warmup_cosine_scheduler, loader_kwargs,
    maybe_compile, param_groups_weight_decay, resolve_out_dir, setup_perf,
)

warnings.filterwarnings("ignore", category=UserWarning)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 与 pretrain_mae.py EarthMamba small + mamba3 一致
PRETRAIN_SMALL_DEPTHS = [2, 2, 27, 2]
PRETRAIN_SMALL_DIMS = [96, 192, 384, 768]

# ── torchvision 预训练模型注册表 ────────────────────────────────────────────────

# (family, arch_name, params_M)
TORCHVISION_MODELS: dict[str, Tuple[str, str, float]] = {
    "swin_t":   ("swin", "swin_t",   28.3),
    "swin_s":   ("swin", "swin_s",   49.6),
    "swin_b":   ("swin", "swin_b",   87.8),
    "vit_b_16": ("vit",  "vit_b_16", 86.6),
    "vit_l_16": ("vit",  "vit_l_16", 304.3),
}


# ── 数据探测 ───────────────────────────────────────────────────────────────────

def _is_class_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(p.suffix.lower() in IMAGE_EXTS for p in path.iterdir())


def detect_patternnet_layout(data_dir: Path) -> Tuple[str, Path, Optional[Path]]:
    """
    返回 (layout, train_or_all_root, val_root_or_none)。
    layout: "preset_split" | "images" | "root_classes"
    """
    data_dir = data_dir.expanduser().resolve()
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if train_dir.is_dir() and any(_is_class_dir(d) for d in train_dir.iterdir()):
        if not val_dir.is_dir():
            raise FileNotFoundError(f"存在 train/ 但缺少 val/: {val_dir}")
        return "preset_split", train_dir, val_dir

    images_dir = data_dir / "images"
    if images_dir.is_dir():
        class_dirs = [d for d in images_dir.iterdir() if _is_class_dir(d)]
        if len(class_dirs) >= 10:
            return "images", images_dir, None

    class_dirs = [d for d in data_dir.iterdir() if _is_class_dir(d)]
    if len(class_dirs) >= 10:
        return "root_classes", data_dir, None

    raise FileNotFoundError(
        f"无法识别 PatternNet 目录结构: {data_dir}\n"
        "需要 images/<class>/* 或 train|val/<class>/* 或根目录下各类别文件夹。"
    )


def stratified_train_val_indices(
    dataset: ImageFolder,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """按类别分层划分 train/val 索引。"""
    by_class: dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for label in sorted(by_class.keys()):
        indices = by_class[label][:]
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        if len(indices) <= 1:
            train_idx.extend(indices)
            continue
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class PatternNetIndexDataset(Dataset):
    """按 ImageFolder 索引读取，支持独立 train/val 变换。"""

    def __init__(self, root: Path, indices: Sequence[int], transform):
        self.root = root
        self.indices = list(indices)
        self.transform = transform
        base = ImageFolder(root, transform=None)
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx
        self.samples = base.samples

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        path, label = self.samples[self.indices[i]]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _all_indices_limited(samples: List, max_per_class: int, seed: int) -> List[int]:
    if max_per_class <= 0:
        return list(range(len(samples)))
    by_class: dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(samples):
        by_class.setdefault(label, []).append(idx)
    rng = random.Random(seed)
    chosen: List[int] = []
    for indices in by_class.values():
        rng.shuffle(indices)
        chosen.extend(indices[:max_per_class])
    rng.shuffle(chosen)
    return chosen


def get_patternnet_loaders(
    data_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    max_per_class: int = 0,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    linear_probe: bool = False,
) -> Tuple[DataLoader, DataLoader, int, List[str], dict]:
    root = Path(data_dir)
    layout, train_root, val_root = detect_patternnet_layout(root)

    if linear_probe:
        # 线性探测用更强增强（MAE/DINO/BEiT 标准）：RandomResizedCrop + ColorJitter
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    meta = {"layout": layout, "data_dir": str(root)}

    if layout == "preset_split":
        assert val_root is not None
        train_base = ImageFolder(train_root, transform=None)
        val_base = ImageFolder(val_root, transform=None)
        if train_base.classes != val_base.classes:
            raise ValueError("train/ 与 val/ 类别名不一致，请检查划分。")
        train_keep = _all_indices_limited(train_base.samples, max_per_class, seed)
        val_keep = _all_indices_limited(val_base.samples, max_per_class, seed + 1)
        train_ds = PatternNetIndexDataset(train_root, train_keep, train_tf)
        val_ds = PatternNetIndexDataset(val_root, val_keep, val_tf)
        meta["train_dir"] = str(train_root)
        meta["val_dir"] = str(val_root)
        meta["train_size"] = len(train_keep)
        meta["val_size"] = len(val_keep)
        classes = train_base.classes
    else:
        assert val_root is None
        base = ImageFolder(train_root, transform=None)
        meta["image_root"] = str(train_root)
        keep = _all_indices_limited(base.samples, max_per_class, seed)
        sub_samples = [base.samples[i] for i in keep]
        tmp = ImageFolder(train_root, transform=None)
        tmp.samples = sub_samples
        tmp.classes = base.classes
        tmp.class_to_idx = base.class_to_idx
        train_idx, val_idx = stratified_train_val_indices(tmp, val_ratio, seed)
        train_idx = [keep[i] for i in train_idx]
        val_idx = [keep[i] for i in val_idx]
        meta["val_ratio"] = val_ratio
        meta["train_size"] = len(train_idx)
        meta["val_size"] = len(val_idx)
        train_ds = PatternNetIndexDataset(train_root, train_idx, train_tf)
        val_ds = PatternNetIndexDataset(train_root, val_idx, val_tf)
        classes = base.classes

    num_classes = len(classes)
    meta["num_classes"] = num_classes

    _lk = loader_kwargs(num_workers, prefetch_factor, persistent_workers)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, **_lk,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, **_lk,
    )
    return train_loader, val_loader, num_classes, classes, meta


# ── 模型构建 ────────────────────────────────────────────────────────────────────

def _interpolate_pos_embed(model: nn.Module, ckpt_pos: torch.Tensor) -> torch.Tensor:
    pe = model.pos_embed
    if pe is None or pe.shape == ckpt_pos.shape:
        return ckpt_pos
    return F.interpolate(ckpt_pos, size=(pe.shape[2], pe.shape[3]),
                         mode="bicubic", align_corners=False)


def build_torchvision_model(
    arch: str,
    num_classes: int,
    img_size: int,
    freeze_encoder: bool = False,
) -> nn.Module:
    """构建 Swin/ViT 模型，自动下载 ImageNet-1K 预训练权重。"""
    from torchvision.models import swin_t, swin_s, swin_b, vit_b_16, vit_l_16
    from torchvision.models import (
        Swin_T_Weights, Swin_S_Weights, Swin_B_Weights,
        ViT_B_16_Weights, ViT_L_16_Weights,
    )

    _builders = {
        "swin_t":   (swin_t,   Swin_T_Weights.IMAGENET1K_V1),
        "swin_s":   (swin_s,   Swin_S_Weights.IMAGENET1K_V1),
        "swin_b":   (swin_b,   Swin_B_Weights.IMAGENET1K_V1),
        "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
        "vit_l_16": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1),
    }
    builder, weights = _builders[arch]

    if arch.startswith("swin"):
        model = builder(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:  # vit
        model = builder(weights=weights, image_size=img_size)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    if freeze_encoder:
        for name, p in model.named_parameters():
            if "head" not in name:
                p.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  线性探测: 冻结编码器 ({arch}), 可训练参数 {n_trainable:,}")

    return model


class LinearProbeHead(nn.Module):
    """MAE/DINO/BEiT 标准线性探测头：BN1d（固定统计量）+ Linear。

    BN1d affine=False：归一化特征到同一尺度，但不引入额外可训练参数，
    有效提升线性探测精度（参考 MAE, He et al., CVPR 2022）。
    """

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim, affine=False, eps=1e-6)
        self.fc = nn.Linear(feat_dim, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.01)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(x))


class EarthMambaLinearProbe(nn.Module):
    """EarthMamba 线性探测模型：冻结全部预训练权重，只训练 BN1d + Linear 头。

    严格遵循 MAE 线性探测协议（He et al., 2022）：
    - 冻结包括最终 LayerNorm 在内的所有预训练参数
    - 在特征后接 BN1d (affine=False) + Linear
    - 特征提取路径：patch_embed → layers → classifier[:-1]（norm+pool+flatten）
    """

    def __init__(
        self,
        num_classes: int,
        ckpt_path: Optional[str],
        img_size: int,
        patch_size: int = 16,
        ssm_version: str = "mamba3",
    ):
        super().__init__()
        from earth_mamba.models.earth_mamba import EarthMamba

        self.backbone = EarthMamba(
            patch_size=patch_size,
            in_chans=3,
            num_classes=1,  # head 不用，后续替换
            depths=PRETRAIN_SMALL_DEPTHS,
            dims=PRETRAIN_SMALL_DIMS,
            ssm_d_state=64 if ssm_version == "mamba3" else 16,
            ssm_ratio=2.0,
            ssm_version=ssm_version,
            ssm_headdim=64,
            mlp_ratio=4.0,
            drop_path_rate=0.0,  # 线性探测不加 drop path
            posembed=True,
            imgsize=img_size,
            downsample_version="v3",
        )

        if ckpt_path:
            ckpt_path = str(Path(ckpt_path).expanduser())
            print(f"  加载预训练 backbone: {ckpt_path}")
            raw = torch.load(ckpt_path, map_location="cpu")
            sd = raw.get("model", raw.get("state_dict", raw))
            cleaned = {}
            for k, v in sd.items():
                if k.startswith("decoder.") or k == "mask_value":
                    continue
                if k.startswith("encoder."):
                    k = k[len("encoder."):]
                cleaned[k] = v
            if "pos_embed" in cleaned and getattr(self.backbone, "pos_embed", None) is not None:
                cleaned["pos_embed"] = _interpolate_pos_embed(self.backbone, cleaned["pos_embed"])
            for key in ["classifier.head.weight", "classifier.head.bias",
                        "head.weight", "head.bias"]:
                cleaned.pop(key, None)
            missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
            if missing:
                m = [k for k in missing if "classifier" not in k and "head" not in k]
                if m:
                    print(f"  未匹配 ({len(m)}): {m[:5]}...")
            if unexpected:
                print(f"  多余键 ({len(unexpected)}): {unexpected[:5]}...")
        else:
            print("  无预训练权重，随机初始化 (EarthMamba)")

        # 严格冻结所有预训练参数（包括 classifier.norm）
        for p in self.backbone.parameters():
            p.requires_grad = False

        feat_dim = PRETRAIN_SMALL_DIMS[-1]
        self.head = LinearProbeHead(feat_dim, num_classes)
        n_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        print(f"  线性探测: 全冻结 backbone, 可训练参数 {n_trainable:,} (BN1d+Linear)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed is not None:
            pe = self.backbone.pos_embed
            if not self.backbone.channel_first:
                pe = pe.permute(0, 2, 3, 1)
            x = x + pe
        for layer in self.backbone.layers:
            x = layer(x)
        # classifier[:-1] = norm + permute + avgpool + flatten（去掉 head）
        for module in list(self.backbone.classifier.children())[:-1]:
            x = module(x)
        return self.head(x)


def build_earth_mamba_model(
    num_classes: int,
    ckpt_path: Optional[str],
    img_size: int,
    patch_size: int = 16,
    freeze_encoder: bool = False,
    ssm_version: str = "mamba3",
) -> nn.Module:
    """与 pretrain_mae small 配置对齐的 EarthMamba 分类模型。

    freeze_encoder=True 时使用严格线性探测（EarthMambaLinearProbe）：
    全冻结预训练权重，只训练 BN1d + Linear 头（MAE 标准协议）。
    """
    if freeze_encoder:
        return EarthMambaLinearProbe(
            num_classes=num_classes,
            ckpt_path=ckpt_path,
            img_size=img_size,
            patch_size=patch_size,
            ssm_version=ssm_version,
        )

    # ── 全量微调模式 ──────────────────────────────────────────────────────────
    from earth_mamba.models.earth_mamba import EarthMamba

    model = EarthMamba(
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes,
        depths=PRETRAIN_SMALL_DEPTHS,
        dims=PRETRAIN_SMALL_DIMS,
        ssm_d_state=64 if ssm_version == "mamba3" else 16,
        ssm_ratio=2.0,
        ssm_version=ssm_version,
        ssm_headdim=64,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        posembed=True,
        imgsize=img_size,
        downsample_version="v3",
    )

    if ckpt_path:
        ckpt_path = str(Path(ckpt_path).expanduser())
        print(f"  加载预训练 backbone: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu")
        state_dict = raw.get("model", raw.get("state_dict", raw))
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("decoder.") or k == "mask_value":
                continue
            if k.startswith("encoder."):
                k = k[len("encoder."):]
            cleaned[k] = v

        if "pos_embed" in cleaned and getattr(model, "pos_embed", None) is not None:
            cleaned["pos_embed"] = _interpolate_pos_embed(model, cleaned["pos_embed"])

        for key in [
            "classifier.head.weight", "classifier.head.bias",
            "head.weight", "head.bias",
        ]:
            cleaned.pop(key, None)

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            m = [k for k in missing if "classifier" not in k and "head" not in k]
            if m:
                print(f"  未匹配 ({len(m)}): {m[:5]}...")
        if unexpected:
            print(f"  多余键 ({len(unexpected)}): {unexpected[:5]}...")
    else:
        print("  无预训练权重，随机初始化 (EarthMamba)")

    return model


# ── 对比模式 ──────────────────────────────────────────────────────────────────

def run_compare_mode(results_dir: str):
    """读取 results_dir 下所有子目录的 results.json，输出对比表和 CSV。"""
    import glob as _glob
    import csv as _csv
    from io import StringIO

    root = Path(results_dir).expanduser().resolve()
    results_files = sorted(root.glob("*/results.json"))
    if not results_files:
        # 也可能是顶层直接放了 results.json
        results_files = sorted(root.glob("results.json"))
    if not results_files:
        print(f"在 {root} 下未找到任何 results.json")
        return

    models: list[dict] = []
    print(f"\n{'='*70}")
    print(f"PatternNet 下游验证对比")
    print(f"数据源: {root}")
    print(f"{'='*70}\n")

    # 读取所有结果
    for rf in results_files:
        d = json.loads(rf.read_text(encoding="utf-8"))
        args = d.get("args", {})
        arch = args.get("model_arch", "unknown")
        hist = d.get("history", [])
        best = d["best_val_acc"]
        best_ep = max(hist, key=lambda x: x["val_acc"])["epoch"] if hist else "?"
        params_m = d.get("meta", {}).get("num_classes", "?")
        freeze = args.get("freeze_encoder", False)
        ckpt = args.get("ckpt", None)
        models.append({
            "arch": arch,
            "best_acc": best,
            "best_epoch": best_ep,
            "freeze": freeze,
            "ckpt": ckpt is not None,
            "history": hist,
            "dir": rf.parent.name,
        })

    # 按 best_acc 降序排列
    models.sort(key=lambda x: -x["best_acc"])

    # ── 终端表格 ──
    print(f"{'模型':<16s} {'冻结':<5s} {'best_acc':>8s} {'最佳epoch':>9s} {'目录':>30s}")
    print("-" * 70)
    for m in models:
        mode = "LP" if m["freeze"] else "FT"
        print(f"{m['arch']:<16s} {mode:<5s} {m['best_acc']:>8.4f} {m['best_epoch']:>9}  {m['dir']:>30s}")
    print("-" * 70)

    # ── 相对 EarthMamba 的 Delta ──
    em = next((m for m in models if m["arch"] == "earth_mamba"), None)
    if em and len(models) > 1:
        print(f"\n相对 EarthMamba 差异:")
        for m in models:
            if m["arch"] == "earth_mamba":
                continue
            delta = m["best_acc"] - em["best_acc"]
            sign = "+" if delta > 0 else ""
            print(f"  {m['arch']:<16s} {sign}{delta:.4f}")

    # ── CSV 导出 ──
    csv_path = root / "comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        # header: arch, epoch_1_val_acc, ..., epoch_N_val_acc
        all_epochs = sorted({e for m in models for r in m["history"] for e in [r["epoch"]]})
        header = ["model"] + [f"ep{e}" for e in all_epochs]
        writer.writerow(header)
        for m in models:
            acc_by_ep = {r["epoch"]: r["val_acc"] for r in m["history"]}
            row = [m["arch"]] + [f"{acc_by_ep.get(e, ''):.4f}" if e in acc_by_ep else "" for e in all_epochs]
            writer.writerow(row)
    print(f"\n逐 epoch 对比 CSV: {csv_path}")

    # ── 终态判决 ──
    print(f"\n{'='*70}")
    best = models[0]
    if best["arch"] == "earth_mamba":
        print(f"EarthMamba 预训练在 PatternNet 上排名第一 ({best['best_acc']:.4f})，")
        print("说明遥感专用预训练有效。可继续长训预训练或论文消融。")
    else:
        print(f"目前 {best['arch']} 排名第一 ({best['best_acc']:.4f})。")
        print("EarthMamba 需更多预训练 epoch 或调整下游微调策略。")
    print(f"{'='*70}\n")


# ── 主流程 ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    p = argparse.ArgumentParser("PatternNet 下游验证 (EarthMamba / Swin / ViT)")
    p.add_argument("--data_dir", required=True,
                   help="PatternNet 根目录，如 J:/datasets/PatternNet")
    p.add_argument("--model_arch", default="earth_mamba",
                   choices=["earth_mamba", "swin_t", "swin_s", "swin_b",
                            "vit_b_16", "vit_l_16"],
                   help="模型架构。earth_mamba 需 --ckpt 指定预训练路径；swin/vit 自动下载 ImageNet 权重")
    p.add_argument("--ckpt", default=None,
                   help="pretrain backbone.pth 路径（仅 earth_mamba 使用）")
    add_output_args(p)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--freeze_encoder", action="store_true",
                   help="线性探测：只训练分类头")
    p.add_argument("--max_per_class", type=int, default=0,
                   help="每类最多采样 N 张，0=全量")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ssm_version", default="mamba3", choices=["mamba1", "mamba3"])
    p.add_argument("--compare", default=None,
                   help="对比模式：读取指定目录下所有 results.json 并生成对比表（不训练）")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="CrossEntropy label smoothing，0=禁用（线性探测推荐 0.0）")
    add_perf_args(p)
    add_warmup_args(p, default_warmup=5)
    args = p.parse_args()

    if args.compare:
        run_compare_mode(args.compare)
        return

    setup_perf()
    args.amp_helper = AmpHelper(args.amp_dtype)

    is_earth_mamba = args.model_arch == "earth_mamba"
    if is_earth_mamba and args.ckpt is None and not args.freeze_encoder:
        print("提示: earth_mamba 无 --ckpt 且未冻结，等价于随机初始化全量训练。")
    if not is_earth_mamba and args.ckpt:
        print(f"提示: --model_arch={args.model_arch} 不使用 --ckpt（使用 torchvision ImageNet 权重），忽略 --ckpt")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir)
    print(f"输出目录: {out_dir}")

    train_loader, val_loader, num_classes, classes, meta = get_patternnet_loaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers,
        args.val_ratio, args.seed, args.max_per_class,
        args.prefetch_factor, args.persistent_workers,
        linear_probe=args.freeze_encoder,
    )

    print(f"设备: {device}")
    print(f"模型: {args.model_arch}  |  AMP: {args.amp_dtype}")
    if is_earth_mamba:
        print(f"  参数量: ~86M (small)")
    else:
        _, _, params_m = TORCHVISION_MODELS[args.model_arch]
        print(f"  参数量: ~{params_m:.1f}M")
    print(f"数据布局: {meta.get('layout')} | 类别数: {num_classes}")
    print(f"训练 batch 数: {len(train_loader)} | 验证 batch 数: {len(val_loader)}")

    if is_earth_mamba:
        model = build_earth_mamba_model(
            num_classes=num_classes,
            ckpt_path=args.ckpt,
            img_size=args.img_size,
            patch_size=args.patch_size,
            freeze_encoder=args.freeze_encoder,
            ssm_version=args.ssm_version,
        )
    else:
        model = build_torchvision_model(
            arch=args.model_arch,
            num_classes=num_classes,
            img_size=args.img_size,
            freeze_encoder=args.freeze_encoder,
        )
    model = model.to(device)
    model = maybe_compile(model, args.compile, device)

    ls = 0.0 if args.freeze_encoder else args.label_smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    # 线性探测：weight_decay=0（特征已经归一化，正则由 BN 提供）
    # 全量微调：参数组分离（bias/norm 不加 weight_decay）
    if args.freeze_encoder:
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=0.0,
        )
    else:
        optimizer = optim.AdamW(
            param_groups_weight_decay(model, args.weight_decay),
            lr=args.lr,
        )
    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)

    best_acc = 0.0
    history = []
    for epoch in range(args.epochs):
        train_loss, train_acc = ft.train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args)
        val_loss, val_acc = ft.validate(
            model, val_loader, criterion, device, args.amp_helper)
        scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  (best={best_acc:.4f})"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "model": model.state_dict(),
                "classes": classes,
                "args": args_to_dict(args),
            }, out_dir / "best.pth")
            print(f"  * 保存 {out_dir / 'best.pth'}")

    elapsed = time.time() - t_start
    print(f"\n[PatternNet] 总耗时: {elapsed/60:.1f} min ({elapsed:.0f}s) | Backbone: {args.ckpt or 'random_init'}")

    summary = {
        "args": args_to_dict(args),
        "meta": meta,
        "classes": classes,
        "best_val_acc": best_acc,
        "history": history,
        "time_seconds": elapsed,
        "backbone_path": args.ckpt or "random_init",
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"模型: {args.model_arch}  |  最佳 val_acc = {best_acc:.4f}")
    print(f"耗时: {elapsed/60:.1f} min | 结果: {out_dir / 'results.json'}")

    if is_earth_mamba and best_acc < 0.15 and args.ckpt:
        print("警告: EarthMamba 准确率很低，请检查 ckpt 路径或尝试 --img_size 512")
    elif is_earth_mamba and args.ckpt and best_acc >= 0.5:
        print("预训练 backbone 表现正常，可继续长训预训练或做全量微调对比。")


if __name__ == "__main__":
    main()
