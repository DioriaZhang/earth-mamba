"""baselines2 standalone: inria."""

from __future__ import annotations

import sys
from pathlib import Path



import argparse
import json
import random
import time
import warnings
from typing import List, Optional, Tuple

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
    SegMIoUMeter,
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

warnings.filterwarnings("ignore", category=UserWarning)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 2


# ─────────────────────────── 数据集工具 ──────────────────────────────────────

def resolve_inria_base(root: Path) -> Path:
    for base in (
        root,
        root / "NEW2-AerialImageDataset" / "AerialImageDataset",
        root / "AerialImageDataset",
    ):
        if (base / "train" / "images").is_dir() and (base / "train" / "gt").is_dir():
            return base
    raise FileNotFoundError(
        f"未找到 INRIA train/images + train/gt，已尝试: {root} 及子目录"
    )


def resolve_inria_train(root: Path) -> Tuple[Path, Path, Optional[Path]]:
    base = resolve_inria_base(root)
    img_dir, gt_dir = base / "train" / "images", base / "train" / "gt"
    test_img = base / "test" / "images"
    return img_dir, gt_dir, test_img if test_img.is_dir() else None


class INRIAPatchDataset(Dataset):
    def __init__(self, img_dir: Path, gt_dir: Path, patch_size: int,
                 pairs: List[Tuple[str, str]], augment: bool):
        self.patch_size = patch_size
        self.augment = augment
        self.pairs = pairs
        self.img_dir, self.gt_dir = img_dir, gt_dir
        self._to_tensor = transforms.ToTensor()
        self._normalize = transforms.Normalize(MEAN, STD)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        stem, _ = self.pairs[idx]
        img = Image.open(self.img_dir / f"{stem}.tif").convert("RGB")
        mask = Image.open(self.gt_dir / f"{stem}.tif").convert("L")
        w, h = img.size
        ps = self.patch_size
        if self.augment:
            x = random.randint(0, max(0, w - ps))
            y = random.randint(0, max(0, h - ps))
        else:
            x, y = (w - ps) // 2, (h - ps) // 2
        img = img.crop((x, y, x + ps, y + ps))
        mask = mask.crop((x, y, x + ps, y + ps))
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        img_t = self._normalize(self._to_tensor(img))
        mask_t = (torch.from_numpy(np.array(mask)) > 127).long()
        return img_t, mask_t


# ─────────────────────────── UPerNet Decoder ─────────────────────────────────

class PPM(nn.Module):
    """Pyramid Pooling Module（PSP 全局上下文，UPerNet 标配）。"""

    def __init__(self, in_channels: int, pool_channels: int, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_channels, pool_channels, 1, bias=False),
                nn.BatchNorm2d(pool_channels),
                nn.ReLU(True),
            )
            for b in bins
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(bins) * pool_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        parts = [x] + [
            F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        return self.bottleneck(torch.cat(parts, dim=1))


class UPerNetDecoder(nn.Module):
    """UPerNet decoder（对齐 MMSeg 实现，适配 4-stage backbone）。

    流水线：
      stage0-3 特征 (C0-C3) → FPN lateral + top-down
      C3 额外经 PPM 增强全局上下文
      全部 scale 上采样到 C0 大小后 concat → fuse_conv → 分类头
    """

    def __init__(
        self,
        in_dims: List[int],
        fpn_channels: int = 256,
        num_classes: int = NUM_CLASSES,
        pool_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        c0, c1, c2, c3 = in_dims

        self.ppm = PPM(c3, pool_channels)

        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fpn_channels, 1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(True),
            )
            for c in (c0, c1, c2, c3)
        ])

        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(True),
            )
            for _ in (c0, c1, c2, c3)
        ])

        self.fuse = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(fpn_channels * 4, fpn_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(True),
        )
        self.cls = nn.Conv2d(fpn_channels, num_classes, 1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        c0, c1, c2, c3 = feats

        p3 = self.lateral[3](self.ppm(c3))
        p2 = self.lateral[2](c2) + F.interpolate(
            p3, size=c2.shape[2:], mode="bilinear", align_corners=False
        )
        p1 = self.lateral[1](c1) + F.interpolate(
            p2, size=c1.shape[2:], mode="bilinear", align_corners=False
        )
        p0 = self.lateral[0](c0) + F.interpolate(
            p1, size=c0.shape[2:], mode="bilinear", align_corners=False
        )

        p3 = self.smooth[3](p3)
        p2 = self.smooth[2](p2)
        p1 = self.smooth[1](p1)
        p0 = self.smooth[0](p0)

        # 全部上采样到 p0 分辨率后 concat
        target = p0.shape[2:]
        fused = torch.cat([
            p0,
            F.interpolate(p1, size=target, mode="bilinear", align_corners=False),
            F.interpolate(p2, size=target, mode="bilinear", align_corners=False),
            F.interpolate(p3, size=target, mode="bilinear", align_corners=False),
        ], dim=1)
        return self.cls(self.fuse(fused))


# ─────────────────────────── 分割模型 ────────────────────────────────────────

class INRIASeg(nn.Module):
    """通用多 backbone INRIA 建筑分割模型。

    encoder 由 backbone_registry.build_encoder 提供，out_dims 动态传入 UPerNetDecoder。
    """

    def __init__(
        self,
        encoder: nn.Module,
        out_dims: List[int],
        freeze: bool = False,
        fpn_channels: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = UPerNetDecoder(
            in_dims=out_dims,
            fpn_channels=fpn_channels,
            num_classes=NUM_CLASSES,
        )
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.decoder(feats)


# ─────────────────────────── 训练 / 评估循环 ─────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer,
                device: torch.device, epoch: int, args) -> Tuple[float, float]:
    model.train()
    amp: AmpHelper = args.amp_helper
    meter = SegMIoUMeter(NUM_CLASSES, device=device)
    loss_sum, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with amp.autocast():
            logits = F.interpolate(
                model(imgs), size=masks.shape[1:], mode="bilinear", align_corners=False
            )
            loss = F.cross_entropy(logits, masks)
        amp.backward_step(loss, opt, model, args.clip_grad)
        loss_sum += loss.item()
        meter.update(logits.detach().argmax(1), masks)
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return loss_sum / n, meter.compute()


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             amp: AmpHelper) -> float:
    model.eval()
    meter = SegMIoUMeter(NUM_CLASSES, device=device)
    for imgs, masks in tqdm(loader, desc="  Valid", dynamic_ncols=True, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with amp.autocast():
            logits = F.interpolate(
                model(imgs), size=masks.shape[1:], mode="bilinear", align_corners=False
            )
        meter.update(logits.argmax(1), masks)
    return meter.compute()


# ─────────────────────────── 入口 ────────────────────────────────────────────
