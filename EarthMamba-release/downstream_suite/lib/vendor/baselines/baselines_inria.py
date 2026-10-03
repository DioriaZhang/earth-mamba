"""
baselines_inria.py — 多模型 INRIA 建筑物语义分割基线对比脚本

支持 7 个 backbone（通过 --backbone 参数切换）：
  earthmamba | skysense | satmae | dofa | clay | roma | rsmamba

流水线：Image → Backbone（全量微调）→ UPerNet Decoder → 分割 Logits → CrossEntropyLoss
主指标：mIoU（建筑 / 背景二分类）

数据布局（自动探测）::
  INRIA/
    NEW2-AerialImageDataset/AerialImageDataset/
      train/images/*.tif + train/gt/*.tif   (180 对, 二值建筑 mask)
      test/images/*.tif                     (180 张, 无公开 gt)
  或扁平 train/images + train/gt

标准做法: 5000×5000 大图随机裁 512 patch 训练；评估用中心 patch。

用法示例：
  python baselines_inria.py --backbone earthmamba --data_dir /data01/.../INRIA \\
      --ckpt /path/to/earthmamba.pth --epochs 50 --batch_size 4
  python baselines_inria.py --backbone satmae --data_dir /data01/.../INRIA \\
      --encoder_embed_dim 768 --epochs 50
"""

from __future__ import annotations

import sys
from pathlib import Path

_BASELINES_ROOT = Path(__file__).resolve().parent
if str(_BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINES_ROOT))
from shared.bootstrap import setup_import_paths

setup_import_paths()

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
from shared.backbone_registry import build_encoder
from shared.paths import resolve_backbone_ckpt

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

def main() -> None:
    t_start = time.time()
    p = argparse.ArgumentParser("baselines_inria — 多模型 INRIA 建筑分割")
    p.add_argument("--data_dir", required=True, help="INRIA 数据根目录")
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
                   help="ViT 系列 embed_dim")
    p.add_argument("--encoder_depth", type=int, default=None,
                   help="ViT 系列 depth（不填则使用 backbone 默认值）")
    p.add_argument("--backbone_model_size", default="huge",
                   help="仅 skysense 使用，huge/large/base")
    add_output_args(p)
    p.add_argument("--patch_size", type=int, default=512, help="训练 patch 大小（图像裁剪尺寸）")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3, help="decoder 学习率")
    p.add_argument("--encoder_lr", type=float, default=1e-5, help="backbone 学习率")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--freeze_encoder", action="store_true", help="冻结 backbone（线性探测）")
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

    # ── 数据集布局探测 ──────────────────────────────────────────────────────
    root = Path(args.data_dir).expanduser().resolve()
    img_dir, gt_dir, test_img = resolve_inria_train(root)
    stems = sorted({p.stem for p in img_dir.glob("*.tif")})
    pairs = [(s, s) for s in stems if (gt_dir / f"{s}.tif").is_file()]
    rng = random.Random(args.seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_ratio))
    val_pairs, train_pairs = shuffled[:n_val], shuffled[n_val:]

    print(f"INRIA base={resolve_inria_base(root)}")
    print(f"  train tiles={len(train_pairs)} val={len(val_pairs)} (从 {len(pairs)} 对 train 划分)")
    if test_img:
        print(f"  test images={len(list(test_img.glob('*.tif')))} (无 gt，不参与评估)")
    print(f"  patch_size={args.patch_size}")

    if args.dry_run:
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
    elif args.backbone == "skysense":
        enc_kwargs["swin_size"] = args.backbone_model_size
    elif args.backbone in ("satmae", "dofa", "clay"):
        enc_kwargs["embed_dim"] = args.encoder_embed_dim
        if args.encoder_depth is not None:
            enc_kwargs["depth"] = args.encoder_depth

    print(f"\n[baselines_inria] backbone={args.backbone}")
    encoder = build_encoder(
        backbone=args.backbone,
        img_size=args.patch_size,
        ckpt=ckpt,
        **enc_kwargs,
    )

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

    model = INRIASeg(
        encoder=encoder,
        out_dims=encoder.out_dims,
        freeze=args.freeze_encoder,
        fpn_channels=256,
    ).to(device)
    model = maybe_compile(model, args.compile, device)

    if args.freeze_encoder:
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        # encoder 小 lr，decoder 大 lr；bias/norm 不加 weight_decay
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
        history.append({
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "train_miou": tr_miou,
            "val_miou": val_miou,
            "lr": optimizer.param_groups[-1]["lr"],
        })
        print(
            f"  train_miou={tr_miou:.4f} val_miou={val_miou:.4f} best={best_miou:.4f}"
        )
        if val_miou > best_miou:
            best_miou, best_epoch = val_miou, epoch + 1
            torch.save(
                {
                    "epoch": best_epoch,
                    "val_miou": val_miou,
                    "model": model.state_dict(),
                    "args": args_to_dict(args),
                },
                out_dir / "best.pth",
            )

    elapsed = time.time() - t_start
    print(
        f"\n[INRIA] 总耗时: {elapsed/60:.1f} min ({elapsed:.0f}s) "
        f"| Backbone: {args.backbone} | ckpt: {ckpt or 'random_init'}"
    )

    summary = {
        "args": args_to_dict(args),
        "backbone": args.backbone,
        "best_val_miou": best_miou,
        "best_epoch": best_epoch,
        "history": history,
        "time_seconds": elapsed,
        "backbone_path": ckpt or "random_init",
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_final_summary(
        dataset="INRIA",
        task="building_seg",
        script=f"baselines_inria.py [{args.backbone}]",
        ckpt=ckpt,
        metrics={"val_mIoU": best_miou},
        out_dir=out_dir,
        split=f"val({len(val_pairs)} tiles)",
        epoch=best_epoch,
        extra_lines=[
            f"backbone={args.backbone}",
            f"elapsed={elapsed/60:.1f}min",
        ],
    )


if __name__ == "__main__":
    main()
