from __future__ import annotations

import sys
from pathlib import Path

_BASELINES_ROOT = Path(__file__).resolve().parent
if str(_BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINES_ROOT))
from shared.bootstrap import setup_import_paths

setup_import_paths()

"""
SECOND 语义变化检测 — 多骨干网络 Siamese + FPN（多类）

支持 backbone: earthmamba / skysense / satmae / dofa / clay / roma / rsmamba
骨干由 shared.backbone_registry.build_encoder 统一加载。

数据布局（自动探测）::
  second_dataset/
    SECOND_train_set/  im1/ im2/ label1/ label2/  (2965 对)
    SECOND_total_test/test/  im1/ im2/ label1/ label2/  (1691 对)

标签约定（7 类）:
  0 = 未变化 (label1 == label2)
  1-6 = 变化后地物类别 (取 label2 语义值，对齐 SECOND 6 类地物 + 背景)

评估指标:
  mIoU  = 7 类（含 class 0 = 未变化）的平均 IoU ← 主指标

用法:
  python baselines_second.py --data_dir /hy-tmp/task/second_dataset --dry_run
  python baselines_second.py --data_dir .../second_dataset --backbone earthmamba \\
      --ckpt ep22.pth --epochs 50
"""

import argparse
import json
import random
import time
import warnings
from typing import List, Optional, Tuple  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from downstream_common import (
    DEFAULT_OUTPUT_DIR,
    AmpHelper,
    SegMIoUMeter,
    add_eval_interval_args,
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
NUM_CLASSES = 7  # 0=未变化, 1-6=变化后语义

BACKBONE_CHOICES = ["earthmamba", "skysense", "satmae", "dofa", "clay", "roma", "rsmamba"]


# ── 数据集路径解析 ─────────────────────────────────────────────────────────────

def _second_root_candidates(root: Path) -> List[Path]:
    """SECOND 数据集根目录候选（兼容 SECOND / second_dataset 两种文件夹名）。"""
    root = root.expanduser().resolve()
    candidates = [
        root,
        root / "second_dataset",
        root.parent / "second_dataset",
    ]
    if root.name.lower() in ("second", "second_dataset"):
        candidates.append(root.parent / "second_dataset")
    seen: set[Path] = set()
    ordered: List[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen and rp.is_dir():
            seen.add(rp)
            ordered.append(rp)
    return ordered


def _resolve_second_split_at(root: Path, split: str) -> Path:
    if split == "train":
        candidates = [
            root / "SECOND_train_set",
            root / "train",
            root / f"SECOND_{split}_set",
        ]
    else:
        candidates = [
            root / "SECOND_total_test" / "test",
            root / "SECOND_test_set",
            root / "test",
            root / f"SECOND_{split}_set",
        ]
    for d in candidates:
        if (d / "im1").is_dir() and (d / "im2").is_dir():
            return d
    raise FileNotFoundError(f"未找到 SECOND {split} (需 im1+im2): {root}")


def resolve_second_split(root: Path, split: str) -> Path:
    tried: List[str] = []
    for base in _second_root_candidates(root):
        try:
            split_dir = _resolve_second_split_at(base, split)
            if base != root.resolve():
                print(f"  [SECOND] 数据根自动解析: {root} → {base}")
            return split_dir
        except FileNotFoundError:
            tried.append(str(base))
    raise FileNotFoundError(
        f"未找到 SECOND {split} (需 im1+im2)。已尝试数据根: {tried}\n"
        f"  提示: 服务器上目录名通常为 second_dataset，请使用:\n"
        f"    --data_dir .../datasets/task/second_dataset"
    )


def resolve_label_dirs(split_dir: Path) -> Tuple[Path, Path]:
    for l1, l2 in [
        (split_dir / "label1_gray", split_dir / "label2_gray"),
        (split_dir / "label1", split_dir / "label2"),
        (split_dir / "label", split_dir / "label"),
    ]:
        if l1.is_dir() and l2.is_dir():
            return l1, l2
    raise FileNotFoundError(f"未找到 label 目录: {split_dir}")


def _list_ids(im1_dir: Path) -> List[str]:
    exts = ("*.png", "*.jpg", "*.tif", "*.PNG")
    ids = set()
    for pat in exts:
        ids.update(p.stem for p in im1_dir.glob(pat))
    out = sorted(ids)
    if not out:
        raise FileNotFoundError(f"{im1_dir} 下无图片")
    return out


def build_change_label(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """0=未变化；变化像素取 label2 值（裁剪到 0-6）。"""
    l1 = l1.astype(np.int64)
    l2 = l2.astype(np.int64)
    changed = l1 != l2
    target = np.zeros_like(l2, dtype=np.int64)
    target[changed] = np.clip(l2[changed], 0, NUM_CLASSES - 1)
    return target


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _find_file(folder: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".tif", ".PNG", ".TIF"):
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"{folder}/{stem}.*")


# ── 数据集 ────────────────────────────────────────────────────────────────────

class SECONDDataset(Dataset):
    def __init__(self, split_dir: Path, img_size: int = 512, augment: bool = False):
        self.split_dir = split_dir
        self.im1_dir = split_dir / "im1"
        self.im2_dir = split_dir / "im2"
        self.l1_dir, self.l2_dir = resolve_label_dirs(split_dir)
        self.ids = _list_ids(self.im1_dir)
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        img1 = _load_rgb(_find_file(self.im1_dir, sid))
        img2 = _load_rgb(_find_file(self.im2_dir, sid))
        l1 = Image.open(_find_file(self.l1_dir, sid)).convert("L")
        l2 = Image.open(_find_file(self.l2_dir, sid)).convert("L")

        if img1.size != (self.img_size, self.img_size):
            img1 = img1.resize((self.img_size, self.img_size), Image.BILINEAR)
            img2 = img2.resize((self.img_size, self.img_size), Image.BILINEAR)
            l1 = l1.resize((self.img_size, self.img_size), Image.NEAREST)
            l2 = l2.resize((self.img_size, self.img_size), Image.NEAREST)

        l1 = np.array(l1, dtype=np.uint8)
        l2 = np.array(l2, dtype=np.uint8)
        target = build_change_label(l1, l2)

        if self.augment:
            if random.random() < 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)
                target = np.fliplr(target).copy()

        t1 = TF.normalize(TF.to_tensor(img1), MEAN, STD)
        t2 = TF.normalize(TF.to_tensor(img2), MEAN, STD)
        return t1, t2, torch.from_numpy(target).long()


# ── 解码器 + 变化检测模型 ──────────────────────────────────────────────────────

def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm：batch_size=1 时 PPM 1×1 分支上 BatchNorm 会报错，改用 GN。"""
    for g in (max_groups, 16, 8, 4, 2, 1):
        if g <= channels and channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class PPM(nn.Module):
    """Pooling Pyramid Module（UPerNet 顶层汇聚）。"""

    def __init__(self, in_dim: int, pool_sizes: Tuple = (1, 2, 3, 6), out_dim: int = 512):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                _gn(out_dim), nn.ReLU(True),
            ) for s in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim + out_dim * len(pool_sizes), out_dim, 3, 1, 1, bias=False),
            _gn(out_dim), nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        parts = [F.interpolate(s(x), size=(h, w), mode="bilinear", align_corners=False)
                 for s in self.stages]
        return self.bottleneck(torch.cat([x, *parts], dim=1))


class UPerNetDecoder(nn.Module):
    """UPerNet 解码头：FPN 侧边路径 + PPM 顶层 + 融合分类头。"""

    def __init__(self, in_dims: List[int] = None, fpn_dim: int = 256,
                 num_classes: int = NUM_CLASSES, ppm_pool_sizes: Tuple = (1, 2, 3, 6)):
        super().__init__()
        if in_dims is None:
            in_dims = [96, 192, 384, 768]
        self.ppm = PPM(in_dims[-1], ppm_pool_sizes, out_dim=fpn_dim)
        self.lateral = nn.ModuleList([nn.Conv2d(d, fpn_dim, 1) for d in in_dims[:-1]])
        self.fpn_conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=False),
                          _gn(fpn_dim), nn.ReLU(True))
            for _ in in_dims[:-1]
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * len(in_dims), fpn_dim, 3, 1, 1, bias=False),
            _gn(fpn_dim), nn.ReLU(True),
        )
        self.head = nn.Conv2d(fpn_dim, num_classes, 1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # 最高分辨率特征尺寸作为输出分辨率
        out_hw = feats[0].shape[2:]
        x = self.ppm(feats[-1])
        fpn_outs = [x]
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[2:], mode="bilinear", align_corners=False)
            x = x + self.lateral[i](feats[i])
            x = self.fpn_conv[i](x)
            fpn_outs.insert(0, x)
        fpn_outs = [F.interpolate(f, size=out_hw, mode="bilinear", align_corners=False)
                    for f in fpn_outs]
        return self.head(self.fuse(torch.cat(fpn_outs, dim=1)))


class SECONDSeg(nn.Module):
    """Siamese 变化检测：对时相图像合批过 encoder，差异特征融合后 UPerNet 解码。"""

    def __init__(self, encoder: nn.Module, freeze: bool = False, fpn_dim: int = 256):
        super().__init__()
        self.enc = encoder
        enc_dims = encoder.out_dims  # [C0, C1, C2, C3]
        # 三元融合投影：[im1, im2, |im1-im2|] → 各自维度
        self.fuse_proj = nn.ModuleList([
            nn.Conv2d(enc_dims[i] * 3, enc_dims[i], 1) for i in range(4)
        ])
        self.dec = UPerNetDecoder(in_dims=list(enc_dims), fpn_dim=fpn_dim)
        if freeze:
            for p in self.enc.parameters():
                p.requires_grad = False

    def forward(self, im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        # 合并 batch 一次过 encoder（≈2x 快于 im1/im2 各跑一遍）
        b = im1.shape[0]
        feats = self.enc(torch.cat([im1, im2], dim=0))
        f1 = [f[:b] for f in feats]
        f2 = [f[b:] for f in feats]
        fused = [
            self.fuse_proj[i](torch.cat([a, bf, torch.abs(a - bf)], dim=1))
            for i, (a, bf) in enumerate(zip(f1, f2))
        ]
        return self.dec(fused)


# ── ckpt 自动解析 ──────────────────────────────────────────────────────────────

def resolve_ckpt(args) -> Optional[str]:
    from shared.paths import resolve_backbone_ckpt
    ckpt = resolve_backbone_ckpt(
        args.backbone,
        ckpt=args.ckpt,
        ckpt_dir=getattr(args, "backbone_ckpt_dir", None),
    )
    if ckpt:
        print(f"  [auto-ckpt] 使用权重: {ckpt}")
    return ckpt


# ── 训练 / 验证 ───────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader, opt, device, epoch: int, args) -> float:
    model.train()
    amp: AmpHelper = args.amp_helper
    loss_sum, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} Train", dynamic_ncols=True)
    for im1, im2, tgt in pbar:
        im1 = im1.to(device, non_blocking=True)
        im2 = im2.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with amp.autocast():
            logits = F.interpolate(model(im1, im2), size=tgt.shape[1:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, tgt, ignore_index=-1)
        amp.backward_step(loss, opt, model, args.clip_grad)
        loss_sum += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return loss_sum / max(1, n)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device, amp: AmpHelper, desc: str = "Valid") -> float:
    """返回 7 类 mIoU（含 class 0 = 未变化）。"""
    model.eval()
    meter = SegMIoUMeter(NUM_CLASSES, device=device)
    for im1, im2, tgt in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False):
        im1 = im1.to(device, non_blocking=True)
        im2 = im2.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        with amp.autocast():
            logits = F.interpolate(model(im1, im2), size=tgt.shape[1:], mode="bilinear", align_corners=False)
        meter.update(logits.argmax(1), tgt)
    return meter.compute()


def inspect_stats(split_dir: Path, img_size: int, n: int = 16) -> None:
    ds = SECONDDataset(split_dir, img_size=img_size, augment=False)
    change_ratios = []
    for i in range(min(n, len(ds))):
        _, _, tgt = ds[i]
        change_ratios.append(float((tgt > 0).float().mean()))
    print(f"  [SECOND] {split_dir.name} 样本={len(ds)} 变化像素占比 mean={np.mean(change_ratios):.4f}")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    p = argparse.ArgumentParser("SECOND 语义变化检测（多骨干）")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", default=None, help="骨干预训练权重路径（优先于 backbone_ckpt_dir）")
    p.add_argument("--backbone", default="earthmamba", choices=BACKBONE_CHOICES,
                   help="骨干网络名称")
    p.add_argument("--ssm_version", default="mamba3",
                   help="仅 earthmamba 生效，SSM 版本（mamba2/mamba3）")
    p.add_argument("--backbone_ckpt_dir", default=None,
                   help="骨干权重目录，未指定 --ckpt 时自动找第一个 .pth")
    add_output_args(p)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3, help="decoder 学习率")
    p.add_argument("--encoder_lr", type=float, default=1e-4, help="encoder 学习率")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--eval_split", default="test", choices=["val", "test"])
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--fpn_dim", type=int, default=256, help="UPerNet FPN 通道数")
    add_perf_args(p)
    add_warmup_args(p, default_warmup=5)
    add_eval_interval_args(p, default=1)
    args = p.parse_args()
    setup_perf()
    args.amp_helper = AmpHelper(args.amp_dtype)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.data_dir).expanduser().resolve()
    train_dir = resolve_second_split(root, "train")
    try:
        eval_dir = resolve_second_split(root, args.eval_split)
    except FileNotFoundError:
        eval_dir = resolve_second_split(root, "test")
        args.eval_split = "test"
    print(f"SECOND train={train_dir} eval={eval_dir} img_size={args.img_size} backbone={args.backbone}")
    inspect_stats(train_dir, args.img_size)
    inspect_stats(eval_dir, args.img_size)

    if args.dry_run:
        print("  dry_run 完成。")
        return

    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _lk = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)
    train_loader = DataLoader(
        SECONDDataset(train_dir, args.img_size, True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, **_lk,
    )
    eval_loader = DataLoader(
        SECONDDataset(eval_dir, args.img_size, False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, **_lk,
    )
    print(f"  train={len(train_loader.dataset)} ({len(train_loader)} batches) "
          f"eval={len(eval_loader.dataset)} ({len(eval_loader)} batches) "
          f"eval_interval={args.eval_interval}")

    # 通过 backbone_registry 统一加载 encoder
    from shared.backbone_registry import build_encoder
    ckpt_path = resolve_ckpt(args)
    encoder = build_encoder(args.backbone, args.img_size, ckpt_path, ssm_version=args.ssm_version)
    n_enc = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"  encoder={args.backbone} 参数量={n_enc:.1f}M out_dims={encoder.out_dims} ckpt={ckpt_path or '随机初始化'}")

    model = SECONDSeg(encoder, freeze=args.freeze_encoder, fpn_dim=args.fpn_dim).to(device)
    model = maybe_compile(model, args.compile, device)
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  总参数量: {n_total:.1f}M")

    if args.freeze_encoder:
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        # encoder 小 lr，bias/norm 不加 weight_decay
        enc_groups = param_groups_weight_decay(model.enc, args.weight_decay)
        for g in enc_groups:
            g["lr"] = args.encoder_lr
        # fuse_proj + dec 使用 decoder lr
        fuse_dec_params = list(model.fuse_proj.parameters()) + list(model.dec.parameters())
        fuse_dec_groups = [
            {"params": [p for p in fuse_dec_params if p.ndim > 1],
             "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": [p for p in fuse_dec_params if p.ndim <= 1],
             "lr": args.lr, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(enc_groups + fuse_dec_groups)
    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)

    best_miou, best_epoch = 0.0, 0
    # 训练时只保存 epoch/loss/miou，不积累完整验证记录以节省内存
    history = []

    for epoch in range(args.epochs):
        t_ep = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, device, epoch, args)

        do_eval = (
            (epoch + 1) % args.eval_interval == 0
            or epoch + 1 == args.epochs
        )
        if do_eval:
            ev_miou = validate(
                model, eval_loader, device, args.amp_helper,
                desc=f"Valid ep{epoch+1}/{args.epochs}",
            )
        else:
            ev_miou = float("nan")

        scheduler.step()
        ep_sec = time.time() - t_ep
        history.append({
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            f"{args.eval_split}_miou": ev_miou if do_eval else None,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(ep_sec, 1),
        })

        if do_eval:
            print(
                f"  ep{epoch+1:03d} loss={tr_loss:.4f} "
                f"{args.eval_split}_miou={ev_miou:.4f} best={best_miou:.4f} ({ep_sec/60:.1f}min)"
            )
            if ev_miou > best_miou:
                best_miou, best_epoch = ev_miou, epoch + 1
                torch.save(
                    {"epoch": best_epoch, "miou": best_miou,
                     "model": model.state_dict(), "args": args_to_dict(args)},
                    out_dir / "best.pth",
                )
        else:
            print(f"  ep{epoch+1:03d} loss={tr_loss:.4f} (skip val) ({ep_sec/60:.1f}min)")

    elapsed = time.time() - t_start
    print(f"\n[SECOND] 总耗时: {elapsed/60:.1f} min ({elapsed:.0f}s) | backbone={args.backbone} ckpt={ckpt_path or 'random_init'}")
    print(f"  best ep{best_epoch}: mIoU={best_miou:.4f}")

    summary = {
        "args": args_to_dict(args),
        f"best_{args.eval_split}_miou": best_miou,
        "best_epoch": best_epoch,
        "history": history,
        "num_classes": NUM_CLASSES,
        "label_scheme": "0=unchanged, 1-6=post-change semantic (label2)",
        "metric_note": "mIoU = 7-class IoU (including class 0 unchanged)",
        "time_seconds": elapsed,
        "backbone": args.backbone,
        "backbone_path": ckpt_path or "random_init",
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_final_summary(
        dataset="SECOND", task="semantic_change_detection", script="baselines_second.py",
        ckpt=ckpt_path,
        metrics={f"{args.eval_split}_mIoU": best_miou},
        out_dir=out_dir,
        split=args.eval_split, epoch=best_epoch,
        extra_lines=[f"backbone={args.backbone}", f"elapsed={elapsed/60:.1f}min"],
    )


if __name__ == "__main__":
    main()
