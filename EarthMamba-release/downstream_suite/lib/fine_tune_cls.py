"""
遥感图像分类微调 — EarthMamba 下游迁移。

用法:
  # EuroSAT（自动下载，最快验证）
  python fine_tune_cls.py --dataset eurosat --epochs 50 --lr 1e-3

  # 自定义数据集（ImageFolder 格式）
  python fine_tune_cls.py --data_dir /path/to/train --dataset folder --epochs 100

支持的数据集:
  - eurosat: torchvision 内置，10 类，自动下载
  - folder:  通用 ImageFolder 格式（train/class/*.jpg, val/class/*.jpg）
"""

import argparse, json, math, os, sys, time, warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder, EuroSAT
from tqdm import tqdm

from common import (
    args_to_dict, build_warmup_cosine_scheduler,
    maybe_compile, param_groups_weight_decay,
    add_perf_args, AmpHelper, setup_perf, loader_kwargs,
)
from earth_mamba.models.earth_mamba import EarthMamba

warnings.filterwarnings("ignore", category=UserWarning)


# ── 数据集 ───────────────────────────────────────────────────────────────────

def get_dataset(name: str, data_dir: Optional[str], img_size: int):
    common_transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_augment = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.3, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if name == "eurosat":
        root = data_dir or "/tmp/eurosat"
        train_ds = EuroSAT(root=root, transform=train_augment, download=True)
        val_ds   = EuroSAT(root=root, transform=common_transform, download=True)
        # 自动划分 train/val (80/20)
        n = len(train_ds)
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(train_ds, [int(n*0.8), n - int(n*0.8)],
                                        generator=torch.Generator().manual_seed(42))
        num_classes = 10
        return train_ds, val_ds, num_classes

    elif name == "folder":
        assert data_dir is not None, "--data_dir 必须指定"
        p = Path(data_dir)
        train_dir = p / "train"
        val_dir   = p / "val"
        assert train_dir.is_dir(), f"训练目录不存在: {train_dir}"
        train_ds = ImageFolder(train_dir, transform=train_augment)
        val_ds   = ImageFolder(val_dir if val_dir.is_dir() else train_dir,
                               transform=common_transform)
        num_classes = len(train_ds.classes)
        print(f"类别数: {num_classes} | 类别: {train_ds.classes}")
        return train_ds, val_ds, num_classes

    else:
        raise ValueError(f"未知数据集: {name}，支持: eurosat, folder")


# ── 模型 ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, ckpt_path: Optional[str] = None,
                patch_size: int = 16, img_size: int = 224,
                freeze_encoder: bool = False):
    """构建 EarthMamba 分类模型，可选加载预训练权重。"""
    model = EarthMamba(
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes,
        depths=[2, 2, 27, 2],
        dims=[96, 192, 384, 768],
        ssm_d_state=64,
        ssm_ratio=2.0,
        ssm_version="mamba3",
        ssm_headdim=64,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        imgsize=img_size,
    )

    if ckpt_path:
        print(f"加载预训练权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # 兼容两种 checkpoint 格式
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # 去掉 decoder、mask_value 等 pretrain 专用 key
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("decoder.") and k != "mask_value"}

        # encoder.xxx → xxx（如果 state_dict 有 encoder. 前缀）
        if any(k.startswith("encoder.") for k in state_dict):
            state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

        # 去掉 classifier.head.weight/bias 用随机初始化
        for key in ["classifier.head.weight", "classifier.head.bias",
                    "head.weight", "head.bias", "classifier.5.weight", "classifier.5.bias"]:
            state_dict.pop(key, None)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  未匹配的参数 ({len(missing)}): {[k for k in missing if 'head' not in k][:5]}...")
        if unexpected:
            print(f"  多余的参数 ({len(unexpected)}): {unexpected[:5]}...")

    if freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name and "head" not in name:
                param.requires_grad = False
        print("编码器已冻结，仅训练分类头")

    return model


# ── 训练 ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, args):
    model.train()
    amp = getattr(args, "amp_helper", None)
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{args.epochs} Train",
                dynamic_ncols=True)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if amp is not None:
            with amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            amp.backward_step(loss, optimizer, model, args.clip_grad)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})
    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, amp=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, desc="Valid", dynamic_ncols=True):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if amp is not None:
            with amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser("EarthMamba 下游分类微调")
    p.add_argument("--dataset",     default="eurosat", choices=["eurosat", "folder"])
    p.add_argument("--data_dir",    default=None, help="数据集根目录")
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--patch_size",  type=int, default=16)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip_grad",   type=float, default=1.0)
    p.add_argument("--freeze_encoder", action="store_true",
                   help="冻结编码器，只训练分类头")
    p.add_argument("--ckpt",        default=None,
                   help="预训练 checkpoint 路径（.pth）")
    p.add_argument("--output_dir",  default="./finetune_output")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="线性 warmup epoch 数")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="CrossEntropy label smoothing（0=禁用）")
    add_perf_args(p)
    args = p.parse_args()

    setup_perf()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp_helper = AmpHelper(args.amp_dtype)
    print(f"设备: {device}  |  AMP: {args.amp_helper.amp_dtype}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    train_ds, val_ds, num_classes = get_dataset(args.dataset, args.data_dir, args.img_size)
    _lk = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, **_lk,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, **_lk,
    )
    print(f"训练集: {len(train_ds)} | 验证集: {len(val_ds)} | 类别: {num_classes}")

    # 模型
    model = build_model(num_classes, ckpt_path=args.ckpt,
                        patch_size=args.patch_size, img_size=args.img_size,
                        freeze_encoder=args.freeze_encoder)
    model = model.to(device)
    model = maybe_compile(model, getattr(args, "compile", False), device)

    # 优化器（参数组分离：bias/norm 不加 weight_decay）
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        param_groups_weight_decay(model, args.weight_decay),
        lr=args.lr,
    )
    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)

    # 训练
    best_acc = 0
    results = []
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args)
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, args.amp_helper)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        results.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
        })

        print(f"  → train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  (best={best_acc:.4f})")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch + 1, "val_acc": val_acc,
                "model": model.state_dict(),
                "args": args_to_dict(args),
            }, out_dir / "best.pth")
            print(f"  ★ 保存 best.pth (val_acc={val_acc:.4f})")

    # 保存结果
    with open(out_dir / "results.json", "w") as f:
        json.dump({"args": args_to_dict(args), "best_acc": best_acc, "history": results}, f, indent=2)
    print(f"\n完成！最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()
