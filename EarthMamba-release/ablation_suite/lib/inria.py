from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path
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

from lib.common import (
    AmpHelper,
    SegMIoUMeter,
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
from lib.earthmamba import build_dense_encoder as build_earthmamba_encoder, build_earthmamba_param_groups
from lib.linux_env import clean_str

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 2


def resolve_inria_base(root: Path) -> Path:
    for base in (root, root / "NEW2-AerialImageDataset" / "AerialImageDataset", root / "AerialImageDataset"):
        if (base / "train" / "images").is_dir() and (base / "train" / "gt").is_dir():
            return base
    raise FileNotFoundError(f"Cannot find INRIA train/images and train/gt under {root}")


def _resolve_ablation_flags(args) -> Tuple[bool, bool, bool]:
    import sys
    from pathlib import Path
    run_dir = Path(__file__).resolve().parents[1]
    if str(run_dir) not in sys.path:
        sys.path.insert(0, str(run_dir))
    from variant_flags import resolve_flags
    if args.ablation_variant:
        return resolve_flags(args.ablation_variant)
    return resolve_flags(None, no_sparse=args.no_sparse, no_graph=args.no_graph, no_armg=args.no_armg)


class INRIAPatchDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        gt_dir: Path,
        patch_size: int,
        pairs: List[str],
        augment: bool,
        *,
        samples_per_tile: int = 1,
        min_building_ratio: float = 0.0,
        crop_tries: int = 1,
        color_jitter: float = 0.0,
        rotate90: bool = False,
        vflip: bool = False,
    ):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.pairs = pairs
        self.augment = augment
        self.samples_per_tile = max(1, samples_per_tile)
        self.min_building_ratio = float(min_building_ratio)
        self.crop_tries = max(1, crop_tries)
        self.rotate90 = rotate90
        self.vflip = vflip
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(MEAN, STD)
        self.jitter = None
        if color_jitter > 0:
            self.jitter = transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(0.05, color_jitter * 0.25),
            )

    def __len__(self) -> int:
        return len(self.pairs) * (self.samples_per_tile if self.augment else 1)

    def __getitem__(self, idx: int):
        if self.augment:
            idx = idx // self.samples_per_tile
        stem = self.pairs[idx]
        image = Image.open(self.img_dir / f"{stem}.tif").convert("RGB")
        mask = Image.open(self.gt_dir / f"{stem}.tif").convert("L")
        w, h = image.size
        ps = self.patch_size
        if self.augment:
            x, y = self._sample_xy(mask, w, h, ps)
        else:
            x, y = (w - ps) // 2, (h - ps) // 2
        image = image.crop((x, y, x + ps, y + ps))
        mask = mask.crop((x, y, x + ps, y + ps))
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if self.augment and self.vflip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if self.augment and self.rotate90:
            k = random.randint(0, 3)
            for _ in range(k):
                image = image.transpose(Image.ROTATE_90)
                mask = mask.transpose(Image.ROTATE_90)
        if self.augment and self.jitter is not None:
            image = self.jitter(image)
        return self.norm(self.to_tensor(image)), (torch.from_numpy(np.array(mask)) > 127).long()

    def _sample_xy(self, mask: Image.Image, w: int, h: int, ps: int) -> Tuple[int, int]:
        max_x, max_y = max(0, w - ps), max(0, h - ps)
        best_xy = (0, 0)
        best_ratio = -1.0
        for _ in range(self.crop_tries):
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            if self.min_building_ratio <= 0:
                return x, y
            ratio = float((np.array(mask.crop((x, y, x + ps, y + ps))) > 127).mean())
            if ratio > best_ratio:
                best_xy, best_ratio = (x, y), ratio
            if ratio >= self.min_building_ratio:
                return x, y
        return best_xy


def _gn(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if groups <= channels and channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class PPM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(in_dim, out_dim, 1, bias=False), _gn(out_dim), nn.ReLU(True))
            for b in bins
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim + out_dim * len(bins), out_dim, 3, padding=1, bias=False),
            _gn(out_dim),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        parts = [x] + [F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False) for stage in self.stages]
        return self.bottleneck(torch.cat(parts, dim=1))


class UPerNetDecoder(nn.Module):
    def __init__(self, in_dims: List[int], fpn_dim: int = 256, num_classes: int = NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.ppm = PPM(in_dims[-1], fpn_dim)
        self.lateral = nn.ModuleList([nn.Conv2d(d, fpn_dim, 1) for d in in_dims[:-1]])
        self.fpn_conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False), _gn(fpn_dim), nn.ReLU(True))
            for _ in in_dims[:-1]
        ])
        self.fuse = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(fpn_dim * len(in_dims), fpn_dim, 3, padding=1, bias=False),
            _gn(fpn_dim),
            nn.ReLU(True),
        )
        self.head = nn.Conv2d(fpn_dim, num_classes, 1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        out_hw = feats[0].shape[-2:]
        x = self.ppm(feats[-1])
        outs = [x]
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.fpn_conv[i](x + self.lateral[i](feats[i]))
            outs.insert(0, x)
        outs = [F.interpolate(o, size=out_hw, mode="bilinear", align_corners=False) for o in outs]
        return self.head(self.fuse(torch.cat(outs, dim=1)))


class INRIASeg(nn.Module):
    def __init__(self, encoder: nn.Module, fpn_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.decoder = UPerNetDecoder(list(encoder.out_dims), fpn_dim=fpn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = logits.softmax(dim=1)[:, 1]
    fg = (target == 1).float()
    inter = (prob * fg).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + fg.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def train_epoch(model: nn.Module, loader: DataLoader, opt, device, epoch: int, args):
    model.train()
    meter = SegMIoUMeter(NUM_CLASSES, device=device)
    total, n, skipped = 0.0, 0, 0
    class_weight = None
    if args.building_weight != 1.0:
        class_weight = torch.tensor([1.0, args.building_weight], dtype=torch.float32, device=device)
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with args.amp_helper.autocast():
            logits = F.interpolate(model(imgs), size=masks.shape[-2:], mode="bilinear", align_corners=False)
            ce = F.cross_entropy(logits, masks, weight=class_weight)
            dice = binary_dice_loss(logits, masks)
            raw_loss = ce + args.dice_weight * dice
        if args.nan_guard and (not torch.isfinite(raw_loss)):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if skipped <= 3:
                warnings.warn(
                    f"[INRIA] skip non-finite batch at epoch={epoch+1}: "
                    f"loss={float(raw_loss.detach().float().cpu()) if raw_loss.numel() == 1 else 'nan'}"
                )
            if skipped >= args.max_nan_batches:
                warnings.warn(f"[INRIA] too many non-finite batches ({skipped}); stop this epoch early")
                break
            continue
        if args.nan_guard and (not torch.isfinite(logits.detach()).all()):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if skipped <= 3:
                warnings.warn(f"[INRIA] skip batch with non-finite logits at epoch={epoch+1}")
            if skipped >= args.max_nan_batches:
                warnings.warn(f"[INRIA] too many non-finite batches ({skipped}); stop this epoch early")
                break
            continue
        loss = raw_loss.clamp(max=args.max_loss) if args.max_loss > 0 else raw_loss
        args.amp_helper.backward_step(loss, opt, model, args.clip_grad)
        total += float(loss.item())
        n += 1
        meter.update(logits.detach().argmax(1), masks)
        pbar.set_postfix(loss=f"{loss.item():.4f}", miou=f"{meter.compute():.4f}", skip=skipped)
    if skipped:
        print(f"  [INRIA] skipped_nonfinite_batches={skipped}")
    return total / max(1, n), meter.compute(), skipped, n


@torch.no_grad()
def _predict(model: nn.Module, imgs: torch.Tensor, size, val_tta: str) -> torch.Tensor:
    logits = [F.interpolate(model(imgs), size=size, mode="bilinear", align_corners=False)]
    if val_tta in ("hflip", "d4"):
        pred = F.interpolate(model(torch.flip(imgs, dims=(-1,))), size=size, mode="bilinear", align_corners=False)
        logits.append(torch.flip(pred, dims=(-1,)))
    if val_tta == "d4":
        for k in (1, 2, 3):
            rot = torch.rot90(imgs, k, dims=(-2, -1))
            pred = F.interpolate(model(rot), size=rot.shape[-2:], mode="bilinear", align_corners=False)
            pred = torch.rot90(pred, -k, dims=(-2, -1))
            logits.append(F.interpolate(pred, size=size, mode="bilinear", align_corners=False))
    return torch.stack(logits).mean(0)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device, args, desc: str = "Valid") -> float:
    model.eval()
    meter = SegMIoUMeter(NUM_CLASSES, device=device)
    for imgs, masks in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with args.amp_helper.autocast():
            logits = _predict(model, imgs, masks.shape[-2:], args.val_tta)
        meter.update(logits.argmax(1), masks)
    return meter.compute()


def main(argv: Optional[List[str]] = None) -> None:
    start = time.time()
    p = argparse.ArgumentParser("INRIA v1 EarthMamba")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", default=None)
    add_output_args(p)
    p.add_argument("--patch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--adapter_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--building_weight", type=float, default=1.0)
    p.add_argument("--dice_weight", type=float, default=0.5)
    p.add_argument("--samples_per_tile", type=int, default=8)
    p.add_argument("--min_building_ratio", type=float, default=0.01)
    p.add_argument("--crop_tries", type=int, default=16)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--overfit_tiles", type=int, default=0)
    p.add_argument("--dense_adapter", default="gated_pyramid", choices=["gated_pyramid", "detail_pyramid", "none"])
    p.add_argument("--val_tta", default="none", choices=["none", "hflip", "d4"])
    p.add_argument("--vflip", action="store_true")
    p.add_argument("--rotate90", action="store_true")
    p.add_argument("--color_jitter", type=float, default=0.05)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--debug_shapes", action="store_true")
    p.add_argument("--nan_guard", action="store_true", default=True,
                   help="skip non-finite loss/logits batches before backward")
    p.add_argument("--no_nan_guard", action="store_false", dest="nan_guard")
    p.add_argument("--max_loss", type=float, default=10.0,
                   help="clamp CE+Dice loss before backward; <=0 disables")
    p.add_argument("--max_nan_batches", type=int, default=8,
                   help="stop current epoch after this many skipped non-finite batches")
    p.add_argument("--stop_on_nan_epoch", action="store_true",
                   help="stop training if an epoch has no finite optimizer step")
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ssm_version", default="mamba3")
    p.add_argument("--ablation_variant", default=None,
                   choices=["ID1", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8", "ID9"])
    p.add_argument("--no_sparse", action="store_true", help="关闭 Sparse SSM Path A")
    p.add_argument("--no_graph", action="store_true", help="关闭 Graph Branch")
    p.add_argument("--no_armg", action="store_true", help="关闭 ARMG")
    add_perf_args(p)
    add_warmup_args(p, default_warmup=5)
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

    use_sparse, use_graph, use_armg = _resolve_ablation_flags(args)
    args.use_sparse_ssm = use_sparse
    args.use_graph = use_graph
    args.use_armg = use_armg
    if args.ablation_variant:
        print(f"  [ablation] {args.ablation_variant} A={use_sparse} B={use_graph} C={use_armg}")

    base = resolve_inria_base(Path(args.data_dir).expanduser().resolve())
    img_dir, gt_dir = base / "train" / "images", base / "train" / "gt"
    stems = sorted(p.stem for p in img_dir.glob("*.tif") if (gt_dir / p.name).is_file())
    random.Random(args.seed).shuffle(stems)
    n_val = max(1, int(len(stems) * args.val_ratio))
    val_ids, train_ids = stems[:n_val], stems[n_val:]
    if args.overfit_tiles > 0:
        train_ids = stems[: args.overfit_tiles]
        val_ids = stems[: args.overfit_tiles]
        print(f"  [INRIA] overfit_tiles={args.overfit_tiles}; train and val use same subset")
    print(f"INRIA base={base}")
    print(f"  train tiles={len(train_ids)} val={len(val_ids)} patch_size={args.patch_size} adapter={args.dense_adapter}")
    if args.dry_run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = build_earthmamba_encoder(
            args.ckpt,
            args.patch_size,
            ssm_version=args.ssm_version,
            dense_adapter=args.dense_adapter,
            use_sparse_ssm=use_sparse,
            use_graph=use_graph,
            use_armg=use_armg,
        )
        model = INRIASeg(encoder).to(device)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, args.patch_size, args.patch_size, device=device)
            out = model(dummy)
        print(f"  [dry_run] encoder+head OK on {device}, out={tuple(out.shape)}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)
    train_loader = DataLoader(
        INRIAPatchDataset(
            img_dir, gt_dir, args.patch_size, train_ids, True,
            samples_per_tile=args.samples_per_tile,
            min_building_ratio=args.min_building_ratio,
            crop_tries=args.crop_tries,
            color_jitter=args.color_jitter,
            rotate90=args.rotate90,
            vflip=args.vflip,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        **kw,
    )
    val_loader = DataLoader(
        INRIAPatchDataset(img_dir, gt_dir, args.patch_size, val_ids, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        **kw,
    )

    encoder = build_earthmamba_encoder(
        args.ckpt,
        args.patch_size,
        ssm_version=args.ssm_version,
        dense_adapter=args.dense_adapter,
        use_sparse_ssm=use_sparse,
        use_graph=use_graph,
        use_armg=use_armg,
    )
    model = INRIASeg(encoder).to(device)
    if args.freeze_encoder:
        for p0 in model.encoder.parameters():
            p0.requires_grad = False
    model = maybe_compile(model, args.compile, device)
    opt = optim.AdamW(
        build_earthmamba_param_groups(
            model,
            lr=args.lr,
            encoder_lr=args.encoder_lr,
            adapter_lr=args.adapter_lr,
            weight_decay=args.weight_decay,
        )
    )
    scheduler = build_warmup_cosine_scheduler(opt, args.warmup_epochs, args.epochs)
    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir, dataset_name="INRIA_v1")

    best, best_epoch, history = 0.0, 0, []
    for epoch in range(args.epochs):
        tr_loss, tr_miou, skipped, finite_steps = train_epoch(model, train_loader, opt, device, epoch, args)
        if finite_steps == 0 and args.stop_on_nan_epoch:
            print(f"  [INRIA] stop: epoch {epoch+1} had no finite training step")
            break
        val_miou = validate(model, val_loader, device, args)
        scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "train_miou": tr_miou,
            "val_miou": val_miou,
            "skipped_nonfinite": skipped,
            "finite_steps": finite_steps,
        })
        print(f"  train_miou={tr_miou:.4f} val_miou={val_miou:.4f} best={best:.4f} skipped={skipped}")
        if val_miou > best:
            best, best_epoch = val_miou, epoch + 1
            torch.save({"epoch": best_epoch, "val_miou": best, "model": model.state_dict(), "args": args_to_dict(args)}, out_dir / "best.pth")

    elapsed = time.time() - start
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": args_to_dict(args), "best_val_miou": best, "best_epoch": best_epoch, "history": history}, f, indent=2)
    print_final_summary(
        dataset="INRIA",
        task="building_seg_v1",
        script="ablation_study/run/task_inria.py",
        ckpt=args.ckpt,
        metrics={"val_mIoU": best},
        out_dir=out_dir,
        split=f"val({len(val_ids)} tiles)",
        epoch=best_epoch,
        extra_lines=[f"elapsed={elapsed/60:.1f}min"],
    )
