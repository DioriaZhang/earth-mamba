"""DeepGlobe 2018 land-cover semantic segmentation — INRIA-style UPerNet protocol."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from lib.earth_mamba_variants import is_earth_mamba
from lib.train_utils import (
    AmpHelper,
    UPerNetSeg,
    add_perf_args,
    add_warmup_args,
    args_to_dict,
    build_warmup_cosine_scheduler,
    checkpoint_payload,
    loader_kwargs,
    load_training_resume,
    maybe_compile,
    param_groups_weight_decay,
    print_final_summary,
    resolve_out_dir,
    setup_perf,
)

DEEPGLOBE_NUM_CLASSES = 7
SCORE_CLASSES = 6  # mIoU-6: classes 0–5, exclude unknown (6)
UNKNOWN_CLASS = 6
PROTOCOL_VERSION = "v3_upernet_deepglobe_inria"

CLASS_NAMES = (
    "urban",
    "agriculture",
    "rangeland",
    "forest",
    "water",
    "barren",
    "unknown",
)

# Kaggle class_dict.csv 常见写法：urban / Urban land / urban_land 等
_CLASS_NAME_ALIASES: dict[str, int] = {
    "urban": 0,
    "agriculture": 1,
    "rangeland": 2,
    "forest": 3,
    "water": 4,
    "barren": 5,
    "unknown": 6,
}


def _normalize_class_key(name: str) -> Optional[str]:
    key = name.strip().lower().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    if key in _CLASS_NAME_ALIASES:
        return key
    for alias in _CLASS_NAME_ALIASES:
        if alias in key or key.startswith(alias):
            return alias
    return None


def _parse_rgb_row(row: dict) -> Optional[tuple[int, int, int]]:
    if all(k in row and row[k] not in (None, "") for k in ("r", "g", "b")):
        return int(row["r"]), int(row["g"]), int(row["b"])
    rgb = row.get("r,g,b") or row.get("rgb")
    if rgb:
        parts = [int(x) for x in str(rgb).replace(" ", "").split(",")]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    return None

DEFAULT_RGB_COLORS = (
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 255),
    (0, 0, 0),
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class TilePair:
    """Resolved image/mask paths — never reconstruct filenames from ids."""

    tile_id: str
    image_path: Path
    mask_path: Path


@dataclass
class DeepGlobeLayout:
    root: Path
    train_pairs: list[TilePair]
    val_pairs: list[TilePair]
    val_source: str
    rgb_colors: tuple[tuple[int, int, int], ...]


def load_rgb_palette(root: Path) -> tuple[tuple[int, int, int], ...]:
    csv_path = root / "class_dict.csv"
    if not csv_path.is_file():
        return DEFAULT_RGB_COLORS

    colors: list[tuple[int, int, int]] = list(DEFAULT_RGB_COLORS)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name") or row.get("class") or row.get("label")
            if not name:
                continue
            class_key = _normalize_class_key(name)
            if class_key is None:
                continue
            rgb = _parse_rgb_row(row)
            if rgb is None:
                continue
            colors[_CLASS_NAME_ALIASES[class_key]] = rgb

    return tuple(colors)


def _discover_pairs(split_dir: Path) -> list[TilePair]:
    """Only keep samples where both image and mask exist on disk."""
    if not split_dir.is_dir():
        return []

    seen: set[str] = set()
    pairs: list[TilePair] = []

    def _try_add(tile_id: str, image_path: Path, mask_path: Path) -> None:
        if tile_id in seen:
            return
        if not image_path.is_file() or not mask_path.is_file():
            return
        seen.add(tile_id)
        pairs.append(
            TilePair(
                tile_id=tile_id,
                image_path=image_path.resolve(),
                mask_path=mask_path.resolve(),
            )
        )

    for sat_path in sorted(split_dir.glob("*_sat.jpg")):
        tile_id = sat_path.name[: -len("_sat.jpg")]
        _try_add(tile_id, sat_path, split_dir / f"{tile_id}_mask.png")

    for mask_path in sorted(split_dir.glob("*_mask.png")):
        tile_id = mask_path.name[: -len("_mask.png")]
        _try_add(tile_id, split_dir / f"{tile_id}_sat.jpg", mask_path)

    pairs.sort(key=lambda p: p.tile_id)
    return pairs


def resolve_deepglobe_layout(
    root: Path,
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DeepGlobeLayout:
    root = root.expanduser().resolve()
    train_dir = root / "train"
    val_dir = root / "valid"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"DeepGlobe train/ not found under {root}")

    all_train_pairs = _discover_pairs(train_dir)
    if not all_train_pairs:
        raise FileNotFoundError(
            f"No labeled pairs under {train_dir} (need both *_sat.jpg and *_mask.png)"
        )

    official_val_pairs = _discover_pairs(val_dir)
    if official_val_pairs:
        train_pairs = all_train_pairs
        val_pairs = official_val_pairs
        val_source = "official_valid"
    else:
        rng = random.Random(seed)
        shuffled = list(all_train_pairs)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        val_pairs = sorted(shuffled[:n_val], key=lambda p: p.tile_id)
        train_pairs = sorted(shuffled[n_val:], key=lambda p: p.tile_id)
        val_source = f"holdout_{val_ratio:g}@train"

    return DeepGlobeLayout(
        root=root,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        val_source=val_source,
        rgb_colors=load_rgb_palette(root),
    )


def _binarized_color_code(r: int, g: int, b: int, *, threshold: int = 128) -> int:
    return (
        ((r >= threshold) << 2)
        | ((g >= threshold) << 1)
        | (b >= threshold)
    )


def _build_binarized_lut(
    palette: tuple[tuple[int, int, int], ...],
    *,
    threshold: int = 128,
) -> np.ndarray:
    lut = np.full(8, UNKNOWN_CLASS, dtype=np.int64)
    for idx, color in enumerate(palette):
        code = _binarized_color_code(color[0], color[1], color[2], threshold=threshold)
        lut[code] = idx
    return lut


def rgb_mask_to_index(
    mask_rgb: np.ndarray,
    palette: tuple[tuple[int, int, int], ...],
    *,
    threshold: int = 128,
    lut: Optional[np.ndarray] = None,
) -> np.ndarray:
    """官方建议：mask 通道先按 threshold 二值化，再匹配 RGB 调色板（向量化）。"""
    mask_rgb = np.asarray(mask_rgb)
    if mask_rgb.ndim == 2:
        mask_rgb = np.stack([mask_rgb] * 3, axis=-1)
    r = mask_rgb[..., 0] >= threshold
    g = mask_rgb[..., 1] >= threshold
    b = mask_rgb[..., 2] >= threshold
    code = (r.astype(np.uint8) << 2) | (g.astype(np.uint8) << 1) | b.astype(np.uint8)
    if lut is None:
        lut = _build_binarized_lut(palette, threshold=threshold)
    return lut[code]


class DeepGlobeTileCache:
    """Preload 2448² tiles into RAM once; DataLoader workers share via fork CoW on Linux."""

    def __init__(self, palette: tuple[tuple[int, int, int], ...]) -> None:
        self.palette = palette
        self._lut = _build_binarized_lut(palette)
        self._rgb: dict[str, np.ndarray] = {}
        self._mask: dict[str, np.ndarray] = {}

    def load(self, pair: TilePair) -> tuple[np.ndarray, np.ndarray]:
        key = pair.tile_id
        if key in self._rgb:
            return self._rgb[key], self._mask[key]
        rgb = np.asarray(Image.open(pair.image_path).convert("RGB"), dtype=np.uint8)
        mask_rgb = np.asarray(Image.open(pair.mask_path).convert("RGB"), dtype=np.uint8)
        mask = rgb_mask_to_index(mask_rgb, self.palette, lut=self._lut)
        self._rgb[key] = rgb
        self._mask[key] = mask
        return rgb, mask

    def preload(self, pairs: list[TilePair], *, desc: str = "tiles") -> None:
        seen: set[str] = set()
        todo: list[TilePair] = []
        for pair in pairs:
            if pair.tile_id in seen:
                continue
            seen.add(pair.tile_id)
            todo.append(pair)
        if not todo:
            return
        for pair in tqdm(todo, desc=f"  [cache] preload {desc}", dynamic_ncols=True):
            self.load(pair)
        mb = sum(self._rgb[p.tile_id].nbytes + self._mask[p.tile_id].nbytes for p in todo) / (1024**2)
        print(f"  [cache] {len(todo)} {desc} in RAM (~{mb:.0f} MB)", flush=True)


def add_cache_args(parser) -> None:
    g = parser.add_argument_group("data cache")
    g.add_argument("--cache_tiles", action="store_true", default=True, help="preload 2448² tiles to RAM (default on)")
    g.add_argument("--no_cache_tiles", action="store_false", dest="cache_tiles")
    g.add_argument("--val_every", type=int, default=1, help="validate every N epochs (1=every epoch)")


def build_tile_cache(layout: DeepGlobeLayout, *, enabled: bool) -> Optional[DeepGlobeTileCache]:
    if not enabled:
        print("  [cache] disabled; each patch will decode JPEG/PNG from disk", flush=True)
        return None
    cache = DeepGlobeTileCache(layout.rgb_colors)
    cache.preload(layout.train_pairs + layout.val_pairs, desc="train+val")
    return cache


class DeepGlobePatchDataset(Dataset):
    def __init__(
        self,
        layout: DeepGlobeLayout,
        split: str,
        patch_size: int,
        *,
        augment: bool = False,
        samples_per_tile: int = 1,
        color_jitter: float = 0.05,
        rotate90: bool = False,
        vflip: bool = False,
        tile_cache: Optional[DeepGlobeTileCache] = None,
    ):
        self.layout = layout
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        self.samples_per_tile = max(1, samples_per_tile)
        self.rotate90 = rotate90
        self.vflip = vflip
        self.pairs = layout.train_pairs if split == "train" else layout.val_pairs
        self.tile_cache = tile_cache
        self._lut = tile_cache._lut if tile_cache is not None else _build_binarized_lut(layout.rgb_colors)
        self.norm = transforms.Normalize(MEAN, STD)
        self.jitter = None
        if color_jitter > 0 and augment:
            self.jitter = transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(0.05, color_jitter * 0.25),
            )

    def _load_tile(self, pair: TilePair) -> tuple[np.ndarray, np.ndarray]:
        if self.tile_cache is not None:
            return self.tile_cache.load(pair)
        rgb = np.asarray(Image.open(pair.image_path).convert("RGB"), dtype=np.uint8)
        mask_rgb = np.asarray(Image.open(pair.mask_path).convert("RGB"), dtype=np.uint8)
        mask = rgb_mask_to_index(mask_rgb, self.layout.rgb_colors, lut=self._lut)
        return rgb, mask

    def __len__(self) -> int:
        return len(self.pairs) * (self.samples_per_tile if self.augment else 1)

    def __getitem__(self, idx: int):
        if self.augment:
            idx = idx // self.samples_per_tile
        pair = self.pairs[idx]
        rgb, mask = self._load_tile(pair)

        h, w = rgb.shape[:2]
        ps = self.patch_size
        if self.augment:
            x = random.randint(0, max(0, w - ps))
            y = random.randint(0, max(0, h - ps))
        else:
            x, y = max(0, (w - ps) // 2), max(0, (h - ps) // 2)

        rgb_patch = rgb[y : y + ps, x : x + ps]
        mask_patch = mask[y : y + ps, x : x + ps]

        if self.augment and random.random() < 0.5:
            rgb_patch = np.ascontiguousarray(rgb_patch[:, ::-1])
            mask_patch = np.ascontiguousarray(mask_patch[:, ::-1])

        if self.augment and self.vflip and random.random() < 0.5:
            rgb_patch = np.ascontiguousarray(rgb_patch[::-1])
            mask_patch = np.ascontiguousarray(mask_patch[::-1])

        if self.augment and self.rotate90:
            k = random.randint(0, 3)
            if k:
                rgb_patch = np.ascontiguousarray(np.rot90(rgb_patch, k))
                mask_patch = np.ascontiguousarray(np.rot90(mask_patch, k))

        if self.augment and self.jitter is not None:
            rgb_patch = np.asarray(self.jitter(Image.fromarray(rgb_patch)), dtype=np.uint8)

        arr = rgb_patch.astype(np.float32) / 255.0
        arr = (arr - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
        img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return img_t, torch.from_numpy(mask_patch.astype(np.int64))


def inspect_deepglobe_stats(dataset: DeepGlobePatchDataset, n: int = 4) -> None:
    """Lightweight check: downsample masks only (no full 2448² image load)."""
    if not dataset.pairs:
        print(f"  [DeepGlobe check] split={dataset.split} tiles=0 (empty)")
        return

    n = min(n, len(dataset.pairs))
    class_hist = np.zeros(DEEPGLOBE_NUM_CLASSES, dtype=np.int64)
    preview = 256
    for i in range(n):
        pair = dataset.pairs[i]
        with Image.open(pair.mask_path) as mask_img:
            small = np.asarray(
                mask_img.convert("RGB").resize((preview, preview), Image.NEAREST),
                dtype=np.uint8,
            )
        mask = rgb_mask_to_index(small, dataset.layout.rgb_colors)
        for c in range(DEEPGLOBE_NUM_CLASSES):
            class_hist[c] += int((mask == c).sum())
    total = max(1, int(class_hist.sum()))
    ratios = class_hist / total
    print(
        f"  [DeepGlobe check] split={dataset.split} tiles={len(dataset.pairs)} "
        f"samples={len(dataset)}",
        flush=True,
    )
    print(f"    val_source={dataset.layout.val_source}", flush=True)
    print(f"    example: {dataset.pairs[0].image_path.name}", flush=True)
    for c, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {ratios[c]:.4f}", flush=True)


def apply_earth_deepglobe_defaults(args, *, backbone_name: str) -> None:
    """Earth-mamba 专用：INRIA 风格 adapter + CE + mIoU-6 Dice。"""
    if not is_earth_mamba(backbone_name):
        return
    if getattr(args, "dense_adapter", None) in (None, "none"):
        args.dense_adapter = "gated_pyramid"
    if float(getattr(args, "dice_weight", 0.0)) <= 0:
        args.dice_weight = 0.5
    if getattr(args, "adapter_lr", None) is None:
        args.adapter_lr = 1e-3
    args.nan_guard = True


def build_optimizer(model: UPerNetSeg, args, *, backbone_name: str):
    if args.freeze_encoder:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    if is_earth_mamba(backbone_name):
        from lib.earthmamba_adapter import build_earthmamba_param_groups, ensure_registered

        ensure_registered()

        adapter_lr = getattr(args, "adapter_lr", None) or args.lr
        groups = build_earthmamba_param_groups(
            model,
            lr=args.lr,
            encoder_lr=args.encoder_lr,
            adapter_lr=adapter_lr,
            weight_decay=args.weight_decay,
        )
        return torch.optim.AdamW(groups)

    enc_groups = param_groups_weight_decay(model.encoder, args.weight_decay)
    for g in enc_groups:
        g["lr"] = args.encoder_lr
    dec_groups = param_groups_weight_decay(model.decoder, args.weight_decay)
    for g in dec_groups:
        g["lr"] = args.lr
    return torch.optim.AdamW(enc_groups + dec_groups)


def confusion_matrix_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = DEEPGLOBE_NUM_CLASSES,
) -> torch.Tensor:
    pred = pred.reshape(-1).to(torch.int64)
    target = target.reshape(-1).to(torch.int64)
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    if not torch.any(valid):
        return torch.zeros((num_classes, num_classes), dtype=torch.float64, device=pred.device)
    idx = target[valid] * num_classes + pred[valid]
    return torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes).to(torch.float64)


def miou_from_confusion(
    conf: torch.Tensor,
    *,
    score_classes: int = SCORE_CLASSES,
) -> tuple[float, list[Optional[float]]]:
    conf = conf.to(torch.float64)
    inter = torch.diag(conf)
    union = conf.sum(dim=1) + conf.sum(dim=0) - inter
    valid_ious: list[float] = []
    per_class: list[Optional[float]] = []
    for c in range(conf.shape[0]):
        if c >= score_classes:
            per_class.append(None)
            continue
        if union[c] > 0:
            v = float((inter[c] / union[c]).item())
            valid_ious.append(v)
            per_class.append(v)
        else:
            per_class.append(None)
    return (float(np.mean(valid_ious)) if valid_ious else 0.0), per_class


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int = SCORE_CLASSES,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = (target < num_classes).unsqueeze(1).float()
    probs = probs[:, :num_classes] * valid
    target_oh = target_oh * valid
    dims = (0, 2, 3)
    inter = (probs * target_oh).sum(dims)
    denom = probs.sum(dims) + target_oh.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    present = denom > 0
    if not bool(present.any()):
        return logits.new_tensor(0.0)
    return 1.0 - dice[present].mean()


def segmentation_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    dice_weight: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, masks, ignore_index=UNKNOWN_CLASS)
    if dice_weight <= 0:
        return ce
    dice = multiclass_dice_loss(logits, masks)
    return ce + dice_weight * dice


def train_one_epoch(model, loader, optimizer, device, epoch, args):
    model.train()
    total_loss, n = 0.0, 0
    skipped = 0
    dice_weight = float(getattr(args, "dice_weight", 0.0))
    nan_guard = bool(getattr(args, "nan_guard", False))
    max_nan_batches = int(getattr(args, "max_nan_batches", 8))
    max_loss = float(getattr(args, "max_loss", 0.0))
    epoch_conf = torch.zeros((DEEPGLOBE_NUM_CLASSES, DEEPGLOBE_NUM_CLASSES), dtype=torch.float64)
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1:03d}/{args.epochs} Train", dynamic_ncols=True)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with args.amp_helper.autocast():
            logits = model(imgs)
            logits_up = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            loss = segmentation_loss(logits_up, masks, dice_weight=dice_weight)
        if nan_guard and (not torch.isfinite(loss) or not torch.isfinite(logits_up.detach()).all()):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if skipped >= max_nan_batches:
                print(f"  [train] too many non-finite batches ({skipped}); stop epoch early")
                break
            continue
        if max_loss > 0:
            loss = loss.clamp(max=max_loss)
        args.amp_helper.backward_step(loss, optimizer, model, args.clip_grad)
        with torch.no_grad():
            epoch_conf += confusion_matrix_torch(logits_up.argmax(1), masks).cpu()
            miou6, _ = miou_from_confusion(epoch_conf)
        total_loss += float(loss.detach().item())
        n += 1
        if n == 1 or n % 20 == 0:
            pbar.set_postfix(loss=f"{float(loss.detach()):.4f}", skip=skipped)
    miou6, _ = miou_from_confusion(epoch_conf)
    if skipped:
        print(f"  [train] skipped_nonfinite_batches={skipped}")
    return total_loss / max(1, n), miou6


@torch.no_grad()
def validate(model, loader, device, amp_helper: AmpHelper, *, dice_weight: float = 0.0):
    model.eval()
    total_loss, n = 0.0, 0
    epoch_conf = torch.zeros((DEEPGLOBE_NUM_CLASSES, DEEPGLOBE_NUM_CLASSES), dtype=torch.float64)
    for imgs, masks in tqdm(loader, desc="Valid", dynamic_ncols=True):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with amp_helper.autocast():
            logits = model(imgs)
            logits_up = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            loss = segmentation_loss(logits_up, masks, dice_weight=dice_weight)
        epoch_conf += confusion_matrix_torch(logits_up.argmax(1), masks).cpu()
        total_loss += float(loss.detach().item())
        n += 1
    miou6, per_class = miou_from_confusion(epoch_conf)
    return total_loss / max(1, n), miou6, per_class


def run_deepglobe_training(
    args,
    model: Optional[nn.Module],
    *,
    script_name: str,
    backbone_name: str,
) -> None:
    apply_earth_deepglobe_defaults(args, backbone_name=backbone_name)
    data_dir = Path(args.data_dir).expanduser().resolve()
    layout = resolve_deepglobe_layout(
        data_dir,
        val_ratio=float(getattr(args, "val_ratio", 0.15)),
        seed=int(getattr(args, "seed", 42)),
    )
    patch_size = int(getattr(args, "patch_size", getattr(args, "img_size", 512)))
    samples_per_tile = int(getattr(args, "samples_per_tile", 8))
    color_jitter = float(getattr(args, "color_jitter", 0.05))
    dice_weight = float(getattr(args, "dice_weight", 0.0)) if is_earth_mamba(backbone_name) else 0.0
    args.dice_weight = dice_weight
    cache_tiles = bool(getattr(args, "cache_tiles", True))
    tile_cache = build_tile_cache(layout, enabled=cache_tiles)
    val_every = max(1, int(getattr(args, "val_every", 1)))

    train_ds = DeepGlobePatchDataset(
        layout,
        "train",
        patch_size,
        augment=True,
        samples_per_tile=samples_per_tile,
        color_jitter=color_jitter,
        rotate90=bool(getattr(args, "rotate90", False)),
        vflip=bool(getattr(args, "vflip", False)),
        tile_cache=tile_cache,
    )
    val_ds = DeepGlobePatchDataset(
        layout,
        "val",
        patch_size,
        augment=False,
        samples_per_tile=1,
        tile_cache=tile_cache,
    )

    print(
        f"DeepGlobe v3 train_tiles={len(layout.train_pairs)} val_tiles={len(layout.val_pairs)} "
        f"backbone={backbone_name} patch={patch_size} val_source={layout.val_source}",
        flush=True,
    )
    print(
        f"  [protocol] {PROTOCOL_VERSION} UPerNet fpn_dim={getattr(args, 'fpn_dim', 256)} "
        f"loss=CE{'+' + str(dice_weight) + '*Dice' if dice_weight > 0 else ''} "
        f"samples_per_tile={samples_per_tile} val=center_crop cache={cache_tiles} val_every={val_every}",
        flush=True,
    )
    if is_earth_mamba(backbone_name):
        adapter_lr = getattr(args, "adapter_lr", None) or args.lr
        print(
            f"  [{backbone_name}] adapter={getattr(args, 'dense_adapter', 'gated_pyramid')} "
            f"lr: encoder={args.encoder_lr:g} adapter={adapter_lr:g} decoder={args.lr:g} "
            f"dice={dice_weight:g} (mIoU-6 classes 0-5)",
            flush=True,
        )

    if getattr(args, "inspect_data", False):
        inspect_deepglobe_stats(train_ds)
        inspect_deepglobe_stats(val_ds)

    if args.dry_run:
        print("  dry_run complete (data/layout only; backbone not required).", flush=True)
        return

    if model is None:
        raise RuntimeError("model is required unless --dry_run is set")

    start = time.time()
    setup_perf()
    args.amp_helper = AmpHelper(args.amp_dtype)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = resolve_out_dir(
        args.output_dir,
        data_dir=str(data_dir),
        dataset_name=f"DeepGlobe_v3_{backbone_name.replace('-', '_')}",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        **kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        **kw,
    )

    model = model.to(device)
    model = maybe_compile(model, args.compile, device)
    print(f"device={device} output={out_dir} amp={args.amp_helper.amp_dtype}")

    optimizer = build_optimizer(model, args, backbone_name=backbone_name)
    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)
    start_epoch = 0
    best_miou6, best_epoch, history = 0.0, 0, []
    stale_epochs = 0
    patience = int(getattr(args, "early_stop_patience", 0))
    resume_path = getattr(args, "resume", None)
    last_val_miou6 = 0.0
    last_val_class_iou: list[Optional[float]] = [None] * SCORE_CLASSES

    if resume_path:
        completed, best_miou6, best_epoch, stale_epochs, history, resume_out = load_training_resume(
            resume_path,
            model,
            optimizer,
            scheduler,
            target_epochs=args.epochs,
        )
        if args.output_dir:
            out_dir = Path(args.output_dir).expanduser().resolve()
        else:
            out_dir = resume_out
        out_dir.mkdir(parents=True, exist_ok=True)
        start_epoch = completed
        if start_epoch >= args.epochs:
            print(f"  [resume] already completed {start_epoch} epochs (target {args.epochs})")
            return

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_miou6 = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        completed_epoch = epoch + 1
        do_val = val_every <= 1 or completed_epoch == args.epochs or completed_epoch % val_every == 0
        if do_val:
            val_loss, val_miou6, val_class_iou = validate(
                model,
                val_loader,
                device,
                args.amp_helper,
                dice_weight=dice_weight,
            )
            last_val_miou6 = val_miou6
            last_val_class_iou = list(val_class_iou)
        else:
            val_loss = 0.0
            val_miou6 = last_val_miou6
            val_class_iou = list(last_val_class_iou)
        scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_miou_6": train_miou6,
            "val_loss": val_loss,
            "val_miou_6": val_miou6,
            "val_class_iou": val_class_iou,
            "lr": optimizer.param_groups[0]["lr"],
        })
        class_line = ", ".join(
            f"{name}:{iou:.3f}" if iou is not None else f"{name}:NA"
            for name, iou in zip(CLASS_NAMES[:SCORE_CLASSES], val_class_iou[:SCORE_CLASSES])
        )
        if do_val:
            print(f"  train_mIoU-6={train_miou6:.4f} val_mIoU-6={val_miou6:.4f} best={best_miou6:.4f}")
            print(f"  val_class_iou={{ {class_line} }}")
            if val_miou6 > best_miou6:
                best_miou6, best_epoch = val_miou6, epoch + 1
                stale_epochs = 0
                torch.save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        scheduler,
                        epoch=epoch + 1,
                        best_miou=best_miou6,
                        best_epoch=best_epoch,
                        stale_epochs=stale_epochs,
                        history=history,
                        args=args,
                        backbone_name=backbone_name,
                        val_miou=val_miou6,
                        val_miou6=val_miou6,
                        val_class_iou=val_class_iou,
                        protocol=PROTOCOL_VERSION,
                    ),
                    out_dir / "best.pth",
                )
                print(f"  * saved {out_dir / 'best.pth'}")
            else:
                stale_epochs += 1
                if patience > 0 and stale_epochs >= patience:
                    print(f"  [early_stop] no val improvement for {patience} epochs; stop at epoch {epoch + 1}")
                    break
        else:
            print(
                f"  train_mIoU-6={train_miou6:.4f} (skip val; next at epoch "
                f"{completed_epoch + (val_every - completed_epoch % val_every)})",
                flush=True,
            )

        torch.save(
            checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch=epoch + 1,
                best_miou=best_miou6,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                history=history,
                args=args,
                backbone_name=backbone_name,
                val_miou=val_miou6,
                val_miou6=val_miou6,
                val_class_iou=val_class_iou,
                protocol=PROTOCOL_VERSION,
            ),
            out_dir / "last.pth",
        )

    elapsed = time.time() - start
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "protocol": PROTOCOL_VERSION,
            "args": args_to_dict(args),
            "backbone": backbone_name,
            "backbone_path": getattr(args, "ckpt", None) or "auto_or_random_init",
            "val_source": layout.val_source,
            "train_tiles": len(layout.train_pairs),
            "val_tiles": len(layout.val_pairs),
            "best_epoch": best_epoch,
            "best_val_miou_6": best_miou6,
            "class_names": list(CLASS_NAMES[:SCORE_CLASSES]),
            "history": history,
            "time_seconds": elapsed,
        }, f, indent=2, ensure_ascii=False)

    print_final_summary(
        dataset="DeepGlobe",
        task="semantic_seg_v3_upernet",
        script=script_name,
        ckpt=getattr(args, "ckpt", None),
        metrics={"val_mIoU_6": best_miou6},
        out_dir=out_dir,
        split=f"val({len(layout.val_pairs)} tiles, {layout.val_source})",
        epoch=best_epoch,
        extra_lines=[f"backbone={backbone_name}", f"elapsed={elapsed/60:.1f}min"],
    )


# Re-export helpers used by run scripts
__all__ = [
    "DEEPGLOBE_NUM_CLASSES",
    "PROTOCOL_VERSION",
    "DeepGlobePatchDataset",
    "UPerNetSeg",
    "add_perf_args",
    "add_warmup_args",
    "add_cache_args",
    "apply_earth_deepglobe_defaults",
    "resolve_deepglobe_layout",
    "run_deepglobe_training",
    "rgb_mask_to_index",
]
