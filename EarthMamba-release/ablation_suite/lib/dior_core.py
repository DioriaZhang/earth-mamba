"""DIOR 水平框检测 — torchvision RetinaNet + CanonicalPyramid（无 MMDet）。

encoder 由 run_dior.py 经 backbone_registry.build_encoder 注入。
检测 loss 固定 FP32（fp16 AMP 下 RetinaNet focal loss 易 NaN）。
"""

from __future__ import annotations

import random
import warnings
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from tqdm import tqdm

from lib.common import AmpHelper

warnings.filterwarnings("ignore", category=UserWarning)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DIOR_CLASSES: Tuple[str, ...] = (
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge", "chimney",
    "dam", "Expressway-Service-area", "Expressway-toll-station", "golffield",
    "groundtrackfield", "harbor", "overpass", "ship", "stadium", "storagetank",
    "tenniscourt", "trainstation", "vehicle", "windmill",
)
CLASS_TO_ID: Dict[str, int] = {c: i + 1 for i, c in enumerate(DIOR_CLASSES)}

DEFAULT_FPN_STRIDES = (4, 8, 16, 32, 64)
FPN_OUT = 256


# ── 数据集路径解析 ─────────────────────────────────────────────────────────────

def resolve_ann_dir(data_dir: Path) -> Path:
    ann_root = data_dir / "Annotations"
    if not ann_root.is_dir():
        raise FileNotFoundError(f"未找到 Annotations: {ann_root}")
    if list(ann_root.glob("*.xml")):
        return ann_root
    for name in (
        "Horizontal Bounding Boxes", "Horizontal_Bounding_Boxes", "HorizontalBoundingBoxes",
    ):
        sub = ann_root / name
        if sub.is_dir() and list(sub.glob("*.xml")):
            return sub
    raise FileNotFoundError(f"{ann_root} 下无 *.xml 标注")


def resolve_trainval_img_dir(data_dir: Path) -> Path:
    for name in ("JPEGImages-trainval", "JPEGImages", "images"):
        p = data_dir / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"未找到 DIOR trainval 图片目录 (JPEGImages-trainval): {data_dir}")


def _has_trainval_image(img_dir: Path, stem: str) -> bool:
    for ext in (".jpg", ".JPG", ".jpeg", ".png"):
        if (img_dir / f"{stem}{ext}").is_file():
            return True
    return False


def labeled_stems_in_trainval(ann_dir: Path, img_dir: Path) -> List[str]:
    stems = []
    for xml in ann_dir.glob("*.xml"):
        if _has_trainval_image(img_dir, xml.stem):
            stems.append(xml.stem)
    if not stems:
        raise FileNotFoundError(f"Annotations 与 {img_dir} 无交集")
    return sorted(stems)


def load_split_ids(split_dir: Path, split: str) -> Optional[List[str]]:
    f = split_dir / f"{split}.txt"
    if not f.is_file():
        return None
    return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


def auto_split_ids(ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def find_image(img_dir: Path, stem: str) -> Path:
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".tif"):
        p = img_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"在 {img_dir} 找不到图片 {stem}")


def parse_voc_xml(path: Path) -> Tuple[str, List[List[float]], List[str], int, int]:
    tree = ET.parse(path)
    root = tree.getroot()
    stem = Path(root.findtext("filename", default=path.stem)).stem
    size = root.find("size")
    ow = int(size.findtext("width", "0")) if size is not None else 0
    oh = int(size.findtext("height", "0")) if size is not None else 0
    boxes, names = [], []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin", "0"))
        ymin = float(bb.findtext("ymin", "0"))
        xmax = float(bb.findtext("xmax", "0"))
        ymax = float(bb.findtext("ymax", "0"))
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        names.append(name)
    return stem, boxes, names, ow, oh


# ── 数据集 ────────────────────────────────────────────────────────────────────

class DIORDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        ann_dir: Path,
        ids: List[str],
        img_size: int = 512,
        augment: bool = True,
    ):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.ids = ids
        self.img_size = img_size
        self.augment = augment
        # 轻量颜色增强，仅训练时生效
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        xml_path = self.ann_dir / f"{stem}.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(f"缺少标注 {xml_path}")
        _, boxes, names, ow_xml, oh_xml = parse_voc_xml(xml_path)
        img_path = find_image(self.img_dir, stem)
        img = Image.open(img_path).convert("RGB")
        ow, oh = img.size
        if ow_xml and oh_xml and (ow_xml != ow or oh_xml != oh):
            sx, sy = ow / ow_xml, oh / oh_xml
            boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = (
            torch.tensor([CLASS_TO_ID.get(n, 0) for n in names], dtype=torch.int64)
            if names else torch.zeros((0,), dtype=torch.int64)
        )
        if labels.numel() > 0:
            keep = labels > 0
            boxes_t, labels = boxes_t[keep], labels[keep]

        if self.augment:
            # 水平翻转
            if random.random() < 0.5:
                img = TF.hflip(img)
                if boxes_t.numel() > 0:
                    x1, x2 = boxes_t[:, 0].clone(), boxes_t[:, 2].clone()
                    boxes_t[:, 0], boxes_t[:, 2] = ow - x2, ow - x1
            # 轻量颜色抖动
            img = self.color_jitter(img)

        img = TF.resize(img, (self.img_size, self.img_size))
        sx, sy = self.img_size / ow, self.img_size / oh
        if boxes_t.numel() > 0:
            boxes_t[:, [0, 2]] *= sx
            boxes_t[:, [1, 3]] *= sy
            boxes_t[:, 0::2] = boxes_t[:, 0::2].clamp(0, self.img_size)
            boxes_t[:, 1::2] = boxes_t[:, 1::2].clamp(0, self.img_size)
            wh = boxes_t[:, 2:] - boxes_t[:, :2]
            keep = (wh[:, 0] >= 2) & (wh[:, 1] >= 2)
            boxes_t, labels = boxes_t[keep], labels[keep]

        img_t = TF.normalize(TF.to_tensor(img), MEAN, STD)
        return img_t, {"boxes": boxes_t, "labels": labels, "image_id": torch.tensor([idx])}


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


# ── 通用检测金字塔 Backbone ───────────────────────────────────────────────────

def _resize_feature(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class CanonicalPyramidBackbone(nn.Module):
    """P2-P6 RetinaNet/FasterRCNN backbone for small-object DIOR detection."""

    def __init__(self, encoder: nn.Module, fpn_out: int = 256):
        super().__init__()
        enc_dims = encoder.out_dims
        self.body = encoder
        self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_out, 1) for c in enc_dims])
        self.output = nn.ModuleList([
            nn.Conv2d(fpn_out, fpn_out, 3, padding=1) for _ in DEFAULT_FPN_STRIDES
        ])
        self.out_channels = fpn_out
        self.strides = DEFAULT_FPN_STRIDES
        for conv in self.lateral:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        for conv in self.output:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        feats = self.body(x)
        proj = [self.lateral[i](feats[i]) for i in range(len(self.lateral))]
        for i in range(len(proj) - 2, -1, -1):
            proj[i] = proj[i] + _resize_feature(proj[i + 1], proj[i].shape[-2:])
        outs = OrderedDict()
        h, w = x.shape[-2:]
        for level, stride in enumerate(self.strides):
            target = (max(1, (h + stride - 1) // stride), max(1, (w + stride - 1) // stride))
            src_idx = min(range(len(proj)), key=lambda i: abs(proj[i].shape[-2] - target[0]))
            outs[str(level)] = self.output[level](_resize_feature(proj[src_idx], target))
        return outs


# ── torchvision 检测器 ────────────────────────────────────────────────────────

def _torchvision_targets(
    targets: List[Dict],
    device: torch.device,
    detector: str,
) -> List[Dict[str, torch.Tensor]]:
    """仅保留 boxes/labels；RetinaNet 用 0-based 类别。"""
    outs: List[Dict[str, torch.Tensor]] = []
    for t in targets:
        boxes = t["boxes"].to(device=device, dtype=torch.float32)
        labels = t["labels"].to(device=device, dtype=torch.int64)
        if detector == "retinanet":
            labels = (labels - 1).clamp(min=0)
        outs.append({"boxes": boxes, "labels": labels})
    return outs


class DetectorWrapper(nn.Module):
    """torchvision RetinaNet / Faster R-CNN 统一接口。"""

    def __init__(self, model: nn.Module, detector: str = "retinanet"):
        super().__init__()
        self.model = model
        self.detector = detector

    def forward_train(
        self,
        images: List[torch.Tensor],
        targets: List[Dict],
        device: torch.device,
    ) -> torch.Tensor:
        # RetinaNet focal loss 在 fp16 下易溢出 → 检测前向固定 FP32
        imgs = [im.float() for im in images]
        tgts = _torchvision_targets(targets, device, self.detector)
        with torch.cuda.amp.autocast(enabled=False):
            losses = self.model(imgs, tgts)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("detection loss is non-finite")
        return loss

    @torch.no_grad()
    def forward_eval(self, images: List[torch.Tensor]) -> List[Dict]:
        self.model.eval()
        imgs = [im.float() for im in images]
        with torch.cuda.amp.autocast(enabled=False):
            outs = self.model(imgs)
        self.model.train()
        outs_cpu = [{k: v.cpu() for k, v in o.items()} for o in outs]
        if self.detector == "retinanet":
            for o in outs_cpu:
                o["labels"] = o["labels"] + 1
        return outs_cpu


# ── 评估 ──────────────────────────────────────────────────────────────────────

def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter).clamp(min=1e-6)


def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map_voc(preds, gts, iou_thresh: float = 0.5, score_thresh: float = 0.05) -> float:
    """纯 Python VOC mAP@0.5（torchmetrics 不可用时降级使用）。"""
    aps = []
    for cls_id in range(1, len(DIOR_CLASSES) + 1):
        scores, tp_flags, n_gt = [], [], 0
        for pred, gt in zip(preds, gts):
            gt_boxes = gt["boxes"][gt["labels"] == cls_id]
            n_gt += gt_boxes.shape[0]
            p_mask = pred["labels"] == cls_id
            p_boxes, p_scores = pred["boxes"][p_mask], pred["scores"][p_mask]
            order = torch.argsort(p_scores, descending=True)
            p_boxes, p_scores = p_boxes[order], p_scores[order]
            matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
            for pb, ps in zip(p_boxes, p_scores):
                if ps < score_thresh:
                    continue
                scores.append(ps.item())
                if gt_boxes.numel() == 0:
                    tp_flags.append(0)
                    continue
                ious = _box_iou(pb.unsqueeze(0), gt_boxes).squeeze(0)
                best = int(ious.argmax())
                if ious[best] >= iou_thresh and not matched[best]:
                    tp_flags.append(1)
                    matched[best] = True
                else:
                    tp_flags.append(0)
        if n_gt == 0:
            continue
        if not scores:
            aps.append(0.0)
            continue
        order = np.argsort(-np.asarray(scores))
        tp = np.asarray(tp_flags)[order]
        fp = 1 - tp
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        rec = tp_cum / n_gt
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
        aps.append(_voc_ap(rec, prec))
    return float(np.mean(aps)) if aps else 0.0


@torch.no_grad()
def evaluate(
    wrapper: "DetectorWrapper",
    loader: DataLoader,
    device,
    score_thresh: float,
    amp: Optional[AmpHelper] = None,
) -> Dict[str, float]:
    wrapper.model.eval()
    preds, gts = [], []
    for images, targets in tqdm(loader, desc="Valid mAP", dynamic_ncols=True, leave=False):
        images = [im.to(device, non_blocking=True) for im in images]
        outs = wrapper.forward_eval(images)
        for out, gt in zip(outs, targets):
            keep = out["scores"] >= score_thresh
            preds.append({k: out[k][keep] for k in ("boxes", "scores", "labels")})
            gts.append({"boxes": gt["boxes"], "labels": gt["labels"]})
    wrapper.model.train()

    try:
        from torchmetrics.detection import MeanAveragePrecision
        m = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
        for pred, gt in zip(preds, gts):
            m.update([pred], [gt])
        stats = m.compute()
        metrics: Dict[str, float] = {"mAP@0.5": float(stats.get("map_50", 0))}
        if "map" in stats:
            metrics["mAP@0.5:0.95"] = float(stats["map"])
        return metrics
    except Exception:
        return {"mAP@0.5": evaluate_map_voc(preds, gts, score_thresh=score_thresh)}


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train_one_epoch(wrapper: "DetectorWrapper", loader: DataLoader, optimizer,
                    device, epoch: int, args, grad_accum: int = 1) -> float:
    wrapper.model.train()
    total, n, n_skip = 0.0, 0, 0
    last_step = len(loader) - 1
    optimizer.zero_grad(set_to_none=True)
    nan_warned = False
    for step, (images, targets) in enumerate(
        tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)
    ):
        images = [im.to(device, non_blocking=True) for im in images]
        try:
            loss = wrapper.forward_train(images, targets, device) / grad_accum
        except FloatingPointError:
            n_skip += 1
            if not nan_warned:
                warnings.warn("[DIOR] 跳过 non-finite loss batch（检测头 FP32 仍异常时请降 lr）")
                nan_warned = True
            optimizer.zero_grad(set_to_none=True)
            continue

        do_step = ((step + 1) % grad_accum == 0) or (step == last_step)
        loss.backward()
        if do_step:
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(wrapper.model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total += loss.item() * grad_accum
        n += 1
    if n_skip:
        print(f"  [warn] skipped {n_skip} non-finite batches")
    return total / max(1, n)


# ── 构建模型 ──────────────────────────────────────────────────────────────────

def build_faster_rcnn_torchvision(
    encoder: nn.Module,
    fpn_out: int,
    freeze_encoder: bool,
    img_size: int,
    nms_thresh: float,
    score_thresh: float,
) -> FasterRCNN:
    """torchvision Faster R-CNN（fallback）。"""
    backbone = CanonicalPyramidBackbone(encoder, fpn_out=fpn_out)
    if freeze_encoder:
        for p in backbone.body.parameters():
            p.requires_grad = False
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return FasterRCNN(
        backbone,
        num_classes=len(DIOR_CLASSES) + 1,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        box_roi_pool=MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2,
        ),
        min_size=img_size, max_size=img_size,
        box_nms_thresh=nms_thresh,
        box_score_thresh=score_thresh,
    )


def build_retinanet_torchvision(
    encoder: nn.Module,
    fpn_out: int,
    freeze_encoder: bool,
    img_size: int,
    nms_thresh: float,
    score_thresh: float,
):
    from torchvision.models.detection import RetinaNet
    backbone = CanonicalPyramidBackbone(encoder, fpn_out=fpn_out)
    if freeze_encoder:
        for p in backbone.body.parameters():
            p.requires_grad = False
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return RetinaNet(
        backbone,
        num_classes=len(DIOR_CLASSES),
        anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        min_size=img_size,
        max_size=img_size,
        detections_per_img=1000,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
    )


def build_detector(
    encoder: nn.Module,
    *,
    fpn_out: int = FPN_OUT,
    freeze_encoder: bool = False,
    detector: str = "retinanet",
    score_thresh: float = 0.05,
    img_size: int = 512,
    nms_thresh: float = 0.5,
) -> Tuple[nn.Module, DetectorWrapper]:
    """构建 torchvision 检测器（baselines2 仅 torchvision）。"""
    if detector == "retinanet":
        raw_model = build_retinanet_torchvision(
            encoder, fpn_out=fpn_out, freeze_encoder=freeze_encoder,
            img_size=img_size, nms_thresh=nms_thresh, score_thresh=score_thresh,
        )
    else:
        raw_model = build_faster_rcnn_torchvision(
            encoder, fpn_out=fpn_out, freeze_encoder=freeze_encoder,
            img_size=img_size, nms_thresh=nms_thresh, score_thresh=score_thresh,
        )
    print(f"  [DIOR] torchvision {detector} (FP32 loss)")
    return raw_model, DetectorWrapper(raw_model, detector)
