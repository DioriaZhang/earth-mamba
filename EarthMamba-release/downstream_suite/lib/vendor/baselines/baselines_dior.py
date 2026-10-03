from __future__ import annotations

import sys
from pathlib import Path

_BASELINES_ROOT = Path(__file__).resolve().parent
if str(_BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINES_ROOT))
from shared.bootstrap import setup_import_paths

setup_import_paths()

"""
DIOR 水平框目标检测 — 多骨干网络 FPN + Faster R-CNN（MMDetection / torchvision）

支持 backbone: earthmamba / skysense / satmae / dofa / clay / roma / rsmamba
骨干由 shared.backbone_registry.build_encoder 统一加载，
FPN 由 UniversalFPNBackbone 适配任意 out_dims。
优先使用 MMDetection（功能更强），不可用时自动 fallback torchvision。

数据布局（自动探测）::
  DIOR/
    Annotations/*.xml              (23,463, VOC 水平框)
    JPEGImages-trainval/*.jpg      (11,722, 有标注)
    JPEGImages-test/*.jpg          (11,741, 无公开标注 → 不参与 val)
  可选 ImageSets/Main/*.txt；若无则从 trainval∩标注 自动 85/15 划分

用法:
  python baselines_dior.py --data_dir /hy-tmp/task/DIOR --dry_run
  python baselines_dior.py --data_dir .../DIOR --backbone earthmamba \\
      --ckpt ep22.pth --img_size 512 --epochs 12

加速选项:
  --grad_accum N      梯度累积 N 步
  --no_mmdet          强制使用 torchvision
"""

import argparse
import json
import random
import time
import warnings
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter
from tqdm import tqdm

from downstream_common import (
    DEFAULT_OUTPUT_DIR, AmpHelper, add_eval_interval_args, add_output_args, add_perf_args,
    add_warmup_args, args_to_dict, build_warmup_cosine_scheduler, loader_kwargs,
    param_groups_weight_decay, resolve_out_dir, print_final_summary, setup_perf,
)

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

BACKBONE_CHOICES = ["earthmamba", "skysense", "satmae", "dofa", "clay", "roma", "rsmamba"]
DEFAULT_FPN_STRIDES = (4, 8, 16, 32, 64)


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


def _canonicalize_feature_sizes(
    feats: List[torch.Tensor],
    image_hw: Tuple[int, int],
    strides: Tuple[int, ...] = DEFAULT_FPN_STRIDES[:4],
) -> List[torch.Tensor]:
    """Resize arbitrary encoder outputs to a detector-friendly P2-P5 pyramid."""
    h, w = image_hw
    outs: List[torch.Tensor] = []
    for i, stride in enumerate(strides):
        src = feats[min(i, len(feats) - 1)]
        target = (max(1, (h + stride - 1) // stride), max(1, (w + stride - 1) // stride))
        outs.append(_resize_feature(src, target))
    return outs

class UniversalFPNBackbone(nn.Module):
    """将任意 encoder 适配为 FasterRCNN 所需的 backbone（带 FPN）。

    encoder 返回 4 个特征图的列表，encoder.out_dims = [C0, C1, C2, C3]。
    侧边投影将各级 Ci → fpn_out，再经 FPN 合并。
    """

    def __init__(self, encoder: nn.Module, fpn_out: int = 256):
        super().__init__()
        enc_dims = encoder.out_dims  # [C0, C1, C2, C3]
        self.body = encoder
        self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_out, 1) for c in enc_dims])
        self.fpn = FeaturePyramidNetwork([fpn_out] * 4, fpn_out)
        self.out_channels = fpn_out

    def forward(self, x: torch.Tensor) -> OrderedDict:
        feats = self.body(x)
        inner = {str(i): self.lateral[i](feats[i]) for i in range(4)}
        return OrderedDict(self.fpn(inner))


class CanonicalPyramidBackbone(nn.Module):
    """P2-P6 RetinaNet/FasterRCNN backbone for small-object DIOR detection."""

    def __init__(self, encoder: nn.Module, fpn_out: int = 256):
        super().__init__()
        enc_dims = encoder.out_dims
        self.body = encoder
        self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_out, 1) for c in enc_dims])
        self.output = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, 3, padding=1)
                                     for _ in DEFAULT_FPN_STRIDES])
        self.out_channels = fpn_out
        self.strides = DEFAULT_FPN_STRIDES

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


# ── MMDetection 支持 ──────────────────────────────────────────────────────────

def _try_build_mmdet(encoder: nn.Module, fpn_out: int,
                     freeze_encoder: bool, detector: str,
                     score_thresh: float) -> Optional[nn.Module]:
    """尝试用 MMDetection 构建 detector。失败则返回 None。"""
    try:
        from mmdet.utils import register_all_modules
        from mmdet.registry import MODELS
        register_all_modules(init_default_scope=False)
    except ImportError:
        return None

    enc_dims = list(encoder.out_dims)
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    class _UniversalMMDetBackbone(nn.Module):
        out_indices = tuple(range(len(enc_dims)))
        def __init__(self):
            super().__init__()
            self.body = encoder
        def forward(self, x):
            feats = self.body(x)
            return tuple(_canonicalize_feature_sizes(feats, x.shape[-2:]))
        def init_weights(self):
            pass

    _tag = f"_UniversalMMDet_{id(encoder)}"
    MODELS.register_module(name=_tag, module=_UniversalMMDetBackbone, force=True)

    if detector == "retinanet":
        model_cfg = dict(
            type="RetinaNet",
            backbone=dict(type=_tag),
            neck=dict(type="FPN", in_channels=enc_dims, out_channels=fpn_out,
                      start_level=0, num_outs=5, add_extra_convs="on_output"),
            bbox_head=dict(
                type="RetinaHead", num_classes=len(DIOR_CLASSES), in_channels=fpn_out,
                stacked_convs=4, feat_channels=fpn_out,
                anchor_generator=dict(
                    type="AnchorGenerator", octave_base_scale=4, scales_per_octave=3,
                    ratios=[0.5, 1.0, 2.0], strides=list(DEFAULT_FPN_STRIDES),
                ),
                bbox_coder=dict(type="DeltaXYWHBBoxCoder",
                                target_means=[.0, .0, .0, .0],
                                target_stds=[1.0, 1.0, 1.0, 1.0]),
                loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0,
                              alpha=0.25, loss_weight=1.0),
                loss_bbox=dict(type="L1Loss", loss_weight=1.0),
            ),
            train_cfg=dict(
                assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.5, neg_iou_thr=0.4,
                              min_pos_iou=0, ignore_iof_thr=-1),
                allowed_border=-1, pos_weight=-1, debug=False,
            ),
            test_cfg=dict(nms_pre=2000, min_bbox_size=0, score_thr=score_thresh,
                          nms=dict(type="nms", iou_threshold=0.5), max_per_img=1000),
        )
    else:
        from mmdet.models.detectors import FasterRCNN as MMFasterRCNN  # noqa: F401
        model_cfg = dict(
            type="FasterRCNN",
            backbone=dict(type=_tag),
            neck=dict(type="FPN", in_channels=enc_dims, out_channels=fpn_out, num_outs=5),
            rpn_head=dict(
                type="RPNHead", in_channels=fpn_out, feat_channels=fpn_out,
                anchor_generator=dict(
                    type="AnchorGenerator", scales=[8], ratios=[0.5, 1.0, 2.0],
                    strides=list(DEFAULT_FPN_STRIDES),
                ),
                bbox_coder=dict(type="DeltaXYWHBBoxCoder",
                                target_means=[.0,.0,.0,.0], target_stds=[1.,1.,1.,1.]),
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
                loss_bbox=dict(type="L1Loss", loss_weight=1.0),
            ),
            roi_head=dict(
                type="StandardRoIHead",
                bbox_roi_extractor=dict(
                    type="SingleRoIExtractor",
                    roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                    out_channels=fpn_out, featmap_strides=list(DEFAULT_FPN_STRIDES[:4]),
                ),
                bbox_head=dict(
                    type="Shared2FCBBoxHead", in_channels=fpn_out, fc_out_channels=1024,
                    roi_feat_size=7, num_classes=len(DIOR_CLASSES),
                    bbox_coder=dict(type="DeltaXYWHBBoxCoder",
                                    target_means=[.0,.0,.0,.0], target_stds=[.1,.1,.2,.2]),
                    reg_class_agnostic=False,
                    loss_cls=dict(type="CrossEntropyLoss", use_softmax=True, loss_weight=1.0),
                    loss_bbox=dict(type="L1Loss", loss_weight=1.0),
                ),
            ),
            train_cfg=dict(
                rpn=dict(
                    assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.3,
                                  min_pos_iou=0.3, match_low_quality=True),
                    sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5,
                                 neg_pos_ub=-1, add_gt_as_proposals=False),
                    allowed_border=-1, pos_weight=-1, debug=False,
                ),
                rpn_proposal=dict(nms_pre=2000, max_per_img=1000,
                                  nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
                rcnn=dict(
                    assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.5, neg_iou_thr=0.5,
                                  min_pos_iou=0.5, match_low_quality=False),
                    sampler=dict(type="RandomSampler", num=512, pos_fraction=0.25,
                                 neg_pos_ub=-1, add_gt_as_proposals=True),
                    pos_weight=-1, debug=False,
                ),
            ),
            test_cfg=dict(
                rpn=dict(nms_pre=1000, max_per_img=1000,
                         nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
                rcnn=dict(score_thr=score_thresh, nms=dict(type="nms", iou_threshold=0.5), max_per_img=300),
            ),
        )
    return MODELS.build(model_cfg)


class DetectorWrapper(nn.Module):
    """统一 mmdet / torchvision forward 接口。"""
    def __init__(self, model, backend: str, detector: str):
        super().__init__()
        self.model = model
        self.backend = backend
        self.detector = detector

    def forward_train(self, images, targets, device, img_size: int) -> torch.Tensor:
        if self.backend == "mmdet":
            from mmdet.structures import DetDataSample
            from mmengine.structures import InstanceData
            imgs_t = torch.stack(images)
            samples = []
            for t in targets:
                ds = DetDataSample()
                gi = InstanceData()
                gi.bboxes = t["boxes"].to(device)
                gi.labels = (t["labels"].to(device) - 1).clamp(min=0)
                ds.gt_instances = gi
                ds.metainfo = {"img_shape": (img_size, img_size),
                               "ori_shape": (img_size, img_size), "scale_factor": (1.0, 1.0)}
                samples.append(ds)
            return sum(self.model(imgs_t, samples, mode="loss").values())
        else:
            tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if self.detector == "retinanet":
                tgts = [{**t, "labels": (t["labels"] - 1).clamp(min=0)} for t in tgts]
            return sum(self.model(images, tgts).values())

    @torch.no_grad()
    def forward_eval(self, images, img_size: int) -> List[Dict]:
        if self.backend == "mmdet":
            from mmdet.structures import DetDataSample
            imgs_t = torch.stack(images)
            dummy = []
            for _ in images:
                ds = DetDataSample()
                ds.metainfo = {"img_shape": (img_size, img_size),
                               "ori_shape": (img_size, img_size)}
                dummy.append(ds)
            results = self.model(imgs_t, dummy, mode="predict")
            outs = []
            for r in results:
                pred = r.pred_instances
                outs.append({"boxes": pred.bboxes.cpu(),
                             "scores": pred.scores.cpu(),
                             "labels": pred.labels.cpu() + 1})
            return outs
        else:
            self.model.eval()
            outs = self.model(images)
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
    img_size: int,
    amp: Optional[AmpHelper] = None,
) -> Dict[str, float]:
    wrapper.model.eval()
    preds, gts = [], []
    for images, targets in tqdm(loader, desc="Valid mAP", dynamic_ncols=True, leave=False):
        images = [im.to(device, non_blocking=True) for im in images]
        if amp is not None and amp.enabled:
            with amp.autocast():
                outs = wrapper.forward_eval(images, img_size)
        else:
            outs = wrapper.forward_eval(images, img_size)
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


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train_one_epoch(wrapper: "DetectorWrapper", loader: DataLoader, optimizer,
                    device, epoch: int, args, grad_accum: int = 1) -> float:
    wrapper.model.train()
    amp: AmpHelper = args.amp_helper
    total, n = 0.0, 0
    last_step = len(loader) - 1
    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets) in enumerate(
        tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)
    ):
        images = [im.to(device, non_blocking=True) for im in images]
        with amp.autocast():
            loss = wrapper.forward_train(images, targets, device, args.img_size) / grad_accum

        do_step = ((step + 1) % grad_accum == 0) or (step == last_step)
        if amp.use_scaler:
            amp.scaler.scale(loss).backward()
            if do_step:
                if args.clip_grad > 0:
                    amp.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(wrapper.model.parameters(), args.clip_grad)
                amp.scaler.step(optimizer)
                amp.scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if do_step:
                if args.clip_grad > 0:
                    nn.utils.clip_grad_norm_(wrapper.model.parameters(), args.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total += loss.item() * grad_accum
        n += 1
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


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    p = argparse.ArgumentParser("DIOR 水平框检测（多骨干）")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", default=None, help="骨干预训练权重路径（优先于 backbone_ckpt_dir）")
    p.add_argument("--backbone", default="earthmamba", choices=BACKBONE_CHOICES,
                   help="骨干网络名称")
    p.add_argument("--ssm_version", default="mamba3",
                   help="仅 earthmamba 生效，SSM 版本（mamba2/mamba3）")
    p.add_argument("--backbone_ckpt_dir", default=None,
                   help="骨干权重目录，未指定 --ckpt 时自动找第一个 .pth")
    p.add_argument("--fpn_out", type=int, default=256, help="FPN 输出通道数")
    p.add_argument("--nms_thresh", type=float, default=0.5, help="NMS IoU 阈值")
    p.add_argument("--detector", choices=["retinanet", "fasterrcnn"], default="retinanet",
                   help="检测任务头：retinanet 更适合 DIOR 小目标；fasterrcnn 保留旧协议")
    add_output_args(p)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4, help="检测头学习率")
    p.add_argument("--encoder_lr", type=float, default=1e-5, help="encoder 学习率")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--score_thresh", type=float, default=0.05)
    p.add_argument("--val_ratio", type=float, default=0.15,
                   help="无 ImageSets 时从 trainval 自动划分 val 比例")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数")
    p.add_argument("--no_mmdet", action="store_true", help="强制使用 torchvision")
    add_perf_args(p)
    add_warmup_args(p, default_warmup=2)
    add_eval_interval_args(p, default=5)
    args = p.parse_args()
    setup_perf()
    args.amp_helper = AmpHelper(args.amp_dtype)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir).expanduser().resolve()
    ann_dir = resolve_ann_dir(data_dir)
    img_dir = resolve_trainval_img_dir(data_dir)
    all_labeled = labeled_stems_in_trainval(ann_dir, img_dir)
    split_dir = data_dir / "ImageSets" / "Main"
    train_ids = load_split_ids(split_dir, "train") if split_dir.is_dir() else None
    val_ids = load_split_ids(split_dir, "val") if split_dir.is_dir() else None
    split_source = "ImageSets/Main"
    if train_ids is None or val_ids is None:
        train_ids, val_ids = auto_split_ids(all_labeled, args.val_ratio, args.seed)
        split_source = f"auto {1-args.val_ratio:.0%}/{args.val_ratio:.0%} from trainval"
    else:
        train_ids = [s for s in train_ids if s in all_labeled]
        val_ids = [s for s in val_ids if s in all_labeled]
    test_img_dir = data_dir / "JPEGImages-test"
    n_test = len(list(test_img_dir.glob("*.jpg"))) if test_img_dir.is_dir() else 0
    print(f"DIOR: train={len(train_ids)} val={len(val_ids)} labeled_in_trainval={len(all_labeled)} backbone={args.backbone}")
    print(f"  划分: {split_source} | img={img_dir} | ann={ann_dir}")
    if n_test:
        print(f"  JPEGImages-test={n_test} (无标注，跳过)")

    if args.dry_run:
        for sid in train_ids[:3]:
            xml = ann_dir / f"{sid}.xml"
            stem, boxes, names, _, _ = parse_voc_xml(xml)
            img = find_image(img_dir, stem)
            print(f"  [{sid}] img={img.name} boxes={len(boxes)} classes={set(names)}")
        print("  dry_run 完成。")
        return

    out_dir = resolve_out_dir(args.output_dir, data_dir=args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _lk = loader_kwargs(args.num_workers, args.prefetch_factor, args.persistent_workers)
    train_loader = DataLoader(
        DIORDataset(img_dir, ann_dir, train_ids, args.img_size, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True, **_lk,
    )
    val_loader = DataLoader(
        DIORDataset(img_dir, ann_dir, val_ids, args.img_size, augment=False),
        batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True, **_lk,
    )

    # 通过 backbone_registry 统一加载 encoder
    from shared.backbone_registry import build_encoder
    ckpt_path = resolve_ckpt(args)
    encoder = build_encoder(args.backbone, args.img_size, ckpt_path, ssm_version=args.ssm_version)
    n_enc = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"  encoder={args.backbone} 参数量={n_enc:.1f}M out_dims={encoder.out_dims} ckpt={ckpt_path or '随机初始化'}")

    # ── 构建检测模型 ─────────────────────────────────────────────────────
    backend = "torchvision"
    raw_model = None
    if not args.no_mmdet:
        raw_model = _try_build_mmdet(
            encoder, args.fpn_out, args.freeze_encoder, args.detector, args.score_thresh
        )
        if raw_model is not None:
            backend = "mmdet"
            print(f"  ✅ 使用 MMDetection {args.detector}")
        else:
            print(f"  ⚠️  mmdet 不可用，回退至 torchvision {args.detector}")

    if raw_model is None:
        if args.detector == "retinanet":
            raw_model = build_retinanet_torchvision(
                encoder, fpn_out=args.fpn_out,
                freeze_encoder=args.freeze_encoder,
                img_size=args.img_size,
                nms_thresh=args.nms_thresh,
                score_thresh=args.score_thresh,
            )
        else:
            raw_model = build_faster_rcnn_torchvision(
                encoder, fpn_out=args.fpn_out,
                freeze_encoder=args.freeze_encoder,
                img_size=args.img_size,
                nms_thresh=args.nms_thresh,
                score_thresh=args.score_thresh,
            )

    raw_model = raw_model.to(device)
    wrapper = DetectorWrapper(raw_model, backend, args.detector)
    n_total = sum(p.numel() for p in raw_model.parameters()) / 1e6
    print(f"  总参数量: {n_total:.1f}M  backend={backend} fpn_out={args.fpn_out} "
          f"grad_accum={args.grad_accum} detector={args.detector}")
    print(f"  train={len(train_loader.dataset)} ({len(train_loader)} batches) "
          f"val={len(val_loader.dataset)} ({len(val_loader)} batches) "
          f"eval_interval={args.eval_interval}")

    # ── 优化器 ───────────────────────────────────────────────────────────
    if args.freeze_encoder:
        optimizer = optim.AdamW(
            [p for p in raw_model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        try:
            enc_body = raw_model.backbone.body if backend == "mmdet" else raw_model.backbone.body
            enc_ids = {id(p) for p in enc_body.parameters()}
            enc_groups = param_groups_weight_decay(enc_body, args.weight_decay)
            for g in enc_groups:
                g["lr"] = args.encoder_lr
            head_params = [p for p in raw_model.parameters()
                           if id(p) not in enc_ids and p.requires_grad]
        except AttributeError:
            enc_groups = []
            head_params = [p for p in raw_model.parameters() if p.requires_grad]
        head_groups = [
            {"params": [p for p in head_params if p.ndim > 1],
             "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": [p for p in head_params if p.ndim <= 1],
             "lr": args.lr, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(enc_groups + head_groups)
    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)

    best_map, best_epoch = 0.0, 0
    history = []

    for epoch in range(args.epochs):
        t_ep = time.time()
        loss = train_one_epoch(wrapper, train_loader, optimizer, device, epoch, args,
                               grad_accum=args.grad_accum)

        do_eval = (
            (epoch + 1) % args.eval_interval == 0
            or epoch + 1 == args.epochs
        )
        if do_eval:
            metrics = evaluate(wrapper, val_loader, device, args.score_thresh,
                               args.img_size, args.amp_helper)
            map50 = metrics["mAP@0.5"]
        else:
            metrics = {}
            map50 = float("nan")

        scheduler.step()
        ep_sec = time.time() - t_ep
        row = {
            "epoch": epoch + 1,
            "train_loss": loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(ep_sec, 1),
        }
        if do_eval:
            row.update(metrics)
        history.append(row)

        if do_eval:
            print(f"  ep{epoch+1} loss={loss:.4f} mAP@0.5={map50:.4f} best={best_map:.4f} "
                  f"({ep_sec/60:.1f}min)")
            if map50 > best_map:
                best_map, best_epoch = map50, epoch + 1
                torch.save(
                    {"epoch": best_epoch, "metrics": metrics, "model": raw_model.state_dict(),
                     "args": args_to_dict(args)},
                    out_dir / "best.pth",
                )
        else:
            print(f"  ep{epoch+1} loss={loss:.4f} (skip val) ({ep_sec/60:.1f}min)")

    elapsed = time.time() - t_start
    print(f"\n[DIOR] 总耗时: {elapsed/60:.1f} min ({elapsed:.0f}s) | "
          f"backbone={args.backbone} ckpt={ckpt_path or 'random_init'} backend={backend}")

    summary = {
        "args": args_to_dict(args),
        "split_source": split_source,
        "classes": list(DIOR_CLASSES),
        "best_mAP@0.5": best_map,
        "best_epoch": best_epoch,
        "backend": backend,
        "detector": args.detector,
        "history": history,
        "time_seconds": elapsed,
        "backbone": args.backbone,
        "backbone_path": ckpt_path or "random_init",
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_final_summary(
        dataset="DIOR", task="horizontal_detection", script="baselines_dior.py",
        ckpt=ckpt_path, metrics={"mAP@0.5": best_map}, out_dir=out_dir,
        split=f"val({len(val_ids)})", epoch=best_epoch,
        extra_lines=[
            f"backbone={args.backbone}",
            f"split_source={split_source}",
            f"backend={backend}",
            f"detector={args.detector}",
            f"elapsed={elapsed/60:.1f}min",
        ],
    )


if __name__ == "__main__":
    main()
