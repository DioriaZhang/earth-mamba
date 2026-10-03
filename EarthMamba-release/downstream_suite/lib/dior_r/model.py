from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from .geometry import order_polygon, polygon_area, rotated_nms


# Six FPN levels: P2–P5 from backbone + P6/P7 from strided convs.
# Including stride-4 (P2) significantly helps small objects (vehicle, ship, chimney).
STRIDES = (4, 8, 16, 32, 64, 128)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )


class FeaturePyramid(nn.Module):
    def __init__(self, in_channels, width: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in in_channels)
        self.output = nn.ModuleList(ConvBlock(width, width) for _ in in_channels)
        self.p6 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(width, width, 3, stride=2, padding=1)

    def forward(self, features, image_shape):
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise RuntimeError("backbone must return four NCHW feature maps")
        height, width = image_shape
        canonical = []
        for index, feature in enumerate(features):
            expected = (max(1, height // (4 * 2**index)), max(1, width // (4 * 2**index)))
            if feature.shape[-2:] != expected:
                feature = F.interpolate(feature, expected, mode="bilinear", align_corners=False)
            canonical.append(feature)
        values = [layer(value) for layer, value in zip(self.lateral, canonical)]
        # Top-down path: P5 → P4 → P3 → P2 (all four levels are now used)
        for index in range(2, -1, -1):
            values[index] = values[index] + F.interpolate(
                values[index + 1], values[index].shape[-2:], mode="nearest"
            )
        values = [layer(value) for layer, value in zip(self.output, values)]
        p2, p3, p4, p5 = values[0], values[1], values[2], values[3]
        p6 = self.p6(p5)
        p7 = self.p7(F.silu(p6))
        return [p2, p3, p4, p5, p6, p7]


class OrientedHead(nn.Module):
    def __init__(self, width: int, num_classes: int):
        super().__init__()
        # Four ConvBlocks give sufficient capacity; two was not enough for ~65+ mAP.
        self.tower = nn.Sequential(
            ConvBlock(width, width), ConvBlock(width, width),
            ConvBlock(width, width), ConvBlock(width, width),
        )
        self.classifier = nn.Conv2d(width, num_classes, 3, padding=1)
        self.regressor = nn.Conv2d(width, 8, 3, padding=1)
        nn.init.constant_(self.classifier.bias, -4.595)

    def forward(self, features):
        outputs = []
        for feature in features:
            value = self.tower(feature)
            outputs.append({"logits": self.classifier(value), "quads": self.regressor(value)})
        return outputs


class FastOrientedDetector(nn.Module):
    def __init__(self, encoder, channels, num_classes: int = 20, width: int = 256):
        super().__init__()
        self.encoder = encoder
        self.fpn = FeaturePyramid(channels, width)
        self.head = OrientedHead(width, num_classes)
        self.num_classes = num_classes

    def forward(self, images):
        features = self.encoder(images)
        pyramid = self.fpn(features, images.shape[-2:])
        return self.head(pyramid)


def _point_grid(height: int, width: int, stride: int, device, dtype):
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing="ij"
    )
    return torch.stack(((x + 0.5) * stride, (y + 0.5) * stride), dim=-1)


def _level_for_polygon(polygon: torch.Tensor) -> int:
    x_span = polygon[:, 0].max() - polygon[:, 0].min()
    y_span = polygon[:, 1].max() - polygon[:, 1].min()
    scale = float(torch.sqrt((x_span * y_span).clamp_min(1.0)))
    # Clamp to [0, len(STRIDES)-1] so all six levels can receive positive targets.
    return max(0, min(len(STRIDES) - 1, round(math.log2(scale / 64.0))))


def build_dense_targets(outputs, targets, num_classes: int):
    """Build per-level dense classification and regression targets.

    Targets are constructed on CPU to avoid the high per-element overhead of
    Python-index writes into CUDA tensors (kernel-launch latency × #objects ×
    #cells × #levels).  A single host→device copy is done at the end.
    """
    dense = []
    batch = outputs[0]["logits"].shape[0]
    device = outputs[0]["logits"].device
    dtype = outputs[0]["logits"].dtype

    for level, (output, stride) in enumerate(zip(outputs, STRIDES)):
        _, _, height, width = output["logits"].shape

        classes_t = torch.zeros((batch, num_classes, height, width), dtype=torch.float32)
        quads_t = torch.zeros((batch, 8, height, width), dtype=torch.float32)
        mask_t = torch.zeros((batch, height, width), dtype=torch.bool)
        owner_area = torch.full((batch, height, width), float("inf"), dtype=torch.float32)

        for batch_index, target in enumerate(targets):
            polygons = target["polygons"].detach().cpu().float()  # (N,4,2)
            labels_list = target["labels"].detach().cpu().tolist()
            difficult_raw = target.get(
                "difficult", torch.zeros(len(labels_list), dtype=torch.bool)
            )
            difficult_list = difficult_raw.detach().cpu().tolist()

            for polygon, label_int, is_difficult in zip(polygons, labels_list, difficult_list):
                if bool(is_difficult) or _level_for_polygon(polygon) != level:
                    continue

                poly = polygon.numpy()  # (4,2), avoids repeated tensor ops
                cx = float(poly[:, 0].mean())
                cy = float(poly[:, 1].mean())
                x_min = float(poly[:, 0].min())
                x_max = float(poly[:, 0].max())
                y_min = float(poly[:, 1].min())
                y_max = float(poly[:, 1].max())
                area = (x_max - x_min) * (y_max - y_min)

                center_gx = int(min(max(cx / stride, 0), width - 1))
                center_gy = int(min(max(cy / stride, 0), height - 1))
                poly_flat = [float(value) for value in poly.reshape(-1)]  # [x0,y0,x1,y1,x2,y2,x3,y3]

                for grid_y in range(max(0, center_gy - 1), min(height, center_gy + 2)):
                    for grid_x in range(max(0, center_gx - 1), min(width, center_gx + 2)):
                        px = (grid_x + 0.5) * stride
                        py = (grid_y + 0.5) * stride
                        if px < x_min or px > x_max or py < y_min or py > y_max:
                            continue
                        if area >= float(owner_area[batch_index, grid_y, grid_x]):
                            continue
                        owner_area[batch_index, grid_y, grid_x] = area
                        mask_t[batch_index, grid_y, grid_x] = True
                        classes_t[batch_index, :, grid_y, grid_x] = 0.0
                        classes_t[batch_index, label_int, grid_y, grid_x] = 1.0
                        quads_t[batch_index, :, grid_y, grid_x] = torch.tensor(
                            [
                                (poly_flat[0] - px) / stride,
                                (poly_flat[1] - py) / stride,
                                (poly_flat[2] - px) / stride,
                                (poly_flat[3] - py) / stride,
                                (poly_flat[4] - px) / stride,
                                (poly_flat[5] - py) / stride,
                                (poly_flat[6] - px) / stride,
                                (poly_flat[7] - py) / stride,
                            ],
                            dtype=torch.float32,
                        )

        dense.append({
            "classes": classes_t.to(device=device, dtype=dtype, non_blocking=True),
            "quads": quads_t.to(device=device, dtype=dtype, non_blocking=True),
            "mask": mask_t.to(device=device, non_blocking=True),
        })
    return dense


def _focal_loss(logits, targets, normalizer):
    probability = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = 0.25 * targets + 0.75 * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(2) * ce).sum() / normalizer


def detection_loss(outputs, targets, num_classes: int = 20):
    dense = build_dense_targets(outputs, targets, num_classes)
    positives = sum(item["mask"].sum() for item in dense).clamp_min(1)
    class_loss = sum(
        _focal_loss(output["logits"], target["classes"], positives)
        for output, target in zip(outputs, dense)
    )
    regression = outputs[0]["quads"].sum() * 0.0
    for output, target in zip(outputs, dense):
        expanded = target["mask"][:, None].expand_as(output["quads"])
        if bool(expanded.any()):
            regression = regression + F.smooth_l1_loss(
                output["quads"][expanded], target["quads"][expanded],
                reduction="sum",
                beta=1.0,  # beta=0.2 was too aggressive; 1.0 gives smoother gradients
            ) / positives
    return {"loss": class_loss + regression, "class_loss": class_loss.detach(),
            "quad_loss": regression.detach(), "positives": positives.detach()}


@torch.no_grad()
def decode_predictions(
    outputs, image_size: int, score_threshold: float = 0.05,
    nms_threshold: float = 0.1, topk: int = 1000, max_detections: int = 300,
):
    batch = outputs[0]["logits"].shape[0]
    per_image = [{"polygons": [], "scores": [], "labels": []} for _ in range(batch)]
    for output, stride in zip(outputs, STRIDES):
        # NMS kernels need float32; model may have run in bf16/fp16.
        scores = output["logits"].sigmoid().float()
        scores = scores * (scores == F.max_pool2d(scores, 3, stride=1, padding=1))
        b, classes, height, width = scores.shape
        count = min(topk, classes * height * width)
        values, indices = scores.flatten(1).topk(count, dim=1)
        labels = torch.div(indices, height * width, rounding_mode="floor")
        spatial = indices.remainder(height * width)
        ys = torch.div(spatial, width, rounding_mode="floor")
        xs = spatial.remainder(width)
        quad_map = output["quads"].float().flatten(2)
        gathered = quad_map.gather(2, spatial[:, None].expand(-1, 8, -1)).transpose(1, 2)
        points = torch.stack(((xs + 0.5) * stride, (ys + 0.5) * stride), dim=-1)
        polygons = gathered.reshape(b, count, 4, 2) * stride + points[:, :, None]
        polygons.clamp_(0, image_size)
        for index in range(batch):
            keep = values[index] >= score_threshold
            per_image[index]["polygons"].append(polygons[index][keep])
            per_image[index]["scores"].append(values[index][keep])
            per_image[index]["labels"].append(labels[index][keep])

    results = []
    for item in per_image:
        if not item["scores"] or sum(value.numel() for value in item["scores"]) == 0:
            results.append({"polygons": [], "scores": [], "labels": []})
            continue
        polygons = torch.cat(item["polygons"])
        scores = torch.cat(item["scores"])
        labels = torch.cat(item["labels"])
        bounds = torch.stack((
            polygons[..., 0].min(1).values, polygons[..., 1].min(1).values,
            polygons[..., 0].max(1).values, polygons[..., 1].max(1).values,
        ), dim=1)
        prekeep = batched_nms(bounds, scores, labels, 0.8)[: max_detections * 3]
        polygon_list = [order_polygon(value) for value in polygons[prekeep].cpu().tolist()]
        score_list = scores[prekeep].cpu().tolist()
        label_list = labels[prekeep].cpu().tolist()
        valid = [i for i, polygon in enumerate(polygon_list) if polygon_area(polygon) >= 4.0]
        polygon_list = [polygon_list[i] for i in valid]
        score_list = [score_list[i] for i in valid]
        label_list = [label_list[i] for i in valid]
        keep = rotated_nms(
            polygon_list, score_list, label_list, nms_threshold, max_detections
        )
        results.append({
            "polygons": [polygon_list[i] for i in keep],
            "scores": [score_list[i] for i in keep],
            "labels": [label_list[i] for i in keep],
        })
    return results
