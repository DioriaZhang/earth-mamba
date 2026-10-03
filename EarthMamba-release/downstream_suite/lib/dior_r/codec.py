from __future__ import annotations

import torch


def encode_quads(anchors: torch.Tensor, polygons: torch.Tensor) -> torch.Tensor:
    centers = (anchors[:, :2] + anchors[:, 2:]) * 0.5
    sizes = (anchors[:, 2:] - anchors[:, :2]).clamp_min(1.0)
    return ((polygons - centers[:, None]) / sizes[:, None]).flatten(1)


def decode_quads(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    centers = (anchors[:, :2] + anchors[:, 2:]) * 0.5
    sizes = (anchors[:, 2:] - anchors[:, :2]).clamp_min(1.0)
    return deltas.reshape(-1, 4, 2) * sizes[:, None] + centers[:, None]

