"""
pipeline_cluster_dataset.py — 供 pipeline_cluster_1_build / pipeline_cluster_2_filter
以及 pipeline_v2 共用的 DataLoader Dataset（必须独立模块，勿放在 __main__ 内）。

Windows 上若 Dataset 定义在脚本顶层 __main__，spawn 子进程 unpickle 会异常或极慢；
独立模块后 pickle 路径稳定，可被多进程 worker 正确加载。
"""
from __future__ import annotations

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# 大图 / 截断：确保 worker 进程内也生效
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageEmbedDataset(Dataset):
    """读路径返回张量；解码失败返回 None，由 collate_fn 过滤。"""

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        try:
            with Image.open(p) as raw:
                img = raw.convert("RGB")
            t = self.transform(img)
        except Exception:
            return idx, None
        return idx, t


def embed_collate(batch):
    idxs: list[int] = []
    tensors: list[torch.Tensor] = []
    for idx, t in batch:
        if t is None:
            continue
        idxs.append(int(idx))
        tensors.append(t)
    if not tensors:
        return None
    return idxs, torch.stack(tensors, dim=0)
