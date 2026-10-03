"""
pretrain_mae.py  ─  EarthMamba MAE/SimMIM 预训练脚本（固定 512 版本）
=====================================================================

训练策略
─────────────────────────────────────────────────
  输入图像已预处理为 512×512；增强阶段默认 RandomResizedCrop
  （scale/ratio 可配，输出仍为 primary_size×primary_size），
  显存占用可预期。可用 --no_rrc 关闭 RRC，退回固定 512 + flip。

预训练方法：SimMIM（Masked Image Modeling）
─────────────────────────────────────────────────
  1. 在 patch 网格（32×32 @ patch_size=16）上随机采样掩码（mask_ratio=0.6）
  2. 被掩码的 patch 区域在像素空间替换为可学习掩码值
  3. 完整 EarthMamba 编码器处理含掩码的输入（所有 patch 均参与编码）
  4. 轻量解码头上采样到输入分辨率，预测原始像素值
  5. 损失：仅在被掩码 patch 的像素上计算归一化 L1

数据 txt 格式（每行 Tab 分隔）
─────────────────────────────────────────────────
  格式A（单文件）: <ID>\\t<图片路径>
    0000001\\t/data/rs/img1.jpg

  格式B（文件夹）: <ID>\\t<文件夹路径>
    A001\\t/data/rs/scene_001

  注释行以 # 开头自动跳过。

脚本位置
─────────────────────────────────────────────────
  推荐放在 earth-mamba 仓库的同级目录，例如：
    D:\\doc\\re\\main\\pretrain_mae.py
    D:\\doc\\re\\main\\earth-mamba\\   （仓库）
  也可仍放在 earth-mamba/ 内（兼容旧路径）。
  自定义仓库路径可设置环境变量 EARTH_MAMBA_ROOT。

使用示例
─────────────────────────────────────────────────
单卡调试：
  python pretrain_mae.py \\
      --data_list /data/pretrain_list.txt \\
      --output_dir ./pretrain_out \\
      --model_size small --patch_size 16 \\
      --batch_size 16 --epochs 800 --warmup_epochs 40 \\
      --amp_dtype fp16

多卡 DDP（8 卡）：
  torchrun --standalone --nproc_per_node=8 pretrain_mae.py \\
      --data_list /data/pretrain_list.txt \\
      --output_dir ./pretrain_out \\
      --model_size small --patch_size 16 \\
      --batch_size 16 --grad_accum 2 \\
      --epochs 800 --amp_dtype fp16

数据检查（不训练）：
  python pretrain_mae.py \\
      --data_list /data/list.txt \\
      --output_dir ./pretrain_out \\
      --dry_run

输出目录结构（每次运行自动生成 pretrain_YYYYmmdd_HHMM/）
──────────────────────────────────────────────────────────
  <output_dir>/pretrain_20260516_1200/
  │
  ├── logs/                       所有日志文件 + 训练结束全局曲线
  │   ├── launch_cmd.txt          原始 sys.argv + 所有参数（含有效 LR）
  │   ├── pretrain_log.json       训练结束汇总（args / train_log / 耗时等）
  │   ├── step_log.jsonl          每 --step_log_every 步追加（loss/lr/显存）
  │   ├── resource_log.jsonl      每 --step_log_every 步追加（GPU/CPU/RAM）
  │   ├── train_log_running.jsonl 每 epoch 追加均值 loss/lr（崩溃安全）
  │   ├── resource_summary.json   ResourceMonitor 后台采样摘要（训练结束后）
  │   ├── loss_curve.png          全量 step 级 loss（训练结束后生成）
  │   ├── lr_curve.png
  │   ├── epoch_loss_lr_curve.png 每 epoch 均值 loss + lr 双 Y 轴
  │   ├── resource_cpu_ram_curve.png
  │   └── resource_gpu_curve.png
  │
  ├── intermediate/               纯临时产物，不含关键训练结果
  │   ├── data_check/             --dry_run 数据检验图
  │   └── mplconfig/              matplotlib 临时配置目录
  │
  命名规则：ep{epoch(从1起)}.{本 epoch 内 optimizer 步数}
  （每个新 epoch 步数从 1 重新计数；只有真正的 optimizer.step 才算一步。）
  checkpoint 内另存 global_update_step（跨 epoch 累计）供恢复与排查。
  │
  ├── periodic/                   每 --periodic_every「本 epoch 内」optimizer 步
  │   ├── ep1.500/
  │   ├── ep1.1000/
  │   ├── ep2.500/
  │   └── ...每个子目录内：
  │       ├── checkpoint.pth      完整训练状态（可恢复）
  │       ├── backbone.pth        仅编码器权重（下游微调用）
  │       ├── recon.png           4列：原图/masked输入/重建融合/mask热力图
  │       ├── loss_running.png    训练开始至今全量 loss 曲线
  │       ├── loss_running_lr.png
  │       ├── loss_segment.png    仅本段（上次→本次 periodic）loss 曲线
  │       └── loss_segment_lr.png
  │
  └── epochs/                     每 --save_every / --viz_every epoch 结束时
      ├── ep1/
      │   ├── checkpoint.pth      完整训练状态（--save_every 触发）
      │   ├── backbone.pth        仅编码器权重（--save_every 触发）
      │   └── recon.png           重建对比图（--viz_every 触发）
      └── ep2/
          └── ...

微调时加载预训练骨干权重：
  backbone = BackboneEarthMamba(...)
  backbone.load_state_dict(torch.load('pretrain_out/.../epochs/ep10/backbone.pth'))
"""

from __future__ import annotations

import argparse
import builtins
import gc
import json
import logging
import math
import os
import io
import struct

# ══ 必须在所有可能触发 matplotlib 的 import 之前设置 ══
# main() 中用 output_dir/.tmp 覆盖兜底；命令行也应设 TMPDIR 防 torchelastic_
os.environ.setdefault("MPLBACKEND", "Agg")

import random
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
try:
    from torch.amp import GradScaler  # PyTorch >= 2.4 推荐
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore[no-redef]
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family"       : "DejaVu Sans",
        "axes.unicode_minus": False,
    })
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── CUDA 性能优化（训练吞吐）──────────────────────────────────────────────
try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False

try:
    import tarfile
    import json as _json
    HAS_TAR = True
except ImportError:
    HAS_TAR = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# 使用 high 精度 TF32 模式（Ampere+ 有效，V100 无影响但安全）
torch.set_float32_matmul_precision("high")

# ── NCCL 环境变量（多卡通信稳定性）──────────────────────────────────────────
# V100 PCIe  NVLink：P2P 不稳定，强制 SHM 通信
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
# NCCL_P2P_DISABLE 不设 1；A800 NVLink 需要 P2P 开启，NCCL 失败自动回退 SHM

# ── 定位 earth-mamba 仓库根并加入 sys.path ───────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_earth_mamba_root() -> Path:
    def _is_repo_root(p: Path) -> bool:
        return (p / "earth_mamba").is_dir() and (p / "setup.py").exists()

    env_root = os.environ.get("EARTH_MAMBA_ROOT", "").strip()
    if env_root:
        p = Path(env_root).resolve()
        if _is_repo_root(p):
            return p
        raise RuntimeError(
            f"EARTH_MAMBA_ROOT={p} 不是有效的 earth-mamba 仓库根（缺少 earth_mamba/ 或 setup.py）"
        )

    candidates = [
        _SCRIPT_DIR,
        _SCRIPT_DIR / "earth-mamba",
        _SCRIPT_DIR.parent / "earth-mamba",
        _SCRIPT_DIR.parent.parent / "earth-mamba",  # 仓库根 earth-mamba（两文件夹布局）
    ]
    for p in candidates:
        p = p.resolve()
        if _is_repo_root(p):
            return p

    raise RuntimeError(
        f"无法定位 earth-mamba 仓库根。脚本目录: {_SCRIPT_DIR}\n"
        "请将 pretrain_mae.py 放在 earth-mamba 同级（如 main/pretrain_mae.py），"
        "或设置环境变量 EARTH_MAMBA_ROOT 指向仓库根。\n"
        "也可在仓库内执行: pip install -e ./kernels/selective_scan && pip install -e ."
    )


EARTH_MAMBA_ROOT = _resolve_earth_mamba_root()
_repo_root_str = str(EARTH_MAMBA_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

# 不在模块导入阶段强制修改 EarthMamba 的 CUDA/Triton 后端。
# 这些环境变量由 main() 根据命令行参数设置，保持默认路径尽量接近原项目。

try:
    import selective_scan_cuda_oflex
    sys.modules["selective_scan_cuda"] = selective_scan_cuda_oflex
except ImportError:
    pass

try:
    from earth_mamba.models.earth_mamba import EarthMamba
    from earth_mamba.models.earth_mamba_block import EarthMambaBlock
except ImportError as e:
    raise ImportError(
        f"无法导入 earth_mamba（仓库根: {EARTH_MAMBA_ROOT}）。\n"
        "请在 earth-mamba 仓库根目录执行：\n"
        f"  cd {EARTH_MAMBA_ROOT}\n"
        "  pip install -e ./kernels/selective_scan\n"
        "  pip install -e .\n"
        "或设置环境变量 EARTH_MAMBA_ROOT 指向正确路径。"
    ) from e


# ══════════════════════════════════════════════════════════════════════════════
#  第一节：数据集与 Collator
# ══════════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_to_tensor = transforms.ToTensor()
_normalize  = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def _parse_float_pair(text: str, default: Tuple[float, float]) -> Tuple[float, float]:
    """解析 '0.6,1.0' → (0.6, 1.0)。"""
    try:
        parts = [float(x.strip()) for x in str(text).split(",")]
        if len(parts) != 2:
            raise ValueError
        lo, hi = parts[0], parts[1]
        if lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError
        return lo, hi
    except (ValueError, TypeError):
        return default


class TrainAugmentor:
    """
    在 DataLoader worker 内执行的增强（输出固定 primary×primary 归一化 Tensor）。

    默认开启 RandomResizedCrop（MAE 风格空间增强，无需 color_jitter）。
    """

    def __init__(
        self,
        primary: int = 512,
        use_rrc: bool = True,
        rrc_scale: Tuple[float, float] = (0.6, 1.0),
        rrc_ratio: Tuple[float, float] = (0.75, 1.3333333),
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        color_jitter: float = 0.0,
    ):
        self.primary = primary
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.use_rrc = use_rrc
        self._rrc = (
            transforms.RandomResizedCrop(
                primary,
                scale=rrc_scale,
                ratio=rrc_ratio,
                antialias=True,
            )
            if use_rrc else None
        )
        self._color_jitter: Optional[transforms.ColorJitter] = (
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter * 0.25,
            )
            if color_jitter > 0 else None
        )

    @staticmethod
    def _to_pil(img) -> Image.Image:
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                arr = np.clip(img, 0, 255).astype(np.uint8)
            else:
                arr = img
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[..., :3]
            return Image.fromarray(arr, mode="RGB")
        if isinstance(img, Image.Image):
            return PretrainDataset._safe_to_rgb(img)
        raise TypeError(f"不支持的图像类型: {type(img)}")

    def __call__(self, img) -> torch.Tensor:
        pil = self._to_pil(img)
        if self._rrc is not None:
            pil = self._rrc(pil)
        elif pil.width != self.primary or pil.height != self.primary:
            pil = pil.resize((self.primary, self.primary), Image.BILINEAR)
        t = _to_tensor(pil)
        if self._color_jitter is not None:
            t = self._color_jitter(t)
        if random.random() < self.hflip_prob:
            t = TF.hflip(t)
        if random.random() < self.vflip_prob:
            t = TF.vflip(t)
        return _normalize(t)


class AugmentedDataset(Dataset):
    """包装底层 Dataset，在 worker 内做增强并返回 (3,H,W) Tensor。"""

    def __init__(
        self,
        base: Dataset,
        augmentor: TrainAugmentor,
        min_sample_std: float = 3.0,
    ):
        self.base = base
        self.augmentor = augmentor
        self.min_sample_std = float(min_sample_std)

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def __getitem__(self, idx: int) -> torch.Tensor:
        n = len(self.base)
        for _ in range(5):
            try:
                sample = self.base[idx]
                if isinstance(sample, torch.Tensor):
                    return sample
                if (
                    self.min_sample_std > 0
                    and isinstance(sample, np.ndarray)
                    and sample.size > 0
                    and float(sample.std()) < self.min_sample_std
                ):
                    idx = random.randint(0, n - 1)
                    continue
                return self.augmentor(sample)
            except Exception:
                idx = random.randint(0, n - 1)
        return torch.zeros(3, self.augmentor.primary, self.augmentor.primary)


def build_train_augmentor(args: argparse.Namespace) -> TrainAugmentor:
    rrc_scale = _parse_float_pair(getattr(args, "rrc_scale", "0.6,1.0"), (0.6, 1.0))
    rrc_ratio = _parse_float_pair(getattr(args, "rrc_ratio", "0.75,1.33"), (0.75, 4.0 / 3.0))
    return TrainAugmentor(
        primary=args.primary_size,
        use_rrc=getattr(args, "rrc", True),
        rrc_scale=rrc_scale,
        rrc_ratio=rrc_ratio,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        color_jitter=args.color_jitter,
    )


class PretrainDataset(Dataset):
    """
    从 txt 索引文件加载图像数据集（所有图像已预处理为 primary_size × primary_size）。

    txt 支持两种格式（自动识别，可混用）：

    格式 A：每行第二列是**直接图片路径**（推荐，单文件一行）
        0000001\\t/data/rs/scene_001/img1.jpg

    格式 B：每行第二列是**文件夹路径**（枚举该文件夹下所有图片）
        A001\\t/data/rs/scene_001

    规则：
    - 若路径是存在的文件  → 直接加入样本池（格式 A）
    - 若路径是存在的文件夹 → 枚举其下所有图片（格式 B）
    - 两者均不存在        → 打印 warning 跳过，不崩溃
    - 跳过空行和 '#' 开头的注释行
    - rglob=True 时文件夹模式递归扫描子目录
    - 支持格式：jpg / jpeg / png / tif / tiff / webp / bmp
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

    def __init__(self, txt_file: str, primary_size: int = 512, rglob: bool = False):
        self.primary_size = primary_size
        self.paths: List[Path] = []
        self.ids:   List[str]  = []

        txt_path = Path(txt_file)
        if not txt_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {txt_file}")

        missing = []
        with open(txt_path, encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    sample_id  = str(lineno)
                    target_str = line
                else:
                    sample_id  = parts[0].strip()
                    target_str = parts[1].strip()

                target = Path(target_str)

                if target.is_file():
                    if target.suffix.lower() in self.EXTENSIONS:
                        self.paths.append(target)
                        self.ids.append(sample_id)
                    else:
                        missing.append((lineno, target, "后缀不支持"))

                elif target.is_dir():
                    scanner = target.rglob("*") if rglob else target.iterdir()
                    imgs = sorted(
                        p for p in scanner
                        if p.is_file() and p.suffix.lower() in self.EXTENSIONS
                    )
                    if not imgs:
                        missing.append((lineno, target, "文件夹下无图片"))
                    for img_path in imgs:
                        self.paths.append(img_path)
                        self.ids.append(sample_id)

                else:
                    missing.append((lineno, target, "路径不存在"))

        if missing:
            show = missing[:5]
            for lineno, p, reason in show:
                print(f"[PretrainDataset] warning: 第 {lineno} 行已跳过（{reason}）: {p}")
            if len(missing) > 5:
                print(f"  ... 共 {len(missing)} 条跳过（详见上方示例）")

        if len(self.paths) == 0:
            raise RuntimeError(
                f"从 {txt_file} 中未找到任何图像文件，请检查路径和格式。\n"
                "  支持两种 txt 格式：\n"
                "    格式A（单文件）: <ID>\\t<图片路径>\n"
                "    格式B（文件夹）: <ID>\\t<文件夹路径>"
            )

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _safe_to_rgb(img: Image.Image) -> Image.Image:
        """将任意 PIL Image 安全转换为 8-bit RGB。"""
        mode = img.mode
        if mode == "RGB":
            return img
        if mode in ("RGBA", "LA", "P", "L"):
            return img.convert("RGB")
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            lo, hi = float(arr.min()), float(arr.max())
            if hi - lo < 1e-6:
                arr = np.zeros_like(arr, dtype=np.uint8)
            else:
                arr = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = arr[..., :3]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        else:
            raise ValueError(f"不支持的图像形状: {arr.shape}")
        return Image.fromarray(arr, mode="RGB")

    def __getitem__(self, idx: int):
        """返回 numpy (H,W,3) uint8 或 PIL Image（cv2 不支持的格式回退 PIL）。"""
        for _ in range(5):
            try:
                if HAS_CV2:
                    bgr = cv2.imread(str(self.paths[idx]), cv2.IMREAD_COLOR)
                    if bgr is not None and bgr.size > 0:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        h, w = rgb.shape[:2]
                        if h != self.primary_size or w != self.primary_size:
                            rgb = cv2.resize(rgb, (self.primary_size, self.primary_size),
                                             interpolation=cv2.INTER_LINEAR)
                        return rgb  # numpy (H,W,3) uint8
                    # cv2 读不了的格式回退 PIL
                img = Image.open(self.paths[idx])
                img.load()
                img = self._safe_to_rgb(img)
                if img.width > 0 and img.height > 0:
                    return img
            except Exception:
                pass
            idx = random.randint(0, len(self.paths) - 1)
        return np.zeros((self.primary_size, self.primary_size, 3), dtype=np.uint8)


class LmdbPretrainDataset(Dataset):
    """LMDB 图片数据集。

    fork 安全：worker_init_fn 调 _reopen() 在子进程重建 env。
    """

    def __init__(self, lmdb_path: str, primary_size: int = 512):
        self.primary_size = primary_size
        self.lmdb_path = lmdb_path  # 保存路径，供 _reopen() 使用
        self._open_env()
        self._raw_mode = False
        self._txn = self.env.begin()
        self._txn_count = 0
        raw_len = self._txn.get(b"__total__")
        if raw_len is not None:
            self._len = int(raw_len.decode())
            self._key_fmt = "str"
        else:
            self._len = struct.unpack(">Q", self._txn.get(b"__len__"))[0]
            self._key_fmt = "bin"
        raw_ts = self._txn.get(b"__target_size__")
        if raw_ts is not None:
            self.primary_size = int(raw_ts.decode())
        if self._txn.get(b"__mode__") == b"raw_bytes":
            self._raw_mode = True

    def _open_env(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                             readahead=True, meminit=False, max_readers=128)

    def _reopen(self):
        """fork 后子进程重建 LMDB 环境（父进程 env 不安全）。"""
        try:
            self.env.close()
        except Exception:
            pass
        self._open_env()
        self._txn = self.env.begin()
        self._txn_count = 0

    def __len__(self) -> int:
        return self._len

    def _make_key(self, idx: int) -> bytes:
        if self._key_fmt == "str":
            return f"{idx:08d}".encode()
        return struct.pack(">Q", idx)

    def __getitem__(self, idx: int):
        """返回 numpy (H,W,3) uint8 或 PIL Image。"""
        self._txn_count += 1
        if self._txn_count % 10000 == 0:
            try:
                self._txn = self.env.begin()
            except Exception:
                self._reopen()
        for _ in range(5):
            try:
                raw = self._txn.get(self._make_key(idx))
                if raw is None:
                    idx = random.randint(0, self._len - 1)
                    continue
                if HAS_CV2:
                    buf = np.frombuffer(raw, dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if bgr is not None and bgr.size > 0:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        h, w = rgb.shape[:2]
                        if h != self.primary_size or w != self.primary_size:
                            rgb = cv2.resize(rgb, (self.primary_size, self.primary_size),
                                             interpolation=cv2.INTER_LINEAR)
                        # 内容完整性检查：丢弃近常数图像（全黑/全白/垃圾像素）
                        # 这些图通过 norm_pix_loss → target=0 → loss=0，污染训练
                        if rgb.std() < 3.0:
                            idx = random.randint(0, self._len - 1)
                            continue
                        return rgb
                img = Image.open(io.BytesIO(raw))
                img.load()
                return PretrainDataset._safe_to_rgb(img)
            except Exception:
                idx = random.randint(0, self._len - 1)
        return np.zeros((self.primary_size, self.primary_size, 3), dtype=np.uint8)

    def close(self):
        self.env.close()


class TarPretrainDataset(Dataset):
    """
    从 tar 分片目录流式读取图片（按偏移量直接读 tar 内 bytes，不解压到磁盘）。

    用法:
        --tar_dir /data/tar_shards/    # 目录含 shard_*.tar + tar_index.jsonl
    """

    def __init__(self, index_path: str, primary_size: int = 512):
        self.primary_size = primary_size
        self.tar_dir = Path(index_path).parent
        # 逐行读取 JSONL，不重读 tar
        with open(index_path, "r", encoding="utf-8") as f:
            self.entries = [json.loads(line) for line in f if line.strip()]

        # 缓存当前打开的 tar 文件句柄（每个 worker 独立，单句柄即可）
        self._tar_fh = None
        self._tar_path: Optional[str] = None

    def __len__(self):
        return len(self.entries)

    def _open_tar(self, path: Path):
        """按需打开 tar 文件，连续读取同一 tar 时复用句柄。"""
        p_str = str(path)
        if self._tar_path != p_str:
            if self._tar_fh is not None:
                self._tar_fh.close()
            self._tar_fh = open(path, "rb")
            self._tar_path = p_str
        return self._tar_fh

    def __getitem__(self, idx: int):
        for _ in range(5):
            try:
                entry = self.entries[idx]
                tar_path = self.tar_dir / entry["tar"]
                fh = self._open_tar(tar_path)
                fh.seek(entry["offset"])
                raw = fh.read(entry["size"])

                if HAS_CV2:
                    buf = np.frombuffer(raw, dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if bgr is not None and bgr.size > 0:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        h, w = rgb.shape[:2]
                        if h != self.primary_size or w != self.primary_size:
                            rgb = cv2.resize(rgb, (self.primary_size, self.primary_size),
                                             interpolation=cv2.INTER_LINEAR)
                        if rgb.std() < 3.0:
                            idx = random.randint(0, len(self) - 1)
                            continue
                        return rgb
                img = Image.open(io.BytesIO(raw))
                img.load()
                return PretrainDataset._safe_to_rgb(img)
            except Exception:
                idx = random.randint(0, len(self) - 1)
        return np.zeros((self.primary_size, self.primary_size, 3), dtype=np.uint8)


class FixedSizeCollator:
    """
    将 worker 返回的 (3,H,W) Tensor 堆叠为 batch。

    增强已在 AugmentedDataset / TrainAugmentor（worker 内）完成；
    此处仅 stack。若传入原始 numpy/PIL（如旧测试路径），仍走兼容分支。
    """

    def __init__(
        self,
        primary: int = 512,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        color_jitter: float = 0.0,
        augmentor: Optional[TrainAugmentor] = None,
    ):
        self.primary = primary
        self._fallback_count = 0
        self._legacy_augmentor = augmentor or TrainAugmentor(
            primary=primary,
            use_rrc=False,
            hflip_prob=hflip_prob,
            vflip_prob=vflip_prob,
            color_jitter=color_jitter,
        )

    def __call__(self, batch: list) -> torch.Tensor:
        if batch and isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, dim=0)
        tensors = [self._legacy_augmentor(img) for img in batch]
        return torch.stack(tensors, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
#  第二节：SimMIM 预训练模型封装
# ══════════════════════════════════════════════════════════════════════════════

class SimMIMDecoder(nn.Module):
    """
    轻量上采样解码头。

    输入：编码器最后阶段特征图 (B, C, Hf, Wf)
    输出：像素级预测图 (B, 3, H_in, W_in)

    优化：conv2（1×1 卷积）与 bilinear 上采样在正交维度上可交换，
    将 conv2 移到上采样前，使上采样在 3 通道上运行（原来 512 通道），
    计算量降 170 倍。
    """

    def __init__(self, encoder_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(encoder_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=hidden_dim)
        self.act1  = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, 3, kernel_size=1)

    def forward(self, feat: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        x = self.conv1(feat)
        x = self.norm1(x)
        x = self.act1(x)
        # 1×1 conv 前置于上采样：conv2(upsample(x)) = upsample(conv2(x))，通道 512→3，上采样快 170×
        x = self.conv2(x)
        # 渐进式 2× 上采样（仅 3 通道）
        _, _, h, w = x.shape
        while h < target_h or w < target_w:
            h2 = min(h * 2, target_h)
            w2 = min(w * 2, target_w)
            x = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
            h, w = h2, w2
        return x


class EarthMambaForPretraining(nn.Module):
    """
    SimMIM 风格的 EarthMamba 预训练封装。

    前向流程
    ────────
    1. 在 patch 网格（Hp × Wp）随机采样布尔掩码 M（mask_ratio=0.6 的位置为 True）
    2. 将输入图像中被掩码 patch 的像素替换为可学习掩码值 mask_value
    3. patch_embed 后加入 2D 可学习位置编码
    4. 完整编码器处理含掩码的输入（所有 patch 均参与编码，符合 SimMIM 设计）
    5. 解码器将特征图上采样到输入分辨率，输出像素级预测
    6. 仅在被掩码 patch 区域计算归一化 L1 损失
    """

    MODEL_CONFIGS = {
        "tiny":  {"depths": [2, 2,  9, 2], "dims": [ 96, 192,  384,  768]},
        "small": {"depths": [2, 2, 27, 2], "dims": [ 96, 192,  384,  768]},
        "base":  {"depths": [2, 2, 27, 2], "dims": [128, 256,  512, 1024]},
        "large": {"depths": [2, 2, 36, 2], "dims": [192, 384,  768, 1536]},
    }

    def __init__(
        self,
        model_size: str = "small",
        patch_size: int = 16,
        primary_size: int = 512,
        mask_ratio: float = 0.60,
        norm_pix_loss: bool = True,
        filter_low_var: bool = True,
        low_var_std_thresh: float = 0.05,
        decoder_hidden: int = 512,
        drop_path_rate: float = 0.1,
        ssm_version: str = "mamba1",
        ssm_headdim: int = 64,
        ssm_backend: Optional[str] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.patch_size    = patch_size
        self.primary_size  = primary_size
        self.mask_ratio    = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.filter_low_var = filter_low_var
        self.low_var_std_thresh = float(low_var_std_thresh)

        cfg = self.MODEL_CONFIGS[model_size]
        encoder_dim = cfg["dims"][-1]

        self.train_grid = primary_size // patch_size   # 512/16 = 32

        _ssm_d_state = 64 if ssm_version == "mamba3" else 16
        self.encoder = EarthMamba(
            depths=cfg["depths"],
            dims=cfg["dims"],
            patch_size=patch_size,
            in_chans=3,
            num_classes=1,
            ssm_d_state=_ssm_d_state,
            ssm_ratio=2.0,
            ssm_version=ssm_version,
            ssm_headdim=ssm_headdim,
            ssm_backend=ssm_backend,
            use_armg=True,
            use_graph=True,
            norm_layer="ln",
            posembed=True,
            imgsize=primary_size,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
            downsample_version="v3",
        )
        del self.encoder.classifier

        self.pos_embed = self.encoder.pos_embed
        self.encoder.pos_embed = None

        self.encoder_dim = encoder_dim
        self.mask_value = nn.Parameter(torch.zeros(1, 3, 1, 1))

        self.decoder = SimMIMDecoder(
            encoder_dim=encoder_dim,
            hidden_dim=decoder_hidden,
        )

        self.enc_norm = nn.GroupNorm(num_groups=32, num_channels=encoder_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_value, std=0.02)

    def _interpolate_pos_embed(self, Hp: int, Wp: int) -> torch.Tensor:
        pos_embed = self.pos_embed
        Hg, Wg = pos_embed.shape[2], pos_embed.shape[3]
        if Hp == Hg and Wp == Wg:
            return pos_embed
        return F.interpolate(
            pos_embed, size=(Hp, Wp),
            mode="bicubic", align_corners=False,
        )

    @torch.no_grad()
    def _generate_mask(
        self, B: int, Hp: int, Wp: int, device: torch.device
    ) -> torch.BoolTensor:
        N = Hp * Wp
        n_mask = max(1, int(N * self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        ids   = torch.argsort(noise, dim=1)
        mask  = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids[:, :n_mask], True)
        return mask.view(B, Hp, Wp)

    def _apply_mask_to_pixels(
        self, imgs: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        B, C, H, W = imgs.shape
        p  = self.patch_size
        Hp, Wp = H // p, W // p

        # 6D reshape：mask 保持 patch 级 (B,Hp,Wp)，通过 broadcasting 作用到 16×16 块
        # 无需 F.interpolate → 省 float32 中间张量 + float→bool 往返转换
        imgs_6d = imgs.reshape(B, C, Hp, p, Wp, p)                  # (B,C,Hp,p,Wp,p)
        mask_6d = mask[:, None, :, None, :, None]                    # (B,1,Hp,1,Wp,1)
        mv = self.mask_value.to(dtype=imgs.dtype).reshape(1, C, 1, 1, 1, 1)
        masked_6d = torch.where(mask_6d, mv, imgs_6d)
        return masked_6d.reshape(B, C, H, W)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.patch_embed(x)

        if self.pos_embed is not None:
            if self.encoder.channel_first:
                _, _, Hp, Wp = x.shape
            else:
                _, Hp, Wp, _ = x.shape
            pos = self._interpolate_pos_embed(Hp, Wp)
            if not self.encoder.channel_first:
                pos = pos.permute(0, 2, 3, 1)
            x = x + pos

        for layer in self.encoder.layers:
            x = layer(x)
        if not self.encoder.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.enc_norm(x)
        return x

    def forward(
        self, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        B, C, H, W = imgs.shape
        p  = self.patch_size
        Hp, Wp = H // p, W // p

        assert H % p == 0 and W % p == 0, (
            f"输入尺寸 ({H}, {W}) 不是 patch_size={p} 的整数倍"
        )

        mask        = self._generate_mask(B, Hp, Wp, imgs.device)
        masked_imgs = self._apply_mask_to_pixels(imgs, mask)
        feat        = self.forward_features(masked_imgs)
        pred        = self.decoder(feat, H, W)
        loss        = self._compute_loss(pred, imgs, mask)
        return loss, mask

    def _compute_loss(
        self,
        pred  : torch.Tensor,
        target: torch.Tensor,
        mask  : torch.BoolTensor,
    ) -> torch.Tensor:
        p  = self.patch_size
        B, C, H, W = pred.shape
        Hp, Wp = H // p, W // p

        def patchify(x):
            x = x.reshape(B, C, Hp, p, Wp, p)
            x = x.permute(0, 2, 4, 1, 3, 5)
            return x.reshape(B, Hp, Wp, C * p * p)

        pred_p   = patchify(pred)
        target_p = patchify(target)

        # 始终在原始 patch 像素上算 std；low_var 过滤与 norm_pix_loss 解耦
        var = target_p.var(dim=-1, keepdim=True, unbiased=False)
        std = (var + 1e-6).sqrt()
        low_var_mask = None
        if self.filter_low_var:
            low_var_mask = (std < self.low_var_std_thresh).squeeze(-1)

        if self.norm_pix_loss:
            mean = target_p.mean(dim=-1, keepdim=True)
            std_norm = std.clamp(min=1e-2)
            target_p = (target_p - mean) / std_norm

        loss_all = F.l1_loss(pred_p, target_p, reduction="none")
        loss_all = loss_all.mean(dim=-1)
        loss_all = loss_all.clamp(max=10.0)  # 单 patch 极端大 loss 不参与梯度
        if low_var_mask is not None:
            loss_all = loss_all.masked_fill(low_var_mask, 0.0)

        # 用 loss_all 的 dtype 转换 mask，避免 BF16 训练时 FP32 mask * BF16 loss 产生隐式提升
        mask_w = mask.to(dtype=loss_all.dtype)
        if low_var_mask is not None:
            mask_w = mask_w * (~low_var_mask).to(dtype=mask_w.dtype)
        loss = (loss_all * mask_w).sum() / (mask_w.sum() + 1e-6)
        return loss


# ══════════════════════════════════════════════════════════════════════════════
#  第三节：训练工具函数
# ══════════════════════════════════════════════════════════════════════════════

def build_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """线性 Warmup + Cosine 衰减，按 step 更新。"""
    total_steps  = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class AverageMeter:
    def __init__(self):  self.reset()
    def reset(self):     self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def is_master() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


_MEAN = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
_STD  = torch.tensor(_IMAGENET_STD ).view(3, 1, 1)


def _denorm(t: torch.Tensor) -> torch.Tensor:
    return (t.cpu().float() * _STD + _MEAN).clamp(0, 1)


@torch.no_grad()
def _dump_debug_batch(
    imgs: torch.Tensor,
    epoch: int,
    opt_step: int,
    loss_val: float,
    out_dir: Path,
):
    """保存 loss 异常 batch 的图片到 out_dir/debug_zero_loss/。"""
    dst = out_dir / "debug_zero_loss"
    dst.mkdir(parents=True, exist_ok=True)
    B = imgs.shape[0]
    denormed = _denorm(imgs)  # (B, 3, H, W), [0, 1]
    # 每张子图独立保存，方便肉眼检视
    for i in range(min(B, 8)):
        arr = (denormed[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        fname = dst / f"ep{epoch+1:03d}_step{opt_step:06d}_b{i}.png"
        Image.fromarray(arr).save(str(fname))
    # 也保存一张拼图 (2×4 或 1×B)
    n_cols = min(4, B)
    n_rows = (B + n_cols - 1) // n_cols
    canvas_h = n_rows * 512
    canvas_w = n_cols * 512
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for i in range(min(B, n_rows * n_cols)):
        r, c = i // n_cols, i % n_cols
        arr = (denormed[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        canvas[r*512:(r+1)*512, c*512:(c+1)*512] = arr
    grid_path = dst / f"ep{epoch+1:03d}_step{opt_step:06d}_grid.png"
    Image.fromarray(canvas).save(str(grid_path))
    return grid_path


@torch.no_grad()
def visualize_reconstruction(
    model    : nn.Module,
    loader   : DataLoader,
    device   : torch.device,
    args     : argparse.Namespace,
    out_dir  : Path,
    epoch    : int,
    n_samples: int = 4,
    save_path: Optional[Path] = None,
    suptitle : Optional[str] = None,
    show_mask_panel: bool = True,
    reference_batch: Optional[torch.Tensor] = None,  # 传入当前 batch 避免创建新迭代器
):
    """保存「原图 | 含掩码输入 | 重建融合 | (可选)patch 掩码」对比图。"""
    if not HAS_MATPLOTLIB:
        return

    model.eval()
    raw_model = model
    if isinstance(model, DDP):
        raw_model = model.module
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(model, FSDP):
            raw_model = model
    except ImportError:
        pass

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    imgs_orig: Optional[torch.Tensor] = None
    if reference_batch is not None:
        imgs_orig = reference_batch[:n_samples].to(device)
    else:
        for batch in loader:
            imgs_orig = batch[:n_samples].to(device)
            break
    if imgs_orig is None:
        return

    B, C, H, W = imgs_orig.shape
    p  = args.patch_size
    Hp, Wp = H // p, W // p

    _ac_enabled = args.use_amp and device.type == "cuda"
    _ac_dev_viz = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(_ac_dev_viz, enabled=_ac_enabled, dtype=amp_dtype):
        mask   = raw_model._generate_mask(B, Hp, Wp, device)
        masked = raw_model._apply_mask_to_pixels(imgs_orig, mask)
        feat   = raw_model.forward_features(masked)
        pred   = raw_model.decoder(feat, H, W)

    pred_f32 = pred.float()
    mask_up  = F.interpolate(
        mask.float().unsqueeze(1), size=(H, W), mode="nearest"
    )
    recon = imgs_orig.float().clone()
    recon[mask_up.bool().expand_as(recon)] = pred_f32[mask_up.bool().expand_as(pred_f32)]

    ncols = 4 if show_mask_panel else 3
    fig, axes = plt.subplots(B, ncols, figsize=(ncols * 2.8, B * 2.8))
    if B == 1:
        axes = axes[None]

    titles = ["Original", "Masked input", "MAE blend"]
    if show_mask_panel:
        titles.append("Mask (upsampled)")
    for i in range(B):
        col_tensors = [imgs_orig[i], masked[i], recon[i]]
        if show_mask_panel:
            # 灰度显示 patch 级掩码（白=被掩码区域）
            m2d = mask_up[i, 0].detach().cpu()
            m3 = m2d.unsqueeze(0).expand(3, -1, -1).clone()
            col_tensors.append(m3)
        for j, (t, title) in enumerate(zip(col_tensors, titles)):
            ax = axes[i][j]
            img_np = _denorm(t).permute(1, 2, 0).numpy() if j < 3 else t.permute(1, 2, 0).numpy()
            ax.imshow(img_np.clip(0, 1))
            ax.set_title(title if i == 0 else "")
            ax.axis("off")

    if suptitle is None:
        suptitle = f"Epoch {epoch + 1}"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    out_png = save_path if save_path is not None else (out_dir / "viz" / f"epoch_{epoch+1:04d}.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=150)
    plt.close()
    # 释放推理临时 tensor，让 empty_cache 真正有效
    del imgs_orig, mask, masked, feat, pred, pred_f32, mask_up, recon
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()
    model.train()


def save_running_curve_pngs(
    loss_history: List[float],
    lr_history  : List[float],
    out_png     : Path,
    title_prefix: str = "running",
    x_label     : str = "optimizer step (cumulative)",
) -> None:
    """将 loss / lr 序列写成 PNG 快照。

    Args:
        title_prefix: 图标题前缀，区分 'running'（全量）和 'segment'（本段）。
        x_label: x 轴标签。
    """
    if not HAS_MATPLOTLIB or not loss_history:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # 下采样：超过 50000 点时等距抽稀，避免 matplotlib 渲染 100 万点卡 CPU
    _max_pts = 50000
    _step = max(1, len(loss_history) // _max_pts)
    _idx = list(range(0, len(loss_history), _step))
    _loss = [loss_history[i] for i in _idx]
    _steps_x = _idx
    if len(lr_history) == len(loss_history):
        _lr = [lr_history[i] for i in _idx]
    else:
        _lr = []

    def _ema(vals: List[float], alpha: float = 0.05) -> List[float]:
        s = vals[0]
        out = [s]
        for v in vals[1:]:
            s = alpha * v + (1 - alpha) * s
            out.append(s)
        return out

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(_steps_x, _loss, color="steelblue", alpha=0.35, linewidth=0.8,
            label="loss (raw)")
    ax.plot(_steps_x, _ema(_loss), color="steelblue", linewidth=1.4,
            label="loss (EMA)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("SimMIM Loss")
    ax.set_title(f"Loss curve ({title_prefix})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(str(out_png), dpi=120)
    plt.close(fig)

    if _lr:
        fig2, ax2 = plt.subplots(figsize=(10, 2.8))
        ax2.plot(_steps_x, _lr, color="darkorange", linewidth=1.0)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("LR")
        ax2.set_title(f"Learning rate ({title_prefix})")
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        lr_path = out_png.with_name(out_png.stem.replace("loss", "lr") + ".png")
        plt.savefig(str(lr_path), dpi=120)
        plt.close(fig2)


@torch.no_grad()
def _check_batch(
    imgs      : torch.Tensor,
    batch_idx : int,
    label     : str,
    patch_size: int,
    primary   : int,
    errors    : list,
    logger    : logging.Logger,
    check_dir : Path,
):
    """对单个 batch tensor 做形状/值域/NaN 检查并保存可视化图。"""
    B, C, H, W = imgs.shape
    vmin, vmax = imgs.min().item(), imgs.max().item()

    if C != 3:
        errors.append(f"[{label}] 批次{batch_idx}: 通道数={C}，期望3")
    if H != primary or W != primary:
        errors.append(f"[{label}] 批次{batch_idx}: 尺寸({H},{W}) 不等于 primary_size={primary}")
    if H % patch_size != 0 or W % patch_size != 0:
        errors.append(f"[{label}] 批次{batch_idx}: ({H},{W}) 不是 patch_size={patch_size} 整数倍")
    if H != W:
        errors.append(f"[{label}] 批次{batch_idx}: H={H}≠W={W}（非正方形）")
    if torch.isnan(imgs).any():
        errors.append(f"[{label}] 批次{batch_idx}: 存在 NaN")
    if torch.isinf(imgs).any():
        errors.append(f"[{label}] 批次{batch_idx}: 存在 Inf")
    if vmin < -5 or vmax > 5:
        errors.append(f"[{label}] 批次{batch_idx}: 值域[{vmin:.2f},{vmax:.2f}] 疑似未归一化")

    logger.info(
        f"  [{label}] 批次{batch_idx:02d}: "
        f"shape={tuple(imgs.shape)}  val=[{vmin:.3f},{vmax:.3f}]  res={H}"
    )

    if HAS_MATPLOTLIB and batch_idx < 8:
        n_show = min(4, B)
        fig, axes = plt.subplots(n_show, 2, figsize=(8, 3 * n_show))
        if n_show == 1:
            axes = axes[None]
        for si in range(n_show):
            img_np = _denorm(imgs[si]).permute(1, 2, 0).numpy()
            axes[si][0].imshow(img_np)
            axes[si][0].set_title(f"[{label}] b{batch_idx} s{si} {H}x{W}")
            axes[si][0].axis("off")
            for ci, color in enumerate(["r", "g", "b"]):
                vals = imgs[si, ci].flatten().numpy()
                axes[si][1].hist(vals, bins=50, color=color, alpha=0.5, density=True)
            axes[si][1].set_title("Normalized pixel value dist.")
            axes[si][1].set_xlabel("value")
        fig.tight_layout()
        fname = check_dir / f"{label}_batch{batch_idx:02d}.png"
        plt.savefig(str(fname), dpi=100)
        plt.close()


def run_data_check(
    args      : argparse.Namespace,
    dataset   : Dataset,
    collator  : FixedSizeCollator,
    logger    : logging.Logger,
    out_dir   : Path,
    n_batches : int = 10,
):
    """
    数据检查模式（--dry_run）。

    随机采样 n_batches 个批次，检查：
      - 形状（B, 3, 512, 512）
      - 值域（归一化后应在 [-3, 3] 左右）
      - NaN / Inf
      - patch_size 整除性
      - 可视化保存到 intermediate/data_check/

    不运行模型前向，仅测数据管道。
    """
    check_dir = out_dir / "intermediate" / "data_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    patch_size = args.patch_size
    primary    = args.primary_size
    batch_size = min(4, args.batch_size)

    logger.info("=" * 60)
    logger.info("数据管道检查模式（--dry_run）")
    logger.info(f"  数据集: {len(dataset)} 张")
    logger.info(f"  主分辨率: {primary}px  patch_size: {patch_size}")
    logger.info(f"  检查批次数: {n_batches}  batch_size(检查用): {batch_size}")
    logger.info(f"  可视化输出: {check_dir}")
    logger.info("=" * 60)

    rng = random.Random(args.seed)
    total_imgs = len(dataset)

    for bi in range(n_batches):
        idxs = [rng.randint(0, total_imgs - 1) for _ in range(batch_size)]
        raw_imgs = [dataset[i] for i in idxs]
        try:
            batch = collator(raw_imgs)
            _check_batch(batch, bi, "primary", patch_size, primary, errors, logger, check_dir)
        except Exception as e:
            errors.append(f"[primary] 批次{bi} collator 异常: {e}")
            logger.error(f"  [primary] 批次{bi} 异常: {e}")

    logger.info("─" * 40)
    logger.info("检查汇总：")
    if errors:
        logger.error(f"共发现 {len(errors)} 个问题：")
        for e in errors:
            logger.error(f"  ✗ {e}")
    else:
        logger.info("✓ 所有批次检查通过，数据管道正常！")
    logger.info(f"  可视化图片已保存到: {check_dir}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CPU/GPU 资源监控（后台采样）
# ─────────────────────────────────────────────────────────────────────────────

class ResourceMonitor:
    """后台线程定时采样 CPU/内存/GPU 资源，训练结束后输出汇总。"""

    def __init__(self, device: torch.device, interval_sec: float = 1.0):
        self.device = device
        self.interval_sec = max(0.2, float(interval_sec))
        self.samples: list = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._ps_proc = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        self._nvml_ready = False
        self._gpu_count = 0

    def _init_nvml(self):
        if not HAS_PYNVML:
            return
        try:
            pynvml.nvmlInit()
            self._gpu_count = int(pynvml.nvmlDeviceGetCount())
            self._nvml_ready = self._gpu_count > 0
        except Exception:
            self._nvml_ready = False

    def _gpu_stats(self) -> dict:
        out: dict = {
            "gpu_util_mean_percent": None,
            "gpu_mem_used_percent_mean": None,
            "gpu_mem_used_mb_mean": None,
            "gpus": [],
            "torch_mem_alloc_mb": None,
            "torch_mem_reserved_mb": None,
        }
        if self.device.type == "cuda":
            idx = self.device.index if self.device.index is not None else 0
            try:
                out["torch_mem_alloc_mb"] = round(
                    torch.cuda.memory_allocated(idx) / 1024 ** 2, 2)
                out["torch_mem_reserved_mb"] = round(
                    torch.cuda.memory_reserved(idx) / 1024 ** 2, 2)
            except Exception:
                pass

        if not self._nvml_ready:
            if self.device.type == "cuda":
                _util_fn = getattr(torch.cuda, "utilization", None)
                if callable(_util_fn):
                    try:
                        _di = self.device.index if self.device.index is not None else 0
                        u = float(_util_fn(device=_di))
                        out["gpu_util_mean_percent"] = round(u, 2)
                        mem_total, mem_free = torch.cuda.mem_get_info(_di)
                        used = mem_total - mem_free
                        used_pct = (100.0 * used / mem_total) if mem_total else 0.0
                        out["gpu_mem_used_percent_mean"] = round(used_pct, 2)
                        out["gpu_mem_used_mb_mean"] = round(used / 1024 ** 2, 2)
                        out["gpus"] = [{"index": _di, "gpu_util_percent": u,
                                        "gpu_mem_used_mb": round(used / 1024 ** 2, 2),
                                        "gpu_mem_total_mb": round(mem_total / 1024 ** 2, 2),
                                        "gpu_mem_used_percent": round(used_pct, 2)}]
                    except Exception:
                        pass
            return out

        gpus = []
        try:
            for gi in range(self._gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gi)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_pct = (100.0 * mem.used / mem.total) if mem.total else 0.0
                gpus.append({
                    "index": gi,
                    "gpu_util_percent": float(util.gpu),
                    "gpu_mem_used_mb": round(mem.used / 1024 ** 2, 2),
                    "gpu_mem_total_mb": round(mem.total / 1024 ** 2, 2),
                    "gpu_mem_used_percent": round(used_pct, 2),
                })
        except Exception:
            return out
        out["gpus"] = gpus
        if gpus:
            out["gpu_util_mean_percent"] = round(
                sum(g["gpu_util_percent"] for g in gpus) / len(gpus), 2)
            out["gpu_mem_used_percent_mean"] = round(
                sum(g["gpu_mem_used_percent"] for g in gpus) / len(gpus), 2)
            out["gpu_mem_used_mb_mean"] = round(
                sum(g["gpu_mem_used_mb"] for g in gpus) / len(gpus), 2)
        return out

    def _cpu_stats(self) -> dict:
        out: dict = {"cpu_percent_mean": None, "cpu_percent_max_core": None,
                     "ram_percent": None}
        if not HAS_PSUTIL:
            return out
        try:
            percpu = psutil.cpu_percent(interval=None, percpu=True)
            if percpu:
                out["cpu_percent_mean"] = round(sum(percpu) / len(percpu), 2)
                out["cpu_percent_max_core"] = round(max(percpu), 2)
            out["ram_percent"] = round(psutil.virtual_memory().percent, 2)
        except Exception:
            pass
        return out

    def sample_once(self) -> dict:
        if self._start_time is None:
            self._start_time = time.time()
        sample = {"time_s": round(time.time() - self._start_time, 3),
                  "wall_time": time.strftime("%Y-%m-%d %H:%M:%S")}
        sample.update(self._cpu_stats())
        sample.update(self._gpu_stats())
        self.samples.append(sample)
        return sample

    def _loop(self):
        self.sample_once()
        while not self._stop.is_set():
            time.sleep(self.interval_sec)
            self.sample_once()

    def start(self):
        self._start_time = time.time()
        self._init_nvml()
        if HAS_PSUTIL:
            try:
                psutil.cpu_percent(interval=0.1, percpu=True)
            except Exception:
                pass
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml_ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_ready = False

    def summary(self) -> dict:
        if not self.samples:
            return {}
        out: dict = {"num_samples": len(self.samples), "gpu_count_nvml": self._gpu_count}

        def _stat(key: str):
            vals = [x[key] for x in self.samples if isinstance(x.get(key), (int, float))]
            if not vals:
                return
            out[f"{key}_avg"] = round(sum(vals) / len(vals), 3)
            out[f"{key}_max"] = round(max(vals), 3)

        for k in ("cpu_percent_mean", "ram_percent",
                  "gpu_util_mean_percent", "gpu_mem_used_percent_mean",
                  "torch_mem_alloc_mb", "torch_mem_reserved_mb"):
            _stat(k)
        return out

    @staticmethod
    def deps_hint():
        if not HAS_PSUTIL:
            print("[监控] 未安装 psutil，CPU/内存统计将为 None。pip install psutil")
        if not HAS_PYNVML:
            print("[监控] 未安装 pynvml，全卡 GPU 统计不可用。pip install nvidia-ml-py3")


def save_resource_curve_plots(output_dir, samples: list):
    if not HAS_MATPLOTLIB or not samples:
        return
    times = [s["time_s"] for s in samples]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.set_xlabel("time_s")
    ax1.set_ylabel("CPU % (mean cores)", color="tab:blue")
    ax2.set_ylabel("RAM %", color="tab:green")
    cpu_m = [s.get("cpu_percent_mean") for s in samples]
    ram   = [s.get("ram_percent") for s in samples]
    if any(isinstance(v, (int, float)) for v in cpu_m):
        ax1.plot(times, cpu_m, color="tab:blue", label="cpu_mean")
    if any(isinstance(v, (int, float)) for v in ram):
        ax2.plot(times, ram, color="tab:green", label="ram%")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    fig.suptitle("CPU & RAM vs time")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "resource_cpu_ram_curve.png"))
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.set_xlabel("time_s")
    ax1.set_ylabel("GPU util % (mean)", color="tab:orange")
    ax2.set_ylabel("GPU VRAM % (mean)", color="tab:red")
    gu = [s.get("gpu_util_mean_percent") for s in samples]
    gm = [s.get("gpu_mem_used_percent_mean") for s in samples]
    if any(isinstance(v, (int, float)) for v in gu):
        ax1.plot(times, gu, color="tab:orange")
    if any(isinstance(v, (int, float)) for v in gm):
        ax2.plot(times, gm, color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.suptitle("GPU util & VRAM % vs time")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "resource_gpu_curve.png"))
    plt.close()




def setup_ddp() -> Tuple[int, bool]:
    if "RANK" not in os.environ:
        return 0, False
    import datetime
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=5),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, True



def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_run_layout(
    base_output_dir: str,
    use_ddp: bool,
) -> Tuple[Path, Path, Path, Path, Path]:
    """创建本次运行的目录结构，返回 5 个 Path。

    Returns:
        out_dir          根目录 pretrain_YYYYmmdd_HHMM/
        intermediate_dir 纯临时产物  out_dir/intermediate/
        logs_dir         所有日志文件 + 全局曲线  out_dir/logs/
        periodic_dir     step 级快照  out_dir/periodic/
        epochs_dir       epoch 结束快照  out_dir/epochs/
    """
    base_dir = Path(base_output_dir).expanduser().resolve()
    run_name: Optional[str] = None

    if is_master():
        ts = time.strftime("%Y%m%d_%H%M")
        stem = f"pretrain_{ts}"
        run_dir = base_dir / stem
        suffix = 1
        while run_dir.exists():
            run_dir = base_dir / f"{stem}_{suffix:02d}"
            suffix += 1
        run_name = run_dir.name

    if use_ddp:
        shared = [run_name]
        dist.broadcast_object_list(shared, src=0)
        run_name = shared[0]

    if not run_name:
        raise RuntimeError("无法确定本次运行目录名（run_name 为空）。")

    out_dir          = (base_dir / run_name).resolve()
    intermediate_dir = out_dir / "intermediate"
    logs_dir         = out_dir / "logs"
    periodic_dir     = out_dir / "periodic"
    epochs_dir       = out_dir / "epochs"

    if is_master():
        for d in (intermediate_dir, logs_dir, periodic_dir, epochs_dir):
            d.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()

    return out_dir, intermediate_dir, logs_dir, periodic_dir, epochs_dir


def _get_model_state_dict(
    model: nn.Module,
    fsdp_used: bool,
    use_ddp: bool,
) -> dict:
    if fsdp_used:
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
                FullStateDictConfig,
            )
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                return model.state_dict()
        except Exception as e:
            raise RuntimeError(f"FSDP state_dict 提取失败: {e}") from e
    elif use_ddp:
        return model.module.state_dict()
    else:
        return model.state_dict()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    args: argparse.Namespace,
    fsdp_used: bool,
    use_ddp: bool,
    global_update_step: Optional[int] = None,
    epoch_local_opt_step: Optional[int] = None,
):
    sd = _get_model_state_dict(model, fsdp_used, use_ddp)
    if not is_master() or not sd:
        return
    # optimizer.state_dict() 直接取 GPU tensor 再 torch.save 会产生大量
    # GPU→CPU 小块传输，碎片化 CUDA 内存池。先整体移到 CPU 再保存。
    opt_sd = optimizer.state_dict()
    opt_sd_cpu = {
        'param_groups': opt_sd['param_groups'],
        'state': {},
    }
    for pid, st in opt_sd['state'].items():
        cpu_st = {}
        for k, v in st.items():
            cpu_st[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        opt_sd_cpu['state'][pid] = cpu_st

    ckpt: Dict[str, Any] = {
        "epoch"    : epoch,
        "model"    : sd,
        "optimizer": opt_sd_cpu,
        "scheduler": scheduler.state_dict(),
        "args"     : vars(args),
    }
    if global_update_step is not None:
        ckpt["global_update_step"] = int(global_update_step)
    if epoch_local_opt_step is not None:
        ckpt["epoch_local_opt_step"] = int(epoch_local_opt_step)
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)


def save_backbone_weights(
    path: Path,
    model: nn.Module,
    fsdp_used: bool,
    use_ddp: bool,
):
    full_sd = _get_model_state_dict(model, fsdp_used, use_ddp)
    if not is_master() or not full_sd:
        return
    sd = {}
    for k, v in full_sd.items():
        if k.startswith("encoder."):
            sd[k[len("encoder."):]] = v
        elif k == "pos_embed":
            sd["pos_embed"] = v
    torch.save(sd, path)


_psutil_cpu_warmed_up: bool = False


def _warmup_psutil() -> None:
    """psutil.cpu_percent(interval=None) 首次调用返回 0；
    需要先调用一次有 interval 的版本来建立基线。"""
    global _psutil_cpu_warmed_up
    if HAS_PSUTIL and not _psutil_cpu_warmed_up:
        try:
            psutil.cpu_percent(interval=0.05, percpu=True)
        except Exception:
            pass
        _psutil_cpu_warmed_up = True


def _sample_resource_now(device: torch.device) -> dict:
    """在 step 结束时同步采样 GPU 显存 + CPU/RAM，用于 per-step resource_log.jsonl。"""
    stats: dict = {"wall_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        try:
            stats["cuda_alloc_mb"]    = round(torch.cuda.memory_allocated(idx) / 1024 ** 2, 2)
            stats["cuda_reserved_mb"] = round(torch.cuda.memory_reserved(idx)  / 1024 ** 2, 2)
        except Exception:
            pass
    if HAS_PSUTIL:
        try:
            percpu = psutil.cpu_percent(interval=None, percpu=True)
            if percpu:
                stats["cpu_pct_mean"] = round(sum(percpu) / len(percpu), 1)
                stats["cpu_pct_max"]  = round(max(percpu), 1)
            stats["ram_pct"] = round(psutil.virtual_memory().percent, 1)
        except Exception:
            pass
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  第四节：训练主循环
# ══════════════════════════════════════════════════════════════════════════════

def _load_optimizer_state_chunked(
    optimizer: torch.optim.Optimizer,
    state_dict: dict,
    device: torch.device,
) -> None:
    """分块加载优化器状态到 GPU，避免一次性分配全部内存。

    PyTorch optimizer.state_dict()['state'] 的 key 是 id(param)（内存地址值），
    必须与 optimizer.state_dict()['param_groups'] 中记录的旧 id 做映射，
    才能找到新 optimizer 中对应的参数对象。

    正常的 load_state_dict() 一次性把所有 momentum/variance 从 CPU
    搬到 GPU，造成瞬时 ~700MB 显存峰值。这里逐参数搬运，峰值 ~50MB。
    """
    old_groups = state_dict['param_groups']
    old_state = state_dict.get('state', {})

    # 先加载 param_groups（不含 state），让 optimizer 知道参数结构
    optimizer.load_state_dict({
        'param_groups': old_groups,
        'state': {},
    })

    if not old_state:
        return

    # 构建旧 id → 新参数 的映射
    # old_groups[g]['params'] = [id_old_0, id_old_1, ...]
    # optimizer.param_groups[g]['params'] = [param_obj_0, param_obj_1, ...]
    id_map: dict = {}
    for old_group, new_group in zip(old_groups, optimizer.param_groups):
        for old_id, new_param in zip(old_group['params'], new_group['params']):
            id_map[old_id] = new_param

    # 逐参数搬运 state 到 GPU，每 20 个清理一次碎片
    loaded_count = 0
    for old_id, param_state in old_state.items():
        target_param = id_map.get(old_id)
        if target_param is None:
            continue

        gpu_state = {}
        for state_key, state_val in param_state.items():
            if isinstance(state_val, torch.Tensor):
                gpu_state[state_key] = state_val.to(device, non_blocking=True)
            else:
                gpu_state[state_key] = state_val
        optimizer.state[target_param] = gpu_state

        loaded_count += 1
        if loaded_count % 20 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()


def train_one_epoch(
    model            : nn.Module,
    loader           : DataLoader,
    optimizer        : torch.optim.Optimizer,
    scheduler,
    scaler           : Optional[GradScaler],
    epoch            : int,
    args             : argparse.Namespace,
    logger           : logging.Logger,
    device           : Optional[torch.device] = None,
    step_log_f       = None,
    resource_log_f   = None,
    loss_history     : Optional[list] = None,
    lr_history       : Optional[list] = None,
    fsdp_used        : bool = False,
    use_ddp          : bool = False,
    global_opt_step       : Optional[Dict[str, int]] = None,
    epoch_local_opt_step  : Optional[Dict[str, int]] = None,
    periodic_every        : int = 0,
    periodic_dir          : Optional[Path] = None,
    out_dir               : Optional[Path] = None,
    periodic_seg_start   : Optional[Dict[str, int]] = None,
    pending_optimizer_state : Optional[dict] = None,
    zero_loss_counter     : Optional[Dict[str, int]] = None,
    zero_loss_dump_dir    : Optional[Path] = None,
) -> Tuple[float, Optional[torch.Tensor]]:
    model.train()
    loss_meter     = AverageMeter()
    accum_steps    = max(1, args.grad_accum)
    amp_dtype      = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    step_log_every = max(1, getattr(args, "step_log_every", 10))
    log_every      = max(1, args.log_every)
    n_total        = len(loader)
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        loader, disable=not is_master(),
        desc=f"Epoch {epoch+1:03d}/{args.epochs}",
        dynamic_ncols=True,
    )

    last_batch_cpu: Optional[torch.Tensor] = None
    _grad_norm: Optional[float] = None
    _grad_norm_tensor: Optional[torch.Tensor] = None
    for step, imgs in enumerate(pbar):
        step_start = time.time()
        _t_data = time.time()  # 诊断：DataLoader 交付数据的时间点
        imgs = imgs.to(device, non_blocking=True) if device else imgs.cuda(non_blocking=True)

        # 保存最后 batch 的 CPU 拷贝（供 epoch 后可视化用，避免额外 DataLoader 迭代器）
        if step == n_total - 1:
            last_batch_cpu = imgs.cpu()

        is_last_step = (step + 1 == n_total)
        is_update    = ((step + 1) % accum_steps == 0) or is_last_step

        _no_sync_ctx = (
            model.no_sync()
            if isinstance(model, DDP) and not is_update
            else nullcontext()
        )

        with _no_sync_ctx:
            _ac_dev = "cuda" if (device is None or device.type == "cuda") else "cpu"
            _ac_on  = args.use_amp and _ac_dev == "cuda"
            with torch.amp.autocast(_ac_dev, enabled=_ac_on, dtype=amp_dtype):
                loss, mask = model(imgs)
                # 始终除以 accum_steps，保证每一步梯度权重相等；
                # epoch 末尾最后一个不足 accum_steps 的窗口会轻微低估，但影响可忽略
                loss = loss / accum_steps

            # NaN loss: 替换为零梯度 loss，保持 DDP 梯度同步
            # sum(p.sum() * 0.0) 对所有参数产生零梯度，参与 all-reduce 不污染
            if not torch.isfinite(loss):
                loss = sum(p.sum() * 0.0 for p in model.parameters())

            # ── 零 loss 异常检测 ──────────────────────────────────────────────
            # loss<1e-6 持续 N 步 → dump 该 batch + 置零梯度（比 continue 安全，
            # DDP 下 continue 会跳过 all-reduce 导致 NCCL 死锁）
            _zero_loss_thresh = getattr(args, "zero_loss_dump_threshold", 0)
            if _zero_loss_thresh > 0 and zero_loss_counter is not None:
                loss_per_sample = float(loss.detach().item()) * accum_steps
                if loss_per_sample < 1e-6:
                    zero_loss_counter["n"] = zero_loss_counter.get("n", 0) + 1
                    if zero_loss_counter["n"] == _zero_loss_thresh:
                        # 首次达到阈值 → dump 图像
                        _zp = zero_loss_dump_dir or out_dir
                        if _zp is not None and is_master():
                            _path = _dump_debug_batch(
                                imgs.cpu(), epoch, step, loss_per_sample, _zp,
                            )
                            logger.warning(
                                f"[zero-loss] loss={loss_per_sample:.2e} "
                                f"持续 {_zero_loss_thresh} 步, "
                                f"已 dump → {_path}"
                            )
                    if zero_loss_counter["n"] >= _zero_loss_thresh:
                        # 达到阈值后持续置零梯度（阻止更新）
                        loss = loss.detach() * 0.0
                else:
                    zero_loss_counter["n"] = 0

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if is_update:
            if args.clip_grad > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                _gn = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                # 保留为 GPU tensor，避免在 optimizer.step() 前插入 CPU-GPU 同步点
                # 仅在需要记录时才调用 .item()（每 step_log_every 步，见下方日志块）
                _grad_norm_tensor = _gn
            # 延迟加载优化器状态：此时 backward 已完成，激活已释放，显存充足
            if pending_optimizer_state is not None:
                _load_optimizer_state_chunked(optimizer, pending_optimizer_state,
                                              device or torch.device("cuda", 0))
                pending_optimizer_state = None
                if device is not None and device.type == "cuda":
                    torch.cuda.empty_cache()
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            _grad_norm = None  # 允许下一个 log step 重新从 _grad_norm_tensor 读取

            # ── 优化器步计数 + 周期性存盘 / 可视化 ───────────────────────
            gu = None
            elu = None
            if global_opt_step is not None:
                global_opt_step["n"] = global_opt_step.get("n", 0) + 1
                gu = global_opt_step["n"]
            if epoch_local_opt_step is not None:
                epoch_local_opt_step["n"] = epoch_local_opt_step.get("n", 0) + 1
                elu = epoch_local_opt_step["n"]
            # 定期回收 CPU 内存 + CUDA 碎片整理（每 500 opt 步）
            if gu is not None and gu > 0 and gu % 500 == 0:
                gc.collect()
                if device is not None and device.type == "cuda":
                    torch.cuda.empty_cache()

            ep1 = epoch + 1  # 对外目录/文件名：第几个 epoch（从 1 起）
            if (
                periodic_every > 0
                and elu is not None
                and elu % periodic_every == 0
                and periodic_dir is not None
                and out_dir is not None
            ):
                if is_master():
                    per_dir = periodic_dir / f"ep{ep1}.{elu}"
                    per_dir.mkdir(parents=True, exist_ok=True)
                    _gtxt = f"  global_opt_step={gu}" if gu is not None else ""
                    logger.info(
                        f"[periodic] epoch={ep1}  epoch_opt_step={elu}{_gtxt}  "
                        f"→ {per_dir}"
                    )

                    # periodic 只存曲线（不存 ckpt，归 epoch 级），纯 CPU 操作无 GPU 干扰
                    if loss_history and lr_history:
                        save_running_curve_pngs(
                            loss_history, lr_history,
                            per_dir / "loss_running.png",
                            title_prefix="running (all steps so far)",
                        )
                        seg_start = 0
                        if periodic_seg_start is not None:
                            seg_start = periodic_seg_start.get("n", 0)
                        seg_loss = loss_history[seg_start:]
                        seg_lr   = lr_history[seg_start:]
                        if seg_loss:
                            save_running_curve_pngs(
                                seg_loss, seg_lr,
                                per_dir / "loss_segment.png",
                                title_prefix="segment (this interval only)",
                                x_label="step within segment",
                            )
                    if periodic_seg_start is not None:
                        periodic_seg_start["n"] = len(loss_history)

                    # periodic checkpoint 时清理 CUDA 碎片，防 NCCL 通信降速
                    if device is not None and device.type == "cuda":
                        torch.cuda.empty_cache()

        # 每步都取 loss 值，保持 loss_history 密度（x 轴与真实步对齐）
        _t_compute = time.time()  # 诊断：GPU 计算完成的时间点
        loss_val = loss.detach().item() * accum_steps

        # 多卡：仅在 log_every 步做一次 all_reduce 取均值，避免每步通信
        if dist.is_initialized() and step % log_every == 0:
            lt = torch.tensor(loss_val, device=imgs.device)
            dist.all_reduce(lt, op=dist.ReduceOp.AVG)
            loss_val = lt.item()

        loss_meter.update(loss_val)
        lr_now = optimizer.param_groups[0]["lr"]

        if loss_history is not None:
            loss_history.append(loss_val)
        if lr_history is not None:
            lr_history.append(lr_now)

        if is_master():
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{lr_now:.2e}"})

            if step % step_log_every == 0:
                _dev = device or torch.device("cuda", 0)
                _gstep = global_opt_step.get("n", 0) if global_opt_step is not None else 0
                _estep = (
                    epoch_local_opt_step.get("n", 0)
                    if epoch_local_opt_step is not None
                    else 0
                )

                _masked_pct = round(mask.float().mean().item() * 100, 2)

                # 延迟到此处才 .item()，消除 optimizer.step() 前的 GPU-CPU 同步气泡
                if _grad_norm_tensor is not None and _grad_norm is None:
                    _grad_norm = float(_grad_norm_tensor.item())

                if step_log_f is not None:
                    row: dict = {
                        "epoch"            : epoch + 1,
                        "step_in_epoch"    : step,
                        "epoch_opt_step"   : _estep,
                        "global_opt_step"  : _gstep,
                        "loss"             : round(loss_val, 6),
                        "lr"               : lr_now,
                        "grad_norm"        : round(_grad_norm, 4) if _grad_norm is not None else None,
                        "res"              : imgs.shape[-1],
                        "masked_pct"       : _masked_pct,
                        "data_wait_s"      : round(_t_data - step_start, 4),
                        "compute_s"        : round(_t_compute - _t_data, 4),
                        "step_time_s"      : round(time.time() - step_start, 4),
                    }
                    if _dev.type == "cuda":
                        _di = _dev.index or 0
                        try:
                            row["cuda_alloc_mb"]    = round(
                                torch.cuda.memory_allocated(_di) / 1024 ** 2, 2)
                            row["cuda_reserved_mb"] = round(
                                torch.cuda.memory_reserved(_di)  / 1024 ** 2, 2)
                        except Exception:
                            pass
                    step_log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    step_log_f.flush()

                # resource_log.jsonl：GPU 显存 + CPU/RAM，追加写入（崩溃安全）
                if resource_log_f is not None:
                    res_row: dict = {
                        "epoch"           : epoch + 1,
                        "step_in_epoch"   : step,
                        "epoch_opt_step"  : _estep,
                        "global_opt_step" : _gstep,
                    }
                    res_row.update(_sample_resource_now(_dev))
                    resource_log_f.write(json.dumps(res_row, ensure_ascii=False) + "\n")
                    resource_log_f.flush()

            if step % log_every == 0:
                _gtxt = ""
                if global_opt_step is not None:
                    _gtxt = f"  gstep={global_opt_step.get('n', 0)}"
                if epoch_local_opt_step is not None:
                    _gtxt += f"  estep={epoch_local_opt_step.get('n', 0)}"
                # 若 step_log_every 已计算过则复用；否则按需算一次
                _mask_pct = (
                    _masked_pct if step % step_log_every == 0
                    else round(mask.float().mean().item() * 100, 1)
                )
                logger.info(
                    f"Ep[{epoch+1:03d}/{args.epochs}] "
                    f"Step[{step:05d}/{n_total}] "
                    f"loss={loss_meter.avg:.4f}  "
                    f"lr={lr_now:.2e}  "
                    f"res={imgs.shape[-1]}  "
                    f"masked={_mask_pct}%"
                    f"{_gtxt}"
                )

    return loss_meter.avg, last_batch_cpu


# ══════════════════════════════════════════════════════════════════════════════
#  第五节：命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "EarthMamba SimMIM 预训练（固定 512 版本）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 数据 ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("数据")
    g.add_argument("--data_list",   default="",
                   help="512px 图片索引 txt，每行格式: <ID>\\t<图片文件路径 或 文件夹路径>")
    g.add_argument("--lmdb",        action="store_true", default=False,
                   help="data_list 指向 LMDB 目录（需先运行 build_lmdb.py 生成）")
    g.add_argument("--tar_dir",     default=None,
                   help="tar 分片目录路径（含 shard_*.tar + tar_index.json，无需解压）")
    g.add_argument("--rglob",       action="store_true", default=False,
                   help="递归扫描 data_list 文件夹子目录（默认只扫顶层）")
    g.add_argument("--output_dir",  default="./pretrain_output")
    g.add_argument("--flat_output", action="store_true", default=False,
                   help="直接写入 output_dir（关闭自动 pretrain_时间戳 子目录；不推荐）")
    g.add_argument("--flat_ckpt", action="store_true", default=False,
                   help="epoch 级 ckpt 平铺在 epochs/ 下，命名为 checkpoint_ep{N}.pth / backbone_ep{N}.pth（不建 ep{N}/ 子目录）")
    g.add_argument("--no_recon", action="store_true", default=False,
                   help="关闭 epoch 末重建可视化（不生成 recon.png）")
    g.add_argument("--num_workers",     type=int, default=8,
                   help="DataLoader 并行读取进程数；V100 + 本地 NVMe 推荐 8～16")
    g.add_argument("--prefetch_factor", type=int, default=4,
                   help="DataLoader 预取批次数（num_workers>0 时生效）；增大可减少 GPU 等数据")
    g.add_argument("--persistent_workers", action="store_true", default=True,
                   help="保持 DataLoader worker 进程常驻（减少每轮 epoch 重启开销，默认开启）")
    g.add_argument("--no_persistent_workers", action="store_false", dest="persistent_workers",
                   help="关闭 persistent_workers（调试时用）")
    g.add_argument("--dry_run",     action="store_true", default=False,
                   help="数据检查模式：只检查数据管道（形状/值域/NaN），不训练，检查完自动退出。")

    # ── 分辨率 / 增强 ─────────────────────────────────────────────────────────
    g = p.add_argument_group("分辨率与增强")
    g.add_argument("--primary_size", type=int,   default=512,
                   help="图像分辨率（图像已预处理到此尺寸，训练固定使用）")
    g.add_argument("--patch_size",   type=int,   default=16,
                   help="Patch 边长；需整除 primary_size；推荐 16")
    g.add_argument("--color_jitter", type=float, default=0.0,
                   help="颜色抖动强度（0=关闭；默认配合 RRC 使用，一般无需开启）")
    g.add_argument("--vflip_prob",   type=float, default=0.5,
                   help="随机垂直翻转概率")
    g.add_argument("--hflip_prob",   type=float, default=0.5,
                   help="随机水平翻转概率")
    g.add_argument("--rrc", action="store_true", default=True,
                   help="RandomResizedCrop（MAE 风格，输出仍为 primary_size；默认开启）")
    g.add_argument("--no_rrc", action="store_false", dest="rrc",
                   help="关闭 RRC，仅固定尺寸 + flip")
    g.add_argument("--rrc_scale", default="0.6,1.0",
                   help="RRC scale 范围，逗号分隔，如 0.6,1.0")
    g.add_argument("--rrc_ratio", default="0.75,1.33",
                   help="RRC 宽高比范围，逗号分隔，如 0.75,1.33")
    g.add_argument("--min_sample_std", type=float, default=3.0,
                   help="全局像素 std 低于此值的样本在 worker 内重采样（0=关闭）")

    # ── 模型 ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("模型")
    g.add_argument("--model_size",      default="small",
                   choices=["tiny", "small", "base", "large"])
    g.add_argument("--mask_ratio",      type=float, default=0.60,
                   help="SimMIM 掩码比例")
    g.add_argument("--norm_pix_loss",   action="store_true",  default=True,
                   help="对 patch 像素归一化再计算损失（默认开启）")
    g.add_argument("--no_norm_pix_loss", action="store_false", dest="norm_pix_loss",
                   help="关闭 patch 归一化损失")
    g.add_argument("--filter_low_var", action="store_true", default=True,
                   help="过滤低纹理 patch，不参与 loss（与 norm_pix_loss 独立）")
    g.add_argument("--no_filter_low_var", action="store_false", dest="filter_low_var",
                   help="关闭低纹理 patch 过滤")
    g.add_argument("--low_var_std_thresh", type=float, default=0.05,
                   help="patch 像素 std 低于此阈值视为低纹理（ImageNet 归一化后尺度）")
    g.add_argument("--decoder_hidden",  type=int,   default=512,
                   help="解码头隐层通道数")
    g.add_argument("--drop_path",       type=float, default=0.0,
                   help="DropPath 率（SimMIM/MAE 预训练推荐 0）")
    g.add_argument("--ssm_version",     default="mamba1", choices=["mamba1", "mamba3"])
    g.add_argument("--ssm_headdim",     type=int,   default=64)
    g.add_argument("--ssm_backend",     default=None,
                   help="SSM CUDA 后端；None=自动；可选 'torch_easy' 调试")
    g.add_argument("--ssm_fp32",        action="store_true", default=False,
                   help="selective_scan 内核内强制 FP32（数值更稳，略慢）")
    g.add_argument("--no_auto_volta_ssm_fp32", action="store_true", default=False,
                   help="关闭：在 sm<80 且 FP16 AMP 时自动设置 EARTH_MAMBA_SELECTIVE_SCAN_FP32=1 "
                        "（仍走 CUDA 扩展，非 PyTorch 重实现；用于缓解 V100 上部分 invalid configuration）")
    g.add_argument("--disable_triton_cross_scan", action="store_true", default=False,
                   help="手动关闭 CrossScan/CrossMerge 的 Triton kernel（排查旧 GPU kernel 报错时用；"
                        "默认关闭此选项，保持原项目 CUDA/Triton 路径）")
    g.add_argument("--use_checkpoint",  action="store_true", default=False,
                   help="梯度检查点（节省显存，略慢）")
    g.add_argument("--use_compile", action="store_true", default=False,
                   help="实验性：启用 torch.compile (reduce-overhead)。"
                        "已知 Earth-Mamba SSM 中有符号形状推理兼容性问题，默认关闭。")
    g.add_argument("--sync_bn",         action="store_true", default=False,
                   help="DDP 模式下将 BN 转为 SyncBN")

    # ── 训练超参 ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("训练")
    g.add_argument("--epochs",        type=int,   default=800)
    g.add_argument("--warmup_epochs", type=int,   default=40)
    g.add_argument("--batch_size",    type=int,   default=16,
                   help="单卡 batch。≤16GB 显存建议 12~16；32GB V100 上可先试 24，稳定后再试 28。"
                        "仍 OOM 可加 --use_checkpoint 或降低 batch。")
    g.add_argument("--grad_accum",    type=int,   default=1,
                   help="梯度累积步数（等效扩大 batch）")
    g.add_argument("--lr",            type=float, default=1.5e-4,
                   help="base lr（实际 lr = lr * eff_batch / 256）")
    g.add_argument("--auto_scale_lr", action="store_true",  default=True,
                   help="按有效 batch size 线性缩放 lr（默认开启）")
    g.add_argument("--no_auto_scale_lr", action="store_false", dest="auto_scale_lr",
                   help="关闭 LR 自动缩放")
    g.add_argument("--weight_decay",  type=float, default=0.05)
    g.add_argument("--beta1",         type=float, default=0.9)
    g.add_argument("--beta2",         type=float, default=0.95)
    g.add_argument("--clip_grad",     type=float, default=1.0)
    g.add_argument("--use_amp",    action="store_true",  default=True,
                   help="开启混合精度训练（默认开启）")
    g.add_argument("--no_amp",     action="store_false", dest="use_amp",
                   help="关闭混合精度训练（全 FP32）")
    g.add_argument("--amp_dtype",  default="fp16", choices=["bf16", "fp16"],
                   help="AMP 数据类型。V100/Turing(sm<80) 只支持 fp16，A100+ 才能用 bf16")
    g.add_argument("--fsdp",          action="store_true", default=False,
                   help="使用 FSDP FULL_SHARD（超大模型 / 超多卡时开启）")

    # ── 杂项 ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("杂项")
    g.add_argument("--seed",            type=int,   default=42)
    g.add_argument("--log_every",       type=int,   default=50,
                   help="每 N step 打印一次日志")
    g.add_argument("--step_log_every",  type=int,   default=10,
                   help="每 N 个 dataloader step 写一行 step_log.jsonl；"
                        "每次写入后立即 flush，便于 tail -f 实时查看；"
                        "正式大规模训练可调大（如 50）以减少 IO 频率")
    g.add_argument("--save_every",      type=int,   default=50,
                   help="每 N 个 epoch 结束保存 checkpoint 与 backbone（flat_ckpt 时为 epochs/checkpoint_ep{E}.pth）")
    g.add_argument("--viz_every",       type=int,   default=50,
                   help="每 N 个 epoch 结束生成 recon.png（0 或 --no_recon 关闭）")
    g.add_argument("--periodic_every",  type=int,   default=2000,
                   help="每 N 次「本 epoch 内」optimizer.step 写一次 periodic/ep{E}.{步}/ "
                        "（E 从1起，步为本轮累计 opt 步；每新 epoch 步数重新从1计）；"
                        "0=关闭。checkpoint 内仍含 global_update_step（全局累计）供恢复。")
    g.add_argument("--resume",          default=None,
                   help="从 checkpoint 恢复训练")
    g.add_argument("--resume_epoch",    type=int, default=None,
                   help="配合 --resume 使用：指定续训的具体 epoch 编号（默认找最新的）")
    g.add_argument("--resume_weights_only", action="store_true", default=False,
                   help="仅加载 checkpoint 中的模型权重；optimizer/scheduler 从零开始，"
                        "epoch 从 0 重新计数（用于逃离梯度停滞的局部最优）")
    g.add_argument("--monitor_interval", type=float, default=1.0,
                   help="CPU/GPU 资源监控采样周期（秒）")
    g.add_argument("--disable_monitor", action="store_true", default=False,
                   help="关闭 ResourceMonitor（减少监控线程开销）")
    g.add_argument("--zero_loss_dump_threshold", type=int, default=0,
                   help="loss<1e-6 连续 N 步则 dump 该 batch 并置零梯度（0=关闭）")

    return p.parse_args()


def _epoch_save_paths(epochs_dir: Path, ep_tag: int, flat_ckpt: bool):
    """返回 (mkdir_target, checkpoint_path, backbone_path)。"""
    if flat_ckpt:
        return (
            epochs_dir,
            epochs_dir / f"checkpoint_ep{ep_tag}.pth",
            epochs_dir / f"backbone_ep{ep_tag}.pth",
        )
    ep_dir = epochs_dir / f"ep{ep_tag}"
    return ep_dir, ep_dir / "checkpoint.pth", ep_dir / "backbone.pth"


def _find_resume_ckpt(
    resume_path: str,
    resume_epoch: Optional[int] = None,
    flat_ckpt: bool = False,
):
    """智能查找断点。支持传入 checkpoint 路径或输出根目录。
    如果指定 resume_epoch，则加载对应 epoch 的 checkpoint。
    返回 (checkpoint_path, output_dir)。
    """
    p = Path(resume_path).expanduser().resolve()
    # 情况 1：直接指向 checkpoint 文件（子目录或 flat 命名）
    if p.is_file() and (
        p.name == "checkpoint.pth" or p.name.startswith("checkpoint_ep")
    ):
        parent = p.parent
        for _ in range(5):
            if (parent / "logs").is_dir() and (parent / "epochs").is_dir():
                return str(p), str(parent)
            parent = parent.parent
        return str(p), None
    # 情况 2：指向了输出根目录
    if p.is_dir():
        if resume_epoch is not None:
            flat_ckpt_path = p / "epochs" / f"checkpoint_ep{resume_epoch}.pth"
            nested_ckpt_path = p / "epochs" / f"ep{resume_epoch}" / "checkpoint.pth"
            if flat_ckpt_path.is_file():
                ep_ckpt = flat_ckpt_path
            elif nested_ckpt_path.is_file():
                ep_ckpt = nested_ckpt_path
            else:
                raise FileNotFoundError(
                    f"指定的 epoch={resume_epoch} checkpoint 不存在: "
                    f"{flat_ckpt_path} 或 {nested_ckpt_path}"
                )
            return str(ep_ckpt), str(p)
        # 不指定 epoch：找最新
        ckpts = (
            list(p.glob("epochs/checkpoint_ep*.pth"))
            + list(p.rglob("epochs/*/checkpoint.pth"))
            + list(p.rglob("periodic/*/checkpoint.pth"))
        )
        if not ckpts:
            raise FileNotFoundError(
                f"在 {p} 及其子目录中未找到任何 checkpoint（含 flat 命名）"
            )
        latest = max(ckpts, key=lambda x: x.stat().st_mtime)
        return str(latest), str(p)
    raise FileNotFoundError(f"断点路径不存在: {p}")


def main():
    args = build_args()

    # sm_80+(A800): expandable_segments 减少显存碎片（不额外 roundup，避免浪费）
    # sm_70(V100): 仅 max_split_size_mb + GC
    if torch.cuda.is_available():
        _cc = torch.cuda.get_device_capability()
        if _cc[0] >= 8:
            os.environ.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF",
                "max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.6"
            )
        else:
            os.environ.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF",
                "max_split_size_mb:128,garbage_collection_threshold:0.6"
            )

    # ── DDP 初始化 ────────────────────────────────────────────────────────────
    local_rank, use_ddp = setup_ddp()
    world_size = dist.get_world_size() if use_ddp else 1
    rank       = dist.get_rank()       if use_ddp else 0

    if not is_master():
        builtins.print = lambda *a, **kw: None

    if getattr(args, "ssm_fp32", False):
        os.environ["EARTH_MAMBA_SELECTIVE_SCAN_FP32"] = "1"
    elif getattr(args, "no_auto_volta_ssm_fp32", False):
        os.environ.pop("EARTH_MAMBA_SELECTIVE_SCAN_FP32", None)

    # 这些是排查开关，默认不要让 shell 里残留的 export 影响训练路径。
    # 保持默认行为与 pretrain_mae_v1 / 原 EarthMamba 项目一致。
    if getattr(args, "disable_triton_cross_scan", False):
        os.environ["EARTH_MAMBA_DISABLE_TRITON_CROSS_SCAN"] = "1"
    else:
        os.environ.pop("EARTH_MAMBA_DISABLE_TRITON_CROSS_SCAN", None)

    if getattr(args, "ssm_backend", None):
        os.environ["EARTH_MAMBA_SSM_BACKEND"] = str(args.ssm_backend)
    else:
        os.environ.pop("EARTH_MAMBA_SSM_BACKEND", None)

    logging.basicConfig(
        level=logging.INFO if is_master() else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("pretrain")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    # ── 智能断点续训与输出目录 ──────────────────────────────────────────────
    ckpt_path_to_load = None
    resumed_out_dir = None
    resume_epoch_specified = args.resume_epoch is not None
    if args.resume:
        ckpt_path_to_load, resumed_out_dir = _find_resume_ckpt(
            args.resume, args.resume_epoch, flat_ckpt=getattr(args, "flat_ckpt", False)
        )
        if is_master() and getattr(args, "resume_weights_only", False) and args.resume_epoch is None:
            logger.warning(
                "resume_weights_only 建议配合 --resume_epoch 指定权重来源，"
                "否则将加载最新 checkpoint 的模型权重"
            )
        if is_master():
            logger.info(f"断点续训：找到检查点 {ckpt_path_to_load}")
            if resumed_out_dir:
                logger.info(f"原始输出目录：{resumed_out_dir}")
            if resume_epoch_specified:
                logger.info("指定了 --resume_epoch，将写入新目录（不覆盖原始目录）")
            if getattr(args, "resume_weights_only", False):
                logger.info("resume_weights_only：仅加载模型权重，optimizer/scheduler 重置")
    if args.flat_output:
        out_dir          = Path(args.output_dir).expanduser().resolve()
        intermediate_dir = out_dir / "intermediate"
        logs_dir         = out_dir / "logs"
        periodic_dir     = out_dir / "periodic"
        epochs_dir       = out_dir / "epochs"
        if is_master():
            for _d in (intermediate_dir, logs_dir, periodic_dir, epochs_dir):
                _d.mkdir(parents=True, exist_ok=True)
        if use_ddp:
            dist.barrier()
    elif resumed_out_dir and not resume_epoch_specified and not getattr(args, "resume_weights_only", False):
        # 续训但未指定 epoch：复用旧目录
        out_dir          = Path(resumed_out_dir).resolve()
        intermediate_dir = out_dir / "intermediate"
        logs_dir         = out_dir / "logs"
        periodic_dir     = out_dir / "periodic"
        epochs_dir       = out_dir / "epochs"
        if is_master():
            for _d in (intermediate_dir, logs_dir, periodic_dir, epochs_dir):
                _d.mkdir(parents=True, exist_ok=True)
            logger.info(f"输出目录（续训）: {out_dir}")
        if use_ddp:
            dist.barrier()
    else:
        (out_dir, intermediate_dir,
         logs_dir, periodic_dir, epochs_dir) = _build_run_layout(args.output_dir, use_ddp)
        if is_master():
            logger.info(f"输出目录（新训练）: {out_dir}")

    # ── 将 ALL 临时文件锁进 intermediate/，不泄漏到工作目录 ──
    mpl_cfg_dir = intermediate_dir / "mplconfig"
    tmp_dir     = intermediate_dir / "tmp"
    if is_master():
        mpl_cfg_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()
    os.environ["MPLCONFIGDIR"] = str(mpl_cfg_dir)
    os.environ["TMPDIR"]       = str(tmp_dir)
    os.environ["TEMPDIR"]      = str(tmp_dir)
    os.environ["TEMP"]         = str(tmp_dir)
    if is_master():
        logger.info(f"输出目录: {out_dir}")
        logger.info(f"  logs/         {logs_dir}")
        logger.info(f"  epochs/       {epochs_dir}")
        logger.info(f"  periodic/     {periodic_dir}")
        logger.info(f"  intermediate/ {intermediate_dir}")
        # launch_cmd.txt 在 auto_scale_lr 之后写，确保记录实际使用的 LR

    # ── GPU capability 检查 ───────────────────────────────────────────────────
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        # 固定输入尺寸（512×512），benchmark 让 cuDNN 自动选最快卷积算法
        torch.backends.cudnn.benchmark = True

    if device.type == "cuda" and is_master():
        _prop   = torch.cuda.get_device_properties(local_rank)
        _mem_gb = _prop.total_memory / (1024 ** 3)
        if _mem_gb >= 29.0:
            logger.info(
                f"GPU 显存约 {_mem_gb:.1f} GB（近似 32GB 级）；"
                "当前单卡 batch_size=%d。"
                "配合 auto_scale_lr 有效 batch 变大后学习率会自动线性放大。"
                % args.batch_size
            )
    if device.type == "cuda" and getattr(args, "use_amp", True) and getattr(args, "amp_dtype", "bf16") == "bf16":
        _cc = torch.cuda.get_device_capability(device)
        if _cc[0] < 8:
            if is_master():
                logger.warning(
                    f"GPU compute capability={_cc[0]}.{_cc[1]}（V100/T4），"
                    "不支持原生 BF16 张量核心。建议改用 --amp_dtype fp16。"
                )

    # Volta/Turing(sm<80) + FP16 AMP：selective_scan CUDA backward 在部分机器报
    # invalid configuration argument → 默认打开内核内 FP32（仍是 CUDA 扩展）
    if (
        device.type == "cuda"
        and args.use_amp
        and args.amp_dtype == "fp16"
        and not getattr(args, "no_auto_volta_ssm_fp32", False)
    ):
        _cc2 = torch.cuda.get_device_capability(device)
        if _cc2[0] < 8 and os.environ.get("EARTH_MAMBA_SELECTIVE_SCAN_FP32") != "1":
            os.environ["EARTH_MAMBA_SELECTIVE_SCAN_FP32"] = "1"
            if is_master():
                logger.info(
                    "[稳定] sm<80 + FP16 AMP：已设置 EARTH_MAMBA_SELECTIVE_SCAN_FP32=1 "
                    "（CUDA selective_scan 内部 FP32，用于提高 V100/Turing 稳定性）"
                )

    if device.type == "cuda" and is_master():
        logger.info(
            "SSM 后端开关: "
            f"EARTH_MAMBA_SELECTIVE_SCAN_FP32={os.environ.get('EARTH_MAMBA_SELECTIVE_SCAN_FP32')} "
            f"EARTH_MAMBA_DISABLE_TRITON_CROSS_SCAN={os.environ.get('EARTH_MAMBA_DISABLE_TRITON_CROSS_SCAN')} "
            "（默认保持原项目 CUDA/Triton 路径；未设置则显示 None）"
        )

    # ── 有效 batch size 与 lr 自动缩放 ───────────────────────────────────────
    eff_batch = args.batch_size * world_size * args.grad_accum
    if args.auto_scale_lr:
        args.lr = args.lr * eff_batch / 256
        if is_master():
            logger.info(f"线性缩放 LR: {args.lr:.2e}  (eff_batch={eff_batch})")

    # 在 auto_scale_lr 之后写 launch_cmd.txt，确保记录的 lr 是实际训练值
    if is_master():
        launch_cmd_path = logs_dir / "launch_cmd.txt"
        with open(launch_cmd_path, "w", encoding="utf-8") as _lf:
            _lf.write("# Raw command line:\n")
            _lf.write(" ".join(sys.argv) + "\n\n")
            _lf.write("# All args (including defaults, lr already auto-scaled):\n")
            json.dump(vars(args), _lf, indent=2, ensure_ascii=False)
            _lf.write("\n")
        logger.info(f"调用命令已保存 → {launch_cmd_path}")

    # ── 数据集 & DataLoader ───────────────────────────────────────────────────
    if getattr(args, "tar_dir", None) and getattr(args, "lmdb", False):
        sys.exit("错误: --tar_dir 和 --lmdb 不能同时指定")
    if not getattr(args, "tar_dir", None) and not getattr(args, "lmdb", False) and not args.data_list:
        sys.exit("错误: 必须指定 --data_list 或 --tar_dir 或 --lmdb")

    if getattr(args, "lmdb", False):
        dataset = LmdbPretrainDataset(
            lmdb_path=args.data_list,
            primary_size=args.primary_size,
        )
        if is_master():
            logger.info(f"数据集: LMDB 模式 ({args.data_list})")
    elif getattr(args, "tar_dir", None) is not None:
        index_path = os.path.join(args.tar_dir, "tar_index.jsonl")
        dataset = TarPretrainDataset(
            index_path=index_path,
            primary_size=args.primary_size,
        )
        if is_master():
            logger.info(f"数据集: Tar分片模式 ({args.tar_dir}, {len(dataset):,} 张)")
    else:
        dataset = PretrainDataset(
            txt_file=args.data_list,
            primary_size=args.primary_size,
            rglob=args.rglob,
        )

    train_augmentor = build_train_augmentor(args)
    dataset = AugmentedDataset(
        dataset,
        augmentor=train_augmentor,
        min_sample_std=getattr(args, "min_sample_std", 3.0),
    )

    collator = FixedSizeCollator(primary=args.primary_size)

    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    dl_kwargs: dict = dict(
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collator,
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = args.prefetch_factor
        dl_kwargs["persistent_workers"] = args.persistent_workers
        # LMDB: fork 后 worker 需重建环境（LMDB 不 fork-safe）
        if getattr(args, "lmdb", False) and HAS_LMDB:
            def _lmdb_worker_init(_worker_id):
                ds = torch.utils.data.get_worker_info().dataset
                base = ds.base if isinstance(ds, AugmentedDataset) else ds
                if hasattr(base, "_reopen"):
                    base._reopen()
            dl_kwargs["worker_init_fn"] = _lmdb_worker_init
        # tar: fork 后清空父进程的 tar handle cache，每个 worker 独立打开
        elif getattr(args, "tar_dir", None) is not None:
            def _tar_worker_init(_worker_id):
                ds = torch.utils.data.get_worker_info().dataset
                base = ds.base if isinstance(ds, AugmentedDataset) else ds
                if hasattr(base, "_tar_fh") and base._tar_fh is not None:
                    base._tar_fh.close()
                    base._tar_fh = None
                    base._tar_path = None
            dl_kwargs["worker_init_fn"] = _tar_worker_init
    loader = DataLoader(dataset, **dl_kwargs)

    if is_master():
        _rrc_scale = _parse_float_pair(getattr(args, "rrc_scale", "0.6,1.0"), (0.6, 1.0))
        _rrc_ratio = _parse_float_pair(getattr(args, "rrc_ratio", "0.75,1.33"), (0.75, 4.0 / 3.0))
        _aug_desc = (
            f"RRC scale={_rrc_scale} ratio={_rrc_ratio}"
            if getattr(args, "rrc", True)
            else "固定尺寸（无 RRC）"
        )
        if args.color_jitter > 0:
            _aug_desc += f"  color_jitter={args.color_jitter}"
        logger.info(
            f"数据集: {len(dataset)} 张图像 | Loader: {len(loader)} steps/epoch\n"
            f"  输出分辨率: {args.primary_size}×{args.primary_size}（增强在 worker 内）\n"
            f"  增强: {_aug_desc}  flip=({args.hflip_prob},{args.vflip_prob})\n"
            f"  patch_size: {args.patch_size}  mask_ratio: {args.mask_ratio:.0%}\n"
            f"  periodic_every(本 epoch 内 optimizer 步): {getattr(args, 'periodic_every', 0)}"
        )

    # ── dry_run 数据检查 ──────────────────────────────────────────────────────
    if args.dry_run:
        if is_master():
            logger.info("dry_run：开始数据管道检查（不训练）…")
            run_data_check(
                args=args,
                dataset=dataset,
                collator=collator,
                logger=logger,
                out_dir=out_dir,
                n_batches=10,
            )
        if use_ddp:
            dist.barrier()
        cleanup_ddp()
        return

    # ── 模型 ─────────────────────────────────────────────────────────────────
    model = EarthMambaForPretraining(
        model_size=args.model_size,
        patch_size=args.patch_size,
        primary_size=args.primary_size,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
        filter_low_var=args.filter_low_var,
        low_var_std_thresh=args.low_var_std_thresh,
        decoder_hidden=args.decoder_hidden,
        drop_path_rate=args.drop_path,
        ssm_version=args.ssm_version,
        ssm_headdim=args.ssm_headdim,
        ssm_backend=args.ssm_backend,
        use_checkpoint=args.use_checkpoint,
    ).to(device)

    if is_master():
        total_params   = sum(p.numel() for p in model.parameters()) / 1e6
        encoder_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
        logger.info(
            f"模型参数量: 总计 {total_params:.1f}M  "
            f"(编码器 {encoder_params:.1f}M + 解码头 {total_params-encoder_params:.1f}M)"
        )
        # 诊断：selective_scan CUDA kernel 加载状态
        try:
            from earth_mamba.utils.vmamba_core import SS_BACKEND, SS_BACKEND as _ssb
            _has_triton = False
            try:
                import triton as _triton
                _has_triton = True
            except ImportError:
                pass
            if SS_BACKEND is None:
                logger.warning("⚠ selective_scan CUDA kernel 未加载！将回退到纯 PyTorch 实现（极慢）。"
                             "请在 earth-mamba 目录执行: pip install -e ./kernels/selective_scan")
            else:
                logger.info(f"selective_scan 后端: {SS_BACKEND}  |  Triton: {'✓' if _has_triton else '✗'}")
        except Exception:
            pass

    if use_ddp and getattr(args, "sync_bn", False):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ── DDP / FSDP 封装 ───────────────────────────────────────────────────────
    fsdp_used = False
    if use_ddp and args.fsdp and world_size > 1:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            import functools

            _wrap_cls: set = set()
            try:
                from earth_mamba.models.earth_mamba_block import EarthMambaBlock as _EMBlock
                _wrap_cls.add(_EMBlock)
            except ImportError:
                pass

            auto_wrap = None
            if _wrap_cls:
                auto_wrap = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=frozenset(_wrap_cls),
                )
            mp = None
            if args.use_amp:
                _dt = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
                mp  = MixedPrecision(param_dtype=_dt, reduce_dtype=_dt, buffer_dtype=_dt)
            model = FSDP(
                model, device_id=local_rank,
                mixed_precision=mp,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=auto_wrap,
                use_orig_params=True,
            )
            fsdp_used = True
            if is_master():
                logger.info("[并行] 已启用 FSDP FULL_SHARD")
        except Exception as e:
            if is_master():
                logger.warning(f"FSDP 初始化失败，回退 DDP: {e}")

    if use_ddp and not fsdp_used:
        model = DDP(
            model, device_ids=[local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,       # 模型无 BN 缓冲区需广播
            gradient_as_bucket_view=True,  # 梯度 all-reduce 减少内存拷贝
        )
        if is_master():
            logger.info("[并行] 已启用 DDP（broadcast_buffers=0 bucket_view=1）")

    # ── torch.compile 加速（实验性，默认关闭）──────────────────────────────
    #  已知问题：Earth-Mamba 的 SSM 模块存在动态符号形状推理，
    #  torch._dynamo 会将张量维度符号化后陷入 sympy 指数递归。
    #  仅当 --use-compile 且确认 PyTorch ≥ 2.5 + 静态输入时启用。
    if getattr(args, "use_compile", False) and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            if is_master():
                logger.info("[编译] 已启用 torch.compile (mode=reduce-overhead)")
        except Exception as e:
            if is_master():
                logger.warning(f"torch.compile 失败，回退 eager 模式: {e}")

    # ── 优化器 ────────────────────────────────────────────────────────────────
    encoder_params = [
        p for n, p in model.named_parameters()
        if "encoder" in n and p.requires_grad
    ]
    decoder_params = [
        p for n, p in model.named_parameters()
        if "encoder" not in n and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr},
            {"params": decoder_params, "lr": args.lr * 2, "weight_decay": 0.0},
        ],
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=True if device.type == "cuda" else False,
    )

    steps_per_epoch = math.ceil(len(loader) / max(1, args.grad_accum))
    scheduler = build_cosine_schedule(
        optimizer, args.warmup_epochs, args.epochs, steps_per_epoch,
    )

    scaler: Optional[GradScaler] = None
    if args.use_amp and args.amp_dtype == "fp16":
        if fsdp_used:
            try:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
                scaler = ShardedGradScaler()
            except ImportError:
                pass
        else:
            try:
                scaler = GradScaler("cuda")    # torch.amp API（PyTorch ≥ 2.0）
            except TypeError:
                scaler = GradScaler()           # torch.cuda.amp API（PyTorch < 2.0）

    # ── 断点恢复 ──────────────────────────────────────────────────────────────
    start_epoch = 0
    global_opt_step: Dict[str, int] = {"n": 0}
    _pending_optimizer_state = None  # 延迟加载：等第一次 backward 后（激活已释放）再加载
    if ckpt_path_to_load is not None:
        ckpt = torch.load(ckpt_path_to_load, map_location="cpu", weights_only=False)
        raw_for_load = model
        if fsdp_used:
            pass
        elif use_ddp:
            raw_for_load = model.module
        raw_for_load.load_state_dict(ckpt["model"], strict=True)
        weights_only = getattr(args, "resume_weights_only", False)
        if weights_only:
            _pending_optimizer_state = None
            start_epoch = 0
            global_opt_step["n"] = 0
            if is_master():
                _src_ep = int(ckpt.get("epoch", -1)) + 1
                logger.info(
                    f"resume_weights_only：已加载 epoch {_src_ep} 的模型权重；"
                    f"optimizer/scheduler 从零开始，训练从 epoch 1 计数"
                )
        else:
            # 优化器状态延迟加载！正常训练时反向传播之后才创建，续训时若提前加载会挤占激活显存
            _pending_optimizer_state = ckpt.pop("optimizer")  # pop 避免 CPU 内存泄漏
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            global_opt_step["n"] = int(ckpt.get("global_update_step", 0))
            if is_master():
                logger.info(
                    f"从 epoch {start_epoch} 恢复训练  "
                    f"global_opt_step={global_opt_step['n']}  "
                    f"(优化器状态延迟加载)"
                )
        del ckpt       # 释放 checkpoint 字典，回收 CPU 内存
        gc.collect()

    # ── ResourceMonitor（后台线程，用于摘要统计）────────────────────────────
    # per-step 资源数据直接在 step 循环里写 resource_log.jsonl（崩溃安全）
    monitor: Optional[ResourceMonitor] = None
    if is_master() and not getattr(args, "disable_monitor", False):
        ResourceMonitor.deps_hint()
        monitor = ResourceMonitor(device=device,
                                  interval_sec=getattr(args, "monitor_interval", 1.0))
        monitor.start()

    step_log_path     = logs_dir / "step_log.jsonl"
    resource_log_path = logs_dir / "resource_log.jsonl"
    # buffering=1 = 行缓冲；每次 write 后显式 flush，确保崩溃/中途 tail -f 可见
    step_log_f = (
        open(step_log_path,     "a", encoding="utf-8", buffering=1) if is_master() else None
    )
    resource_log_f = (
        open(resource_log_path, "a", encoding="utf-8", buffering=1) if is_master() else None
    )

    # psutil cpu_percent 首次调用需要预热，否则返回 0
    if is_master():
        _warmup_psutil()

    # ── 训练主循环 ────────────────────────────────────────────────────────────
    train_log = []
    loss_history: list = []
    lr_history: list   = []
    periodic_seg_start: Dict[str, int] = {"n": 0}
    epoch_local_opt_step: Dict[str, int] = {"n": 0}
    total_start = time.time()
    _zl_counter: Optional[Dict[str, int]] = (
        {"n": 0} if getattr(args, "zero_loss_dump_threshold", 0) > 0 else None
    )
    _zl_dir: Optional[Path] = out_dir if _zl_counter is not None else None

    try:
        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_local_opt_step["n"] = 0
            epoch_loss_n = len(loss_history)  # 记录本 epoch 起点，用于画分段曲线
            t0 = time.time()
            avg_loss, last_batch = train_one_epoch(
                model, loader, optimizer, scheduler, scaler,
                epoch, args, logger,
                device=device,
                step_log_f=step_log_f,
                resource_log_f=resource_log_f,
                loss_history=loss_history,
                lr_history=lr_history,
                fsdp_used=fsdp_used,
                use_ddp=(use_ddp and not fsdp_used),
                global_opt_step=global_opt_step,
                epoch_local_opt_step=epoch_local_opt_step,
                periodic_every=getattr(args, "periodic_every", 0),
                periodic_dir=periodic_dir,
                out_dir=out_dir,
                periodic_seg_start=periodic_seg_start,
                pending_optimizer_state=_pending_optimizer_state,
                zero_loss_counter=_zl_counter,
                zero_loss_dump_dir=_zl_dir,
            )
            # 第一次加载后清空，后续 epoch 不再重复加载
            _pending_optimizer_state = None
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]

            if is_master():
                logger.info(
                    f"── Epoch {epoch+1:03d}/{args.epochs}  "
                    f"avg_loss={avg_loss:.4f}  "
                    f"lr={lr_now:.2e}  "
                    f"time={elapsed:.0f}s"
                )
                train_log.append({"epoch": epoch + 1, "loss": avg_loss, "lr": lr_now})

            do_save = (epoch + 1) == 1 or (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs
            if do_save:
                _gs  = global_opt_step["n"]
                _elu = epoch_local_opt_step["n"]
                ep_tag = epoch + 1
                flat_ckpt = getattr(args, "flat_ckpt", False)
                save_dir, ckpt_path, bb_path = _epoch_save_paths(epochs_dir, ep_tag, flat_ckpt)
                if is_master():
                    save_dir.mkdir(parents=True, exist_ok=True)
                if use_ddp:
                    dist.barrier()
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, scaler,
                    epoch, args,
                    fsdp_used=fsdp_used,
                    use_ddp=(use_ddp and not fsdp_used),
                    global_update_step=_gs,
                    epoch_local_opt_step=_elu,
                )
                if is_master():
                    logger.info(f"Checkpoint → {ckpt_path}")
                save_backbone_weights(
                    bb_path, model,
                    fsdp_used=fsdp_used,
                    use_ddp=(use_ddp and not fsdp_used),
                )
                if is_master():
                    logger.info(f"Backbone   → {bb_path}")
                # save 后清理 CUDA cache，避免 state_dict 操作碎片化
                if device is not None and device.type == "cuda":
                    torch.cuda.synchronize()

            # FSDP 前向传播需要全 rank 参与，不能只在 rank-0 运行推理；
            # 因此 FSDP 模式下跳过可视化（DDP / 单卡正常执行）
            viz_every = getattr(args, "viz_every", 50)
            do_viz = (
                not getattr(args, "no_recon", False)
                and viz_every > 0
                and not fsdp_used
                and ((epoch + 1) % viz_every == 0 or (epoch + 1) == args.epochs)
            )
            if do_viz and not fsdp_used:
                # barrier 确保 DDP 各 rank 进入 / 退出 viz 同步，避免下一 epoch 错位
                if use_ddp:
                    dist.barrier()
                if is_master():
                    ep_tag = epoch + 1
                    flat_ckpt = getattr(args, "flat_ckpt", False)
                    if flat_ckpt:
                        recon_path = epochs_dir / f"recon_ep{ep_tag}.png"
                    else:
                        ep_dir = epochs_dir / f"ep{ep_tag}"
                        ep_dir.mkdir(parents=True, exist_ok=True)
                        recon_path = ep_dir / "recon.png"
                    visualize_reconstruction(
                        model, loader, device, args, out_dir, epoch,
                        n_samples=min(4, args.batch_size),
                        save_path=recon_path,
                        suptitle=f"ep{ep_tag}",
                        reference_batch=last_batch,
                    )
                    logger.info(f"  重建可视化 → {recon_path}")
                    if device is not None and device.type == "cuda":
                        torch.cuda.synchronize()
                if use_ddp:
                    dist.barrier()

            # ── 1. 关键：每 epoch 追加写训练进度（最先执行，崩溃也不丢）──
            if is_master() and train_log:
                _running_log = logs_dir / "train_log_running.jsonl"
                with open(_running_log, "a", encoding="utf-8") as _lf:
                    _lf.write(json.dumps(train_log[-1], ensure_ascii=False) + "\n")

            # ── 2. 非关键：画曲线 / 清理（try/except 全包裹，崩了不影响训练）──
            try:
                if is_master() and HAS_MATPLOTLIB and loss_history and lr_history:
                    save_running_curve_pngs(
                        loss_history, lr_history,
                        logs_dir / "loss_curve.png",
                        title_prefix="running (all steps so far)",
                    )
                    ep_tag_loss = epoch + 1
                    if getattr(args, "flat_ckpt", False):
                        loss_png = epochs_dir / f"loss_epoch_ep{ep_tag_loss}.png"
                    else:
                        ep_dir_loss = epochs_dir / f"ep{ep_tag_loss}"
                        ep_dir_loss.mkdir(parents=True, exist_ok=True)
                        loss_png = ep_dir_loss / "loss_epoch.png"
                    ep_loss = loss_history[epoch_loss_n:]
                    ep_lr   = lr_history[epoch_loss_n:]
                    if ep_loss:
                        save_running_curve_pngs(
                            ep_loss, ep_lr,
                            loss_png,
                            title_prefix=f"epoch {ep_tag_loss} only",
                            x_label="step within epoch",
                        )
            except Exception:
                pass

            gc.collect()
            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()

            if is_master():
                os.system("sync")
                if HAS_MATPLOTLIB:
                    import matplotlib.pyplot as _plt
                    _plt.close("all")

    finally:
        if monitor is not None:
            monitor.stop()
        if step_log_f is not None:
            step_log_f.close()
        if resource_log_f is not None:
            resource_log_f.close()
        # ── 清理临时文件 ──
        if is_master():
            import shutil
            for _td in [".tmp", "intermediate/tmp", "intermediate/mplconfig"]:
                _p = Path(args.output_dir) / _td
                if _p.exists():
                    shutil.rmtree(str(_p), ignore_errors=True)

    total_time = time.time() - total_start

    if is_master():
        total_images = len(loader) * args.epochs * args.batch_size * world_size
        images_per_sec = total_images / total_time if total_time > 0 else 0.0
        max_mem_gb = 0.0
        if device.type == "cuda":
            max_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

        monitor_summary: dict = {}
        if monitor is not None:
            monitor_summary = monitor.summary()

        logger.info("====== 预训练结束 ======")
        logger.info(f"  总耗时: {total_time:.0f}s  吞吐: {images_per_sec:.1f} images/s")
        if device.type == "cuda":
            logger.info(f"  单卡最大显存: {max_mem_gb:.2f} GB")
        if monitor_summary:
            logger.info(f"  资源监控摘要: {monitor_summary}")

        # resource_log.jsonl 已在 step 循环里按 step 追加写入；
        # 这里只写摘要 JSON（后台线程的采样统计，作为补充）
        if monitor is not None and monitor.samples:
            with open(logs_dir / "resource_summary.json", "w", encoding="utf-8") as f:
                json.dump(monitor_summary, f, indent=2, ensure_ascii=False)
            save_resource_curve_plots(logs_dir, monitor.samples)
            logger.info(f"  资源摘要 → {logs_dir / 'resource_summary.json'}")
            logger.info(f"  资源曲线 → {logs_dir}/")
        logger.info(f"  逐 step 资源记录 → {resource_log_path}")

        if HAS_MATPLOTLIB and loss_history:
            steps_x = list(range(len(loss_history)))

            def _ema(vals, alpha=0.05):
                s = vals[0]
                out = [s]
                for v in vals[1:]:
                    s = alpha * v + (1 - alpha) * s
                    out.append(s)
                return out

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps_x, loss_history, color="steelblue", alpha=0.3, linewidth=0.8,
                    label="loss (raw)")
            ax.plot(steps_x, _ema(loss_history), color="steelblue", linewidth=1.5,
                    label="loss (EMA)")
            ax.set_xlabel("step")
            ax.set_ylabel("SimMIM Loss")
            ax.set_title("Pre-training Loss Curve")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            plt.savefig(str(logs_dir / "loss_curve.png"), dpi=150)
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(steps_x, lr_history, color="darkorange", linewidth=1.2)
            ax.set_xlabel("step")
            ax.set_ylabel("LR")
            ax.set_title("Learning Rate Schedule (warmup + cosine)")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            plt.savefig(str(logs_dir / "lr_curve.png"), dpi=150)
            plt.close()

            if train_log:
                ep_x  = [r["epoch"] for r in train_log]
                ep_l  = [r["loss"]  for r in train_log]
                ep_lr = [r["lr"]    for r in train_log]
                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax2 = ax1.twinx()
                ax1.plot(ep_x, ep_l,  color="steelblue",  marker=".", label="avg_loss/epoch")
                ax2.plot(ep_x, ep_lr, color="darkorange", linestyle="--", label="lr/epoch")
                ax1.set_xlabel("epoch")
                ax1.set_ylabel("avg loss", color="steelblue")
                ax2.set_ylabel("lr",       color="darkorange")
                ax1.tick_params(axis="y", labelcolor="steelblue")
                ax2.tick_params(axis="y", labelcolor="darkorange")
                fig.suptitle("Epoch-level Loss & LR")
                fig.tight_layout()
                plt.savefig(str(logs_dir / "epoch_loss_lr_curve.png"), dpi=150)
                plt.close()

            logger.info(f"  曲线图 → {logs_dir}/ (loss_curve / lr_curve / epoch_loss_lr_curve)")

        log_path = logs_dir / "pretrain_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "args"            : vars(args),
                "argv"            : sys.argv,
                "train_log"       : train_log,
                "world_size"      : world_size,
                "total_time_s"    : round(total_time, 2),
                "images_per_sec"  : round(images_per_sec, 2),
                "max_mem_gb"      : round(max_mem_gb, 2),
                "monitor_summary" : monitor_summary,
                "step_log_path"   : str(step_log_path),
                "resource_log_path": str(resource_log_path),
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"训练日志 → {log_path}")
        logger.info("保存文件清单:")
        logger.info(f"   {out_dir}/")
        logger.info(f"     logs/launch_cmd.txt               raw argv + 有效 lr")
        logger.info(f"     logs/pretrain_log.json            训练结束汇总")
        logger.info(f"     logs/step_log.jsonl               per-step loss/lr/显存")
        logger.info(f"     logs/resource_log.jsonl           per-step GPU/CPU/RAM")
        logger.info(f"     logs/train_log_running.jsonl      per-epoch 追加（崩溃安全）")
        logger.info(f"     logs/loss_curve.png               全局曲线 PNG")
        if getattr(args, "flat_ckpt", False):
            logger.info(f"     epochs/checkpoint_ep{{E}}.pth      epoch 末完整状态")
            logger.info(f"     epochs/backbone_ep{{E}}.pth        epoch 末编码器权重")
        else:
            logger.info(f"     epochs/ep{{E}}/checkpoint.pth      E=第几轮(从1)，epoch 末完整状态")
            logger.info(f"     epochs/ep{{E}}/backbone.pth        epoch 末编码器权重")
        if not getattr(args, "no_recon", False):
            logger.info(f"     recon.png（flat: recon_ep{{E}}.png） epoch 末重建对比图")
        logger.info(f"     periodic/ep{{E}}.{{本epoch内opt步}}/  每 N 步快照（目录内 ckpt+图）")
        logger.info(f"     intermediate/data_check/          dry_run 数据检验图")

    cleanup_ddp()


if __name__ == "__main__":
    main()
