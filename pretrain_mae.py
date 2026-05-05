"""
pretrain_mae.py  ─  EarthMamba MAE/SimMIM 预训练脚本
=====================================================

训练策略一句话
─────────────
  以 512×512 为主尺度，采用少量整数倍 patch 对齐的尺度扰动；
  预训练采用 MAE 的随机遮挡重建目标；
  模型端支持位置编码 bicubic 插值的可变分辨率机制，
  以保证推理不锁死输入尺寸。

分辨率采样策略（所有输入图像已预处理为 512×512）
─────────────────────────────────────────────────
  • 85% 批次（primary_prob=0.85）：
      直接使用 512×512，无 resize。

  • 15% 批次（尺度扰动）：
      从预定义的离散候选表中均匀抽取一个辅尺度，例如：
          patch_size=16 时: [448, 480, 544, 576]
      候选表由 patch_size 的整数倍自动生成，确保 token 网格严格整数、
      无需 padding / 截断。同一批次内所有样本使用相同辅尺度。

  • 为何选离散候选表而非随机百分比缩放？
      ViT/Mamba 的 patch 嵌入要求输入 H, W 均为 patch_size 的整数倍。
      如果按百分比连续缩放再四舍五入，会产生 460, 489, 537 等"别扭"尺寸，
      token 网格不规则且不可复现。离散候选表一次性定义好合法尺寸，
      训练和推理行为完全可预测。

预训练方法：SimMIM（Masked Image Modeling）
─────────────────────────────────────────────────
  1. 在 patch 网格（H/patch_size × W/patch_size）上随机采样掩码（mask_ratio=0.6）
  2. 被掩码的 patch 区域在像素空间替换为可学习掩码值
  3. 完整 EarthMamba 编码器处理含掩码的输入（所有 patch 均参与编码）
  4. 轻量解码头上采样到输入分辨率，预测原始像素值
  5. 损失：仅在被掩码 patch 的像素上计算归一化 L1

模型端可变分辨率机制
─────────────────────────────────────────────────
  • 2D 可学习位置编码（posembed=True，训练分辨率 = 512/patch_size = 32×32）：
      推理/微调遇到不同分辨率时，自动 bicubic 插值到新的 token 网格尺寸
      （与 DINOv2 / FlexiViT 的做法一致）。
      训练时多尺度批次已在不同 token 网格下执行过插值，
      让位置编码从训练阶段就学会了对尺度变化的适应。
  • patch_embed: Conv2d(stride=patch_size) → 支持任意 patch_size 整数倍的 H, W
  • 下采样: v3 版 Conv(k=3, s=2, p=1) → 对奇偶空间维均无丢边
  • 解码器: 双线性插值 + Conv → 动态适配任意输入分辨率

使用示例
─────────────────────────────────────────────────
单卡调试：
  python pretrain_mae.py \
      --data_root /path/to/512x512_images \
      --output_dir ./pretrain_out \
      --model_size small --patch_size 16 \
      --batch_size 16 --epochs 800 --warmup_epochs 40 \
      --amp --amp_dtype bf16

多卡 DDP（4 卡）：
  torchrun --nproc_per_node=4 pretrain_mae.py \
      --data_root /path/to/512x512_images \
      --output_dir ./pretrain_out \
      --model_size small --patch_size 16 \
      --batch_size 8 --grad_accum 2 \
      --epochs 800 --amp --amp_dtype bf16

微调时加载预训练骨干权重：
  backbone = BackboneEarthMamba(...)
  backbone.load_state_dict(torch.load('pretrain_out/backbone_ep0800.pth'))
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.transforms import functional as TF

# ── 将仓库根加入 sys.path（兼容直接运行和模块运行两种方式）──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── 尝试加载自定义 CUDA 选择性扫描核（可选，缺失时自动回退）────────────────
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
        "请先在 earth-mamba 仓库根目录执行：\n"
        "  pip install -e ./kernels/selective_scan\n"
        "  pip install -e .\n"
    ) from e


# ══════════════════════════════════════════════════════════════════════════════
#  第一节：离散多尺度分辨率采样
# ══════════════════════════════════════════════════════════════════════════════

def build_scale_candidates(
    primary: int = 512,
    patch_size: int = 16,
    min_scale: float = 0.85,
    max_scale: float = 1.15,
) -> List[int]:
    """
    生成离散辅尺度候选表（patch_size 的严格整数倍，且排除 primary 本身）。

    例：primary=512, patch_size=16, ±15%
        下界 = 512 * 0.85 = 435.2 → 向上取整到 448
        上界 = 512 * 1.15 = 588.8 → 向下取整到 576
        候选: [448, 464, 480, 496, 528, 544, 560, 576]
             （512 自身被排除，因为它已由 primary_prob 控制）
    """
    lo = int(math.ceil (primary * min_scale / patch_size)) * patch_size
    hi = int(math.floor(primary * max_scale / patch_size)) * patch_size
    candidates = list(range(lo, hi + 1, patch_size))
    candidates = [s for s in candidates if s != primary]
    return candidates


def sample_batch_resolution(
    primary: int,
    candidates: List[int],
    primary_prob: float = 0.85,
    rng: Optional[random.Random] = None,
) -> int:
    """
    采样本批次使用的分辨率。

    - 以 primary_prob 概率返回 primary（主分辨率 512，无需 resize）。
    - 否则从 candidates（离散候选表）中**均匀随机**抽取一个辅尺度。
    - candidates 为空时始终返回 primary。
    """
    _rng = rng or random
    if not candidates or _rng.random() < primary_prob:
        return primary
    return _rng.choice(candidates)


# ══════════════════════════════════════════════════════════════════════════════
#  第二节：数据集与多尺度 Collator
# ══════════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_to_tensor = transforms.ToTensor()
_normalize  = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

class PretrainDataset(Dataset):
    """
    扁平目录图像数据集（所有图像已预处理为 primary×primary）。

    - __getitem__ 返回 PIL.Image（未 resize / 未 normalize），
      尺寸转换由 MultiScaleCollator 在 batch 级别统一完成。
    - 支持常见遥感图像格式：jpg / jpeg / png / tif / tiff / webp / bmp。
      对 16-bit TIFF / 多波段 TIFF：转 8-bit RGB 时取前 3 个波段，
      像素范围线性归一到 [0, 255]（仅当 mode 不是常规 RGB 时触发）。
    - 递归扫描目录（含子目录），suffix 大小写不敏感。
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

    def __init__(self, root: str, primary_size: int = 512):
        self.root = Path(root)
        self.primary_size = primary_size
        self.paths: List[Path] = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"在 {root} 下未找到任何图像文件")

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _safe_to_rgb(img: Image.Image) -> Image.Image:
        """
        将任意 PIL Image 安全转换为 8-bit RGB。

        - mode == 'RGB' : 直接返回
        - mode 含 alpha (RGBA / LA / P) : 转 RGB 丢弃 alpha
        - 16-bit / 32-bit TIFF (mode I, I;16, F) : 线性归一到 [0,255] 再复制三通道
        - 多波段 TIFF : 取前 3 个波段
        """
        mode = img.mode
        if mode == "RGB":
            return img
        if mode in ("RGBA", "LA", "P", "L"):
            return img.convert("RGB")
        # 高位深 / 浮点：用 numpy 线性归一
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            lo, hi = float(arr.min()), float(arr.max())
            if hi - lo < 1e-6:
                arr = np.zeros_like(arr, dtype=np.uint8)
            else:
                arr = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:                 # 单通道 → 复制三通道
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = arr[..., :3]            # 多波段 → 取前 3
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        else:
            raise ValueError(f"不支持的图像形状: {arr.shape}")
        return Image.fromarray(arr, mode="RGB")

    def __getitem__(self, idx: int) -> Image.Image:
        try:
            img = Image.open(self.paths[idx])
            return self._safe_to_rgb(img)
        except Exception:
            # 损坏文件：返回纯黑 primary×primary 占位图（不会污染太多 step）
            return Image.new("RGB", (self.primary_size, self.primary_size), 0)


class MultiScaleCollator:
    """
    Batch 级别多尺度 Collator（离散候选表版本）。

    设计原则（输入图像均为 512×512）：
    ─────────────────────────────────
    • primary_prob（默认 85%）概率：目标分辨率 = 512（主尺度）。
        图像已是 512×512，无需 resize；只做随机翻转 + 颜色抖动 + normalize。

    • 1-primary_prob 概率：从离散候选表中均匀抽取一个辅尺度。
        候选表由 patch_size 整数倍自动生成，如 [448, 480, 544, 576]。
        将 512 双线性缩放到选中的辅尺度。

    同一批次内所有样本使用相同目标分辨率 → torch.stack 无歧义。
    """

    def __init__(
        self,
        primary: int = 512,
        patch_size: int = 16,
        primary_prob: float = 0.85,
        candidates: Optional[List[int]] = None,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        color_jitter: float = 0.2,
    ):
        self.primary = primary
        self.patch_size = patch_size
        self.primary_prob = primary_prob
        self.candidates = candidates or []
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self._rng = random.Random()

        self._color_jitter: Optional[transforms.ColorJitter] = (
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter * 0.25,
            )
            if color_jitter > 0 else None
        )

    # ------------------------------------------------------------------
    def _augment_one(self, img: Image.Image, size: int) -> torch.Tensor:
        """对单张 PIL 图像做尺寸调整 + 数据增强，返回归一化 Tensor。"""
        if img.width != size or img.height != size:
            img = img.resize((size, size), Image.BILINEAR)

        if self._color_jitter is not None:
            img = self._color_jitter(img)

        if self._rng.random() < self.hflip_prob:
            img = TF.hflip(img)
        if self._rng.random() < self.vflip_prob:
            img = TF.vflip(img)

        return _normalize(_to_tensor(img))

    # ------------------------------------------------------------------
    def __call__(self, batch: List[Image.Image]) -> torch.Tensor:
        size = sample_batch_resolution(
            primary=self.primary,
            candidates=self.candidates,
            primary_prob=self.primary_prob,
            rng=self._rng,
        )
        tensors = [self._augment_one(img, size) for img in batch]
        return torch.stack(tensors, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
#  第三节：SimMIM 预训练模型封装
# ══════════════════════════════════════════════════════════════════════════════

class SimMIMDecoder(nn.Module):
    """
    轻量上采样解码头。

    输入：编码器最后阶段特征图 (B, C, Hf, Wf)
    输出：像素级预测图 (B, 3, H_in, W_in)

    使用双线性插值 + 两层 Conv 完成上采样，
    不含固定大小参数 → 推理阶段支持任意分辨率。
    """

    def __init__(self, encoder_dim: int, hidden_dim: int = 512):
        super().__init__()
        # Conv → GroupNorm → GELU；GroupNorm 不依赖空间尺寸 → 可变分辨率友好
        self.conv1 = nn.Conv2d(encoder_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=hidden_dim)
        self.act1  = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, 3, kernel_size=1)

    def forward(self, feat: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Args:
            feat       : (B, C, Hf, Wf)  编码器输出特征图
            target_h/w : 目标高宽（= 输入图像尺寸）

        Returns:
            (B, 3, target_h, target_w)  像素级预测
        """
        x = self.conv1(feat)                         # (B, hidden, Hf, Wf)
        x = self.norm1(x)
        x = self.act1(x)
        x = F.interpolate(x, size=(target_h, target_w),
                          mode="bilinear", align_corners=False)   # 动态上采样
        x = self.conv2(x)                            # (B, 3, H, W)
        return x


class EarthMambaForPretraining(nn.Module):
    """
    SimMIM 风格的 EarthMamba 预训练封装。

    前向流程
    ────────
    1. 在 patch 网格（Hp × Wp）随机采样布尔掩码 M（mask_ratio=0.6 的位置为 True）
    2. 将输入图像中被掩码 patch 的像素替换为可学习掩码值 mask_value
    3. patch_embed 后加入 2D 可学习位置编码（如遇辅尺度自动 bicubic 插值）
    4. 完整编码器处理含掩码的输入（所有 patch 均参与编码，符合 SimMIM 设计）
    5. 解码器将特征图上采样到输入分辨率，输出像素级预测
    6. 仅在被掩码 patch 区域计算归一化 L1 损失

    可变分辨率机制（推理不锁死输入尺寸）
    ────────────────────────────────────────
    - 2D 可学习位置编码以主分辨率 512 训练（32×32 token 网格 @ patch_size=16）；
      训练时遇到辅尺度批次（如 448 → 28×28）自动 bicubic 插值位置编码，
      让权重从训练阶段就学会尺度适应。
      推理时遇到任意分辨率同样做 bicubic 插值 → 无需微调即可迁移。
      这与 DINOv2 / FlexiViT 的位置编码处理方式一致。
    - patch_embed、downsample 均为纯卷积 → H, W 只需是 patch_size 的整数倍
    - 解码器双线性插值 → 尺寸无关
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

        cfg = self.MODEL_CONFIGS[model_size]
        encoder_dim = cfg["dims"][-1]

        # 主分辨率对应的 token 网格尺寸（位置编码的训练尺寸）
        self.train_grid = primary_size // patch_size   # 512/16 = 32

        # ── 编码器：EarthMamba（posembed=True + 位置编码由本类管理插值）──
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
            posembed=True,            # ★ 开启 2D 可学习位置编码
            imgsize=primary_size,     # ★ 以 512 为训练分辨率初始化 pos_embed
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
            downsample_version="v3",
        )
        del self.encoder.classifier

        # 将 encoder 的 pos_embed 移到本类管理（本类负责插值逻辑）
        # encoder.pos_embed shape: (1, C, train_grid, train_grid)
        self.pos_embed = self.encoder.pos_embed
        self.encoder.pos_embed = None

        self.encoder_dim = encoder_dim

        # ── 可学习掩码值 ──────────────────────────────────────────────
        self.mask_value = nn.Parameter(torch.zeros(1, 3, 1, 1))

        # ── 解码头 ───────────────────────────────────────────────────
        self.decoder = SimMIMDecoder(
            encoder_dim=encoder_dim,
            hidden_dim=decoder_hidden,
        )

        # ── 编码器输出归一化 ──────────────────────────────────────────
        self.enc_norm = nn.GroupNorm(num_groups=32, num_channels=encoder_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.normal_(self.mask_value, std=0.02)

    # ------------------------------------------------------------------
    def _interpolate_pos_embed(self, Hp: int, Wp: int) -> torch.Tensor:
        """
        将训练分辨率的 2D 位置编码 bicubic 插值到目标 token 网格 (Hp, Wp)。

        self.pos_embed 形状: (1, C, train_grid, train_grid)   [channel-first]
        返回形状:             (1, C, Hp, Wp)

        训练时 85% 批次 Hp=Wp=train_grid → 直接返回（无额外计算）；
        辅尺度批次（如 448 → Hp=28）→ bicubic 插值，让位置编码在训练阶段
        就学习到尺度适应能力。
        推理时遇到任意分辨率走同一条路径 → 不锁死分辨率。
        """
        pos_embed = self.pos_embed                       # (1, C, Hg, Wg)
        Hg, Wg = pos_embed.shape[2], pos_embed.shape[3]
        if Hp == Hg and Wp == Wg:
            return pos_embed
        return F.interpolate(
            pos_embed, size=(Hp, Wp),
            mode="bicubic", align_corners=False,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_mask(
        self, B: int, Hp: int, Wp: int, device: torch.device
    ) -> torch.BoolTensor:
        """
        生成 patch 级别随机布尔掩码。

        Args:
            B, Hp, Wp : batch size、patch 行数、patch 列数
        Returns:
            mask : (B, Hp, Wp) bool，True 表示该 patch 被掩码
        """
        N = Hp * Wp
        n_mask = max(1, int(N * self.mask_ratio))
        # 每个样本独立采样，保证随机性
        noise = torch.rand(B, N, device=device)
        ids   = torch.argsort(noise, dim=1)                 # (B, N)
        mask  = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids[:, :n_mask], True)
        return mask.view(B, Hp, Wp)

    # ------------------------------------------------------------------
    def _apply_mask_to_pixels(
        self, imgs: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        将被掩码 patch 的像素替换为可学习掩码值。

        Args:
            imgs : (B, 3, H, W) 归一化图像
            mask : (B, Hp, Wp) bool，Hp = H // patch_size

        Returns:
            masked_imgs : (B, 3, H, W) 同尺寸，被掩码区域已替换
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        Hp, Wp = H // p, W // p

        # 将掩码从 patch 网格上采样到像素空间 (B, Hp, Wp) → (B, 1, H, W)
        mask_pixel = mask.float().unsqueeze(1)              # (B, 1, Hp, Wp)
        mask_pixel = F.interpolate(
            mask_pixel, size=(H, W), mode="nearest"
        )                                                    # (B, 1, H, W)

        # mask_value 广播到 (B, 3, H, W)
        mv = self.mask_value.expand(B, -1, -1, -1)         # (B, 3, 1, 1)
        mv = mv.expand_as(imgs)

        masked = torch.where(mask_pixel.bool(), mv, imgs)
        return masked

    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅前向通过编码器（不计算损失），用于推理 / 下游微调特征提取。

        位置编码处理逻辑：
          - patch_embed 输出的 token 网格 (Hp, Wp) 可能与训练网格 (train_grid, train_grid) 不同
          - 主分辨率 512 → Hp=Wp=32 与训练网格一致 → 直接加
          - 辅尺度 / 推理分辨率 → bicubic 插值位置编码到 (Hp, Wp) → 再加

        Args:
            x : (B, 3, H, W)，H 和 W 须是 patch_size 的整数倍

        Returns:
            feat : (B, encoder_dim, Hf, Wf)
        """
        x = self.encoder.patch_embed(x)         # channel-last: (B, Hp, Wp, C0)

        # ── 注入位置编码（含 bicubic 插值）────────────────────────────
        if self.pos_embed is not None:
            if self.encoder.channel_first:
                _, _, Hp, Wp = x.shape
            else:
                _, Hp, Wp, _ = x.shape
            pos = self._interpolate_pos_embed(Hp, Wp)
            if not self.encoder.channel_first:
                pos = pos.permute(0, 2, 3, 1)   # (1, C, Hp, Wp) → (1, Hp, Wp, C)
            x = x + pos

        for layer in self.encoder.layers:
            x = layer(x)
        if not self.encoder.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.enc_norm(x)
        return x

    # ------------------------------------------------------------------
    def forward(
        self, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        预训练前向传播。

        Args:
            imgs : (B, 3, H, W) 归一化图像，H = W ∈ {primary, multi-scale}

        Returns:
            loss : 标量，SimMIM 归一化 L1 损失
            mask : (B, Hp, Wp) 布尔掩码（用于可视化 / 监控）
        """
        B, C, H, W = imgs.shape
        p  = self.patch_size
        Hp, Wp = H // p, W // p

        assert H % p == 0 and W % p == 0, (
            f"输入尺寸 ({H}, {W}) 不是 patch_size={p} 的整数倍"
        )

        # 1. 生成掩码
        mask = self._generate_mask(B, Hp, Wp, imgs.device)   # (B, Hp, Wp)

        # 2. 在像素空间应用掩码（被掩码区域替换为可学习掩码值）
        masked_imgs = self._apply_mask_to_pixels(imgs, mask)  # (B, 3, H, W)

        # 3. 编码器处理含掩码的图像（全量 patch 均参与编码）
        feat = self.forward_features(masked_imgs)              # (B, C_enc, Hf, Wf)

        # 4. 解码到像素空间
        pred = self.decoder(feat, H, W)                       # (B, 3, H, W)

        # 5. 计算 SimMIM 损失（仅在掩码 patch 区域）
        loss = self._compute_loss(pred, imgs, mask)

        return loss, mask

    # ------------------------------------------------------------------
    def _compute_loss(
        self,
        pred  : torch.Tensor,    # (B, 3, H, W)
        target: torch.Tensor,    # (B, 3, H, W)  原始归一化图像
        mask  : torch.BoolTensor # (B, Hp, Wp)
    ) -> torch.Tensor:
        """
        在掩码 patch 区域计算归一化 L1 损失。

        norm_pix_loss=True 时对每个 patch 的像素做 zero-mean unit-var 归一化，
        抑制低频偏差，使模型专注学习纹理细节（来自 MAE / SimMIM 论文）。
        """
        p  = self.patch_size
        B, C, H, W = pred.shape
        Hp, Wp = H // p, W // p

        # 将预测和目标 reshape 到 patch 级别 (B, Hp, Wp, C*p*p)
        def patchify(x):
            # x: (B, C, H, W) → (B, Hp, Wp, C*p*p)
            x = x.reshape(B, C, Hp, p, Wp, p)
            x = x.permute(0, 2, 4, 1, 3, 5)     # (B, Hp, Wp, C, p, p)
            return x.reshape(B, Hp, Wp, C * p * p)

        pred_p   = patchify(pred)     # (B, Hp, Wp, C*p*p)
        target_p = patchify(target)   # (B, Hp, Wp, C*p*p)

        if self.norm_pix_loss:
            mean = target_p.mean(dim=-1, keepdim=True)
            var  = target_p.var(dim=-1, keepdim=True, unbiased=False)
            target_p = (target_p - mean) / (var + 1e-6).sqrt()

        # mask: (B, Hp, Wp) → 用于索引
        loss_all = F.l1_loss(pred_p, target_p, reduction="none")  # (B, Hp, Wp, C*p*p)
        loss_all = loss_all.mean(dim=-1)                           # (B, Hp, Wp)

        # 仅在掩码区域求平均
        loss = (loss_all * mask.float()).sum() / (mask.float().sum() + 1e-6)
        return loss


# ══════════════════════════════════════════════════════════════════════════════
#  第四节：训练工具函数
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
    """滑动平均计数器。"""
    def __init__(self):  self.reset()
    def reset(self):     self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def is_master() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def setup_ddp() -> Tuple[int, bool]:
    """初始化 DDP；非分布式环境时返回 (0, False)。"""
    if "RANK" not in os.environ:
        return 0, False
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, True


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    args: argparse.Namespace,
    use_ddp: bool,
):
    """保存完整训练状态（用于断点续训）。"""
    raw = model.module if use_ddp else model
    ckpt = {
        "epoch"    : epoch,
        "model"    : raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args"     : vars(args),
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)


def save_backbone_weights(path: Path, model: nn.Module, use_ddp: bool):
    """
    仅保存骨干编码器权重（下游任务微调时加载）。

    pos_embed 在 EarthMambaForPretraining 中由本类管理（self.pos_embed），
    保存时需映射回 EarthMamba 的 key 约定（pos_embed，无前缀）。

    使用方法：
        backbone = BackboneEarthMamba(
            depths=..., dims=..., patch_size=16,
            posembed=True, imgsize=512, ...
        )
        backbone.load_state_dict(
            torch.load('backbone_ep0800.pth'), strict=False
        )
    """
    raw = model.module if use_ddp else model
    sd = {}
    for k, v in raw.state_dict().items():
        if k.startswith("encoder."):
            sd[k[len("encoder."):]] = v
        elif k == "pos_embed":
            sd["pos_embed"] = v
    torch.save(sd, path)


# ══════════════════════════════════════════════════════════════════════════════
#  第五节：训练主循环
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    optimizer : torch.optim.Optimizer,
    scheduler,
    scaler    : Optional[GradScaler],
    epoch     : int,
    args      : argparse.Namespace,
    logger    : logging.Logger,
) -> float:
    model.train()
    loss_meter  = AverageMeter()
    accum_steps = args.grad_accum
    amp_dtype   = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    optimizer.zero_grad()

    for step, imgs in enumerate(loader):
        imgs = imgs.cuda(non_blocking=True)      # (B, 3, H, W)，H 动态变化

        # ── 前向（AMP 上下文）────────────────────────────────────────────
        with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=amp_dtype):
            loss, mask = model(imgs)
            loss = loss / accum_steps

        # ── 反向传播 ─────────────────────────────────────────────────────
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ── 梯度更新（含梯度累积）────────────────────────────────────────
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if args.clip_grad > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── 统计（all-reduce 同步各卡损失）──────────────────────────────
        loss_val = loss.item() * accum_steps
        if dist.is_initialized():
            lt = torch.tensor(loss_val, device="cuda")
            dist.all_reduce(lt, op=dist.ReduceOp.AVG)
            loss_val = lt.item()
        loss_meter.update(loss_val)

        # ── 日志 ─────────────────────────────────────────────────────────
        if step % args.log_every == 0 and is_master():
            lr_now   = scheduler.get_last_lr()[0]
            res      = imgs.shape[-1]
            mask_pct = mask.float().mean().item() * 100
            logger.info(
                f"Ep[{epoch+1:03d}/{args.epochs}] "
                f"Step[{step:05d}/{len(loader)}] "
                f"loss={loss_meter.avg:.4f}  "
                f"lr={lr_now:.2e}  "
                f"res={res}  "
                f"masked={mask_pct:.1f}%"
            )

    return loss_meter.avg


# ══════════════════════════════════════════════════════════════════════════════
#  第六节：命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "EarthMamba SimMIM 预训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 数据 ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("数据")
    g.add_argument("--data_root",   required=True,  help="512×512 图像根目录（递归扫描）")
    g.add_argument("--output_dir",  default="./pretrain_output")
    g.add_argument("--num_workers", type=int, default=8)

    # ── 分辨率策略 ────────────────────────────────────────────────────────────
    g = p.add_argument_group("多尺度分辨率")
    g.add_argument("--primary_size", type=int,   default=512,
                   help="主分辨率（图像已预处理到此尺寸）")
    g.add_argument("--patch_size",   type=int,   default=16,
                   help="Patch 边长；需整除 primary_size；推荐 16")
    g.add_argument("--primary_prob", type=float, default=0.85,
                   help="使用主分辨率的批次比例 [0.80, 0.90]")
    g.add_argument("--min_scale",    type=float, default=0.85,
                   help="辅尺度下界缩放因子（用于自动生成离散候选表）")
    g.add_argument("--max_scale",    type=float, default=1.15,
                   help="辅尺度上界缩放因子")
    g.add_argument("--aux_scales",   type=int, nargs="*", default=None,
                   help="手动指定辅尺度候选列表（优先于 min/max_scale 自动生成）"
                        "例: --aux_scales 448 480 544 576")
    g.add_argument("--color_jitter", type=float, default=0.2,
                   help="颜色抖动强度（遥感建议 0.1~0.2，0 = 关闭）")
    g.add_argument("--vflip_prob",   type=float, default=0.5,
                   help="随机垂直翻转概率（遥感俯视图通常合法）")
    g.add_argument("--hflip_prob",   type=float, default=0.5,
                   help="随机水平翻转概率")

    # ── 模型 ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("模型")
    g.add_argument("--model_size",    default="small",
                   choices=["tiny", "small", "base", "large"])
    g.add_argument("--mask_ratio",    type=float, default=0.60,
                   help="SimMIM 掩码比例")
    g.add_argument("--norm_pix_loss", action="store_true", default=True,
                   help="对 patch 像素归一化再计算损失")
    g.add_argument("--decoder_hidden",type=int,   default=512,
                   help="解码头隐层通道数")
    g.add_argument("--drop_path",     type=float, default=0.1)
    g.add_argument("--ssm_version",   default="mamba1", choices=["mamba1", "mamba3"])
    g.add_argument("--ssm_headdim",   type=int,   default=64)
    g.add_argument("--ssm_backend",   default=None,
                   help="SSM CUDA 后端；None=自动；可选 'torch_easy' 调试")
    g.add_argument("--use_checkpoint",action="store_true", default=False,
                   help="梯度检查点（节省显存，略慢）")

    # ── 训练超参 ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("训练")
    g.add_argument("--epochs",        type=int,   default=800)
    g.add_argument("--warmup_epochs", type=int,   default=40)
    g.add_argument("--batch_size",    type=int,   default=16,
                   help="单卡 batch size")
    g.add_argument("--grad_accum",    type=int,   default=1,
                   help="梯度累积步数（等效扩大 batch）")
    g.add_argument("--lr",            type=float, default=1.5e-4,
                   help="base lr（实际 lr = lr * batch_size_eff / 256）")
    g.add_argument("--auto_scale_lr", action="store_true", default=True,
                   help="按有效 batch size 线性缩放 lr")
    g.add_argument("--weight_decay",  type=float, default=0.05)
    g.add_argument("--beta1",         type=float, default=0.9)
    g.add_argument("--beta2",         type=float, default=0.95)
    g.add_argument("--clip_grad",     type=float, default=1.0)
    g.add_argument("--use_amp",       action="store_true", default=True)
    g.add_argument("--amp_dtype",     default="bf16", choices=["bf16", "fp16"])
    g.add_argument("--fsdp",          action="store_true", default=False,
                   help="使用 FSDP FULL_SHARD（超大模型 / 超多卡时开启）")

    # ── 杂项 ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("杂项")
    g.add_argument("--seed",       type=int, default=42)
    g.add_argument("--log_every",  type=int, default=50,  help="每 N step 打印一次日志")
    g.add_argument("--save_every", type=int, default=50,  help="每 N epoch 保存一次 checkpoint")
    g.add_argument("--resume",     default=None,           help="从 checkpoint 恢复训练")

    return p.parse_args()


def main():
    args = build_args()

    # ── DDP 初始化 ────────────────────────────────────────────────────────────
    local_rank, use_ddp = setup_ddp()
    world_size = dist.get_world_size() if use_ddp else 1
    rank       = dist.get_rank()       if use_ddp else 0

    # ── 日志 ─────────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO if is_master() else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("pretrain")

    # ── 随机种子（各卡略有不同保证数据多样性）────────────────────────────────
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    # ── 输出目录 ──────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    if is_master():
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── 有效 batch size 与 lr 自动缩放 ───────────────────────────────────────
    eff_batch = args.batch_size * world_size * args.grad_accum
    if args.auto_scale_lr:
        # linear scaling rule: lr = base_lr * eff_batch / 256
        args.lr = args.lr * eff_batch / 256
        if is_master():
            logger.info(f"线性缩放 LR: {args.lr:.2e}  (eff_batch={eff_batch})")

    # ── 离散辅尺度候选表 ────────────────────────────────────────────────────
    if args.aux_scales is not None:
        candidates = sorted(args.aux_scales)
    else:
        candidates = build_scale_candidates(
            primary=args.primary_size,
            patch_size=args.patch_size,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )

    # ── 数据集 & DataLoader ───────────────────────────────────────────────────
    dataset = PretrainDataset(root=args.data_root, primary_size=args.primary_size)
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    collator = MultiScaleCollator(
        primary=args.primary_size,
        patch_size=args.patch_size,
        primary_prob=args.primary_prob,
        candidates=candidates,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        color_jitter=args.color_jitter,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
        persistent_workers=(args.num_workers > 0),
    )

    if is_master():
        all_scales = sorted([args.primary_size] + candidates)
        logger.info(
            f"数据集: {len(dataset)} 张图像 | "
            f"Loader: {len(loader)} steps/epoch\n"
            f"  主分辨率: {args.primary_size}  (P={args.primary_prob:.0%})\n"
            f"  辅尺度候选（patch_size={args.patch_size} 整数倍）: {candidates}\n"
            f"  完整合法尺度表: {all_scales}"
        )

    # ── 模型 ─────────────────────────────────────────────────────────────────
    model = EarthMambaForPretraining(
        model_size=args.model_size,
        patch_size=args.patch_size,
        primary_size=args.primary_size,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
        decoder_hidden=args.decoder_hidden,
        drop_path_rate=args.drop_path,
        ssm_version=args.ssm_version,
        ssm_headdim=args.ssm_headdim,
        ssm_backend=args.ssm_backend,
        use_checkpoint=args.use_checkpoint,
    ).cuda()

    if is_master():
        total_params   = sum(p.numel() for p in model.parameters()) / 1e6
        encoder_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
        logger.info(
            f"模型参数量: 总计 {total_params:.1f}M  "
            f"(编码器 {encoder_params:.1f}M + 解码头 {total_params-encoder_params:.1f}M)"
        )

    # ── DDP / FSDP 封装 ───────────────────────────────────────────────────────
    fsdp_used = False
    if use_ddp and args.fsdp and world_size > 1:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            import functools
            auto_wrap = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=frozenset({EarthMambaBlock}),
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
                logger.info("已启用 FSDP FULL_SHARD")
        except Exception as e:
            if is_master():
                logger.warning(f"FSDP 初始化失败，回退 DDP: {e}")

    if use_ddp and not fsdp_used:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        if is_master():
            logger.info("已启用 DDP")

    # ── 优化器（编码器 / 解码器分组，解码器 lr 略高）────────────────────────
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
            {"params": decoder_params, "lr": args.lr * 2,   # 解码头学习更快
             "weight_decay": 0.0},                           # 解码头不用 WD
        ],
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    scheduler = build_cosine_schedule(
        optimizer, args.warmup_epochs, args.epochs, len(loader)
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
            scaler = torch.amp.GradScaler("cuda")

    # ── 断点恢复 ──────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        (model.module if (use_ddp and not fsdp_used) else model).load_state_dict(
            ckpt["model"]
        )
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        if is_master():
            logger.info(f"从 epoch {start_epoch} 恢复训练")

    # ── 训练主循环 ────────────────────────────────────────────────────────────
    train_log = []
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        t0 = time.time()
        avg_loss = train_one_epoch(
            model, loader, optimizer, scheduler, scaler,
            epoch, args, logger,
        )
        elapsed = time.time() - t0

        if is_master():
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                f"── Epoch {epoch+1:03d}/{args.epochs}  "
                f"avg_loss={avg_loss:.4f}  "
                f"lr={lr_now:.2e}  "
                f"time={elapsed:.0f}s"
            )
            train_log.append({"epoch": epoch + 1, "loss": avg_loss, "lr": lr_now})

            # 保存 checkpoint
            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
                ckpt_path = out_dir / f"ckpt_ep{epoch+1:04d}.pth"
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, scaler,
                    epoch, args, use_ddp and not fsdp_used,
                )
                logger.info(f"Checkpoint → {ckpt_path}")

                # 仅保存骨干权重（下游微调用）
                bb_path = out_dir / f"backbone_ep{epoch+1:04d}.pth"
                save_backbone_weights(bb_path, model, use_ddp and not fsdp_used)
                logger.info(f"Backbone   → {bb_path}")

    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    if is_master():
        log_path = out_dir / "pretrain_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "args"      : vars(args),
                "train_log" : train_log,
                "world_size": world_size,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"训练日志 → {log_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
