"""
SS2D_Mamba3：Mamba-3 SISO + 四向 cross-scan 2D 空间适配层。

作为 SparseSS2D 的直接替换，在 EarthMambaBlock 内使用 Mamba-3 的
指数梯形离散化（Exponential Trapezoidal）+ RoPE + Triton SISO 核，
同时通过四向 cross-scan 保持对 2D 空间特征图的多方向感知。

依赖：
  - triton >= 3.0（Triton SISO 核，无需编译 CUDA 扩展）
  - einops

Mamba-3 SISO 核已内嵌在 earth_mamba.ops.triton 中，无需外部 mamba_ssm 包。
"""

import warnings
import math

import torch
import torch.nn as nn

from .mamba3 import Mamba3
from ..utils.vmamba_core import cross_scan_fn, cross_merge_fn


class SS2D_Mamba3(nn.Module):
    """
    Mamba-3 SISO + 四向 cross-scan 2D 适配器。

    用作 EarthMambaBlock 中 SparseSS2D 的替换（接口兼容）：
      输入/输出均为 (B, H, W, C) [channel_last] 或 (B, C, H, W) [channel_first]

    数据流::

        (B, H, W, C)
          ↓ permute → (B, C, H, W)
          ↓ cross_scan_fn → (B, 4, C, L)       四向序列化
          ↓ reshape → (B*4, L, C)               合并 batch 与扫描方向
          ↓ Mamba3 SISO → (B*4, L, C)          Mamba-3 核（Triton）
          ↓ reshape → (B, 4, C, L)
          ↓ cross_merge_fn → (B, C, H, W)      四向结果合并
          ↓ permute → (B, H, W, C)

    Args:
        d_model (int): 输入通道数（来自 EarthMamba 各阶段的 dim）。
        d_state (int): SSM 状态维度（per-head），Mamba-3 建议 64–128。
        ssm_ratio (float): 内部扩展比 expand = d_inner / d_model，通常 2.0。
        headdim (int): 每个头的维度，需满足 d_inner % headdim == 0。
            若不满足则自动降级到最近合法值 [64, 32, 16, 8]。
        ngroups (int): B/C 的 group 数（保持 1 即 SISO）。
        channel_first (bool): 输入是否为 (B, C, H, W) 格式。
        ssm_backend (str|None): 兼容接口参数，Mamba-3 忽略此值（始终用 Triton）。
        initialize (str): 兼容接口参数，不影响 Mamba-3 权重初始化。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        ssm_ratio: float = 2.0,
        headdim: int = 64,
        ngroups: int = 1,
        channel_first: bool = False,
        initialize: str = "v0",
        ssm_backend=None,
        dropout: float = 0.0,
        dt_rank="auto",
        act_layer=None,
        d_conv: int = 3,
        conv_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.channel_first = channel_first

        d_inner = int(ssm_ratio * d_model)

        if d_inner % headdim != 0:
            _original = headdim
            for candidate in [64, 32, 16, 8, 4]:
                if d_inner % candidate == 0:
                    headdim = candidate
                    break
            warnings.warn(
                f"SS2D_Mamba3: d_inner={d_inner} 不能被 headdim={_original} 整除，"
                f"已自动调整 headdim → {headdim}。",
                stacklevel=2,
            )

        # chunk_size=16 对常见配置（patch=16, img=224/256）均可整除
        chunk_size = 16

        self.mamba3 = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=ssm_ratio,
            headdim=headdim,
            ngroups=ngroups,
            is_mimo=False,
            chunk_size=chunk_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_first:
            x = x.permute(0, 2, 3, 1).contiguous()

        B, H, W, C = x.shape
        L = H * W

        x_chw = x.permute(0, 3, 1, 2).contiguous()
        y_scan = cross_scan_fn(x_chw)                 # (B, 4, C, L)

        x_seq = y_scan.permute(0, 1, 3, 2).reshape(B * 4, L, C)
        y_seq = self.mamba3(x_seq)                    # (B*4, L, C)

        y_global = y_seq.reshape(B, 4, L, C).permute(0, 1, 3, 2).contiguous()
        y = cross_merge_fn(y_global)                  # (B, C, H, W)

        y = y.permute(0, 2, 3, 1).contiguous()

        if self.channel_first:
            y = y.permute(0, 3, 1, 2).contiguous()

        return y
