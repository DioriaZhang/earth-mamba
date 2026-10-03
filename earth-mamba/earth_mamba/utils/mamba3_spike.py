"""
Mamba-3 接入模块（已激活）。

earth-mamba 通过 SS2D_Mamba3 接入官方 Mamba-3（mamba_ssm >= 2.3.1）。

使用路径：
  EarthMamba(ssm_version="mamba3", ssm_headdim=64, ssm_d_state=64)
  → EarthMambaBlock(ssm_cls=SS2D_Mamba3)
  → SS2D_Mamba3 内部懒导入 mamba_ssm.modules.mamba3.Mamba3（SISO）

技术特性（相对 Mamba-1）：
  - 指数梯形离散化（Exponential Trapezoidal）替代 ZOH，数值更稳定
  - RoPE（旋转位置编码）增强位置感知
  - chunk-wise Triton 核并行化，吞吐优于逐 token 递推
  - SISO 模式（非 MIMO）：适合视觉 backbone fine-tuning

参考：
  论文: https://arxiv.org/pdf/2603.15569
  代码: https://github.com/state-spaces/mamba  (mamba_ssm/modules/mamba3.py)
"""

from .vmamba_core import selective_scan_torch_easy  # Mamba-1 fallback（保留兼容性）
from ..modules.ss2d_mamba3 import SS2D_Mamba3

__all__ = ["selective_scan_torch_easy", "SS2D_Mamba3"]
