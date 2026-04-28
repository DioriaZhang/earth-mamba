# earth_mamba/modules/__init__.py
# 不在包初始化时导入 Mamba3 / SS2D_Mamba3：会链式加载 Triton mamba3 核，其模块级
# triton.set_allocator 在旧版 Triton 上会报错；训练脚本为 FSDP 仅 import EarthMambaBlock
# 时不应触发上述副作用。
from .armg import ARMG
from .latent_graph import LatentGraph
from .spatial_sparse import SparseSS2D

__all__ = ["ARMG", "LatentGraph", "SparseSS2D", "Mamba3", "SS2D_Mamba3"]


def __getattr__(name):
    if name == "Mamba3":
        from .mamba3 import Mamba3
        return Mamba3
    if name == "SS2D_Mamba3":
        from .ss2d_mamba3 import SS2D_Mamba3
        return SS2D_Mamba3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
