"""三模块组合消融（A=Sparse SSM, B=Graph, C=ARMG）。

推理时通过 block 内门控插拔（模块仍构建、权重仍加载）：
  - sparse_enable  → Path A（SSM 支路残差）
  - graph_enable   → Path B（Latent Graph）
  - armg_enable    → ARMG 降噪
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

# (use_sparse_ssm, use_graph, use_armg) 对应 A, B, C
class ModuleFlags(NamedTuple):
    sparse: bool
    graph: bool
    armg: bool


FLAGS: dict[str, ModuleFlags] = {
    "ID1": ModuleFlags(False, False, False),  # Pure baseline
    "ID3": ModuleFlags(True, False, False),   # A only
    "ID4": ModuleFlags(False, True, False),   # B only
    "ID5": ModuleFlags(False, False, True),   # C only
    "ID6": ModuleFlags(True, True, False),    # A+B
    "ID7": ModuleFlags(True, False, True),    # A+C
    "ID8": ModuleFlags(False, True, True),    # B+C
    "ID9": ModuleFlags(True, True, True),     # Full（引用主表）
}

LOGIC: dict[str, str] = {
    "ID1": "原始基线（Pure Baseline）",
    "ID3": "单模块 A",
    "ID4": "单模块 B",
    "ID5": "单模块 C",
    "ID6": "A+B：双流互补（稀疏 + 结构）",
    "ID7": "A+C：稀疏感知 + 降噪",
    "ID8": "B+C：结构感知 + 降噪",
    "ID9": "完整",
}

TABLE_COLUMNS: dict[str, dict[str, bool]] = {
    vid: {"sparse": f.sparse, "graph": f.graph, "armg": f.armg}
    for vid, f in FLAGS.items()
}

LABELS: dict[str, str] = {vid: f"ID{vid[2:]}" for vid in FLAGS}

VARIANT_ORDER = ["ID1", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8", "ID9"]
RUN_VARIANTS = ["ID1", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8"]

REFERENCE_METRICS = {
    "inria_val_miou": 86.64,
    "dfc15_macro_map": 97.54,
    "dior_map50": 66.27,
}


def resolve_flags(
    variant: Optional[str] = None,
    *,
    no_sparse: bool = False,
    no_graph: bool = False,
    no_armg: bool = False,
) -> Tuple[bool, bool, bool]:
    """返回 (use_sparse_ssm, use_graph, use_armg)。"""
    if variant:
        if variant not in FLAGS:
            raise ValueError(f"Unknown ablation_variant: {variant}")
        f = FLAGS[variant]
        return f.sparse, f.graph, f.armg
    return (not no_sparse, not no_graph, not no_armg)


def should_run_training(variant: Optional[str]) -> bool:
    return variant is not None and variant != "ID9"
