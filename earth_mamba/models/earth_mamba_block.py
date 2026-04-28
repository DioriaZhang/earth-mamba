import inspect
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

from ..utils.layer_utils import LayerNorm, Mlp
from ..modules.spatial_sparse import SparseSS2D
from ..modules.armg import ARMG
from ..modules.latent_graph import LatentGraph


class EarthMambaBlock(nn.Module):
    """
    Earth-Mamba 基础块：LayerNorm → **Noise-robust**（ARMG）→ **Compression-aware** 稀疏 SSM（Path A）
    + **语义图**（Path B，与多向 scan 互补）→ 残差 → MLP。
    """
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        channel_first: bool = False,

        # -------- SSM --------
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0.0,
        ssm_init="v0",

        # -------- MLP --------
        mlp_ratio: float = 4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,

        # -------- control --------
        use_checkpoint: bool = False,

        # -------- ablation switches --------
        ssm_cls=None,           # default: SparseSS2D
        use_armg: bool = True,
        use_graph: bool = True,
        graph_num_nodes: int = 64,
        ssm_backend=None,
        ssm_headdim: int = 64,  # Mamba-3 专用：每头维度；SparseSS2D 通过 **kwargs 忽略
        **kwargs,
    ):
        super().__init__()

        self.channel_first = channel_first
        self.use_checkpoint = use_checkpoint

        # ======== Norm ========
        self.norm1 = LayerNorm(hidden_dim, channel_first=channel_first)
        self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)

        # ======== ARMG (always built) ========
        self.armg = ARMG(dim=hidden_dim, init_value=5.0)
        self.armg_enable = float(use_armg)   # buffer-like scalar

        # ======== Sparse SSM (Path A) ========
        OpClass = ssm_cls if ssm_cls is not None else SparseSS2D
        _sig = inspect.signature(OpClass.__init__).parameters
        ssm_kw = dict(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            channel_first=channel_first,
        )
        if "ssm_backend" in _sig:
            ssm_kw["ssm_backend"] = ssm_backend
        if "headdim" in _sig:
            # SS2D_Mamba3 专用：传入 headdim（SparseSS2D 没有此参数，故检查后传）
            ssm_kw["headdim"] = ssm_headdim
        self.ssm = OpClass(**ssm_kw)

        # ======== Graph module (Path B, always built) ========
        self.graph = LatentGraph(dim=hidden_dim, num_nodes=graph_num_nodes)
        self.graph_lambda = nn.Parameter(torch.zeros(1))
        self.graph_enable = float(use_graph)

        # ======== MLP ========
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate,
            channel_first=channel_first,
        )

        self.drop_path = DropPath(drop_path)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # ======== Earth-Mamba branch (compression-aware SSM + semantic graph) ========
        z = self.norm1(x)

        # --- ARMG (always executed, gated by scalar) ---
        armg_out = self.armg(z)
        z = z + self.armg_enable * (armg_out - z)

        # --- Path A: Sparse SSM ---
        y_sparse = self.ssm(z)

        # --- Path B: Graph ---
        y_graph = self.graph(z)
        y_dual = y_sparse + self.graph_enable * self.graph_lambda * y_graph

        x = x + self.drop_path(y_dual)

        # ======== MLP branch ========
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward_impl, x, use_reentrant=True)
        else:
            return self._forward_impl(x)

