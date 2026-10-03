"""
Mamba-3 SS2D 模块：将 Mamba-3（指数梯形离散化 + RoPE）接入 Earth-Mamba 的空间稀疏 2D SSM 框架。

核心设计
--------
1. 保留 SparseSS2D 的四向 cross-scan（Triton cross_scan_fn / cross_merge_fn）
2. 保留 Sparsemax 空间稀疏门控（对 K / B 矩阵进行硬稀疏门控）
3. 以 mamba3_siso_combined（Triton, BF16 原生）替换 Mamba-1 的 selective_scan_fn
   - 指数梯形离散化（Exponential Trapezoidal），数值稳定性优于 ZOH
   - RoPE 旋转位置编码（作用于 B/C），增强位置感知
4. 若 mamba_ssm 未安装，__init__ 时主动报错并给出安装指引

参数约束（来自 mamba3_siso_combined Triton kernel）
-------------------------------------------------------
- headdim（headdim_v）和 d_state（headdim_qk）必须是 2 的幂（如 headdim=64, d_state=64）
- d_inner 必须整除 headdim：nheads = d_inner // headdim
- num_rope_angles ≤ d_state // 2 且为偶数
- kernel 强制 BF16 输入，建议全局训练精度使用 BF16（--amp --amp_dtype bf16）
- 推荐 A100 / H100（TMA 在 H100 更快，A100 亦支持）

依赖安装
--------
在服务器上首次运行前：
    cd <project_root>/mamba && pip install -e .
"""

import sys
import os
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.vmamba_core import cross_scan_fn, cross_merge_fn


# ─── 将本地 mamba 仓库（state-spaces/mamba）加入 sys.path ──────────────────────
def _try_add_mamba_to_path():
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, "..", "..", ".."))  # project_root
    for _candidate in (_root, os.path.join(_root, "mamba")):
        if os.path.isdir(os.path.join(_candidate, "mamba_ssm")) and _candidate not in sys.path:
            sys.path.insert(0, _candidate)
            return
    _mamba_dir = os.path.join(_root, "mamba")
    if os.path.isdir(_mamba_dir) and _mamba_dir not in sys.path:
        sys.path.insert(0, _mamba_dir)


_try_add_mamba_to_path()

try:
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False
    mamba3_siso_combined = None  # type: ignore
    RMSNormGated = None          # type: ignore


# ─── Sparsemax（与 spatial_sparse.py 中相同，避免循环导入时重复定义）──────────────
class _Sparsemax(nn.Module):
    """Sparsemax 激活：将输入投影到概率单纯形并可输出精确零值（硬稀疏）。"""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        n = x.size(dim)
        x = x - x.max(dim=dim, keepdim=True)[0].expand_as(x)
        zs = torch.sort(x, dim=dim, descending=True)[0]
        rng = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
        rng = rng.view([1] * (x.ndim - 1) + [n] if dim == -1 else
                       [1 if i != dim else n for i in range(x.ndim)])
        rng = rng.expand_as(zs)
        bound = 1 + rng * zs
        cumsum = torch.cumsum(zs, dim=dim)
        is_gt = bound > cumsum
        k = torch.max(is_gt * rng, dim, keepdim=True)[0]
        taus = (torch.sum(is_gt * zs, dim, keepdim=True) - 1) / k
        return torch.clamp(x - taus.expand_as(x), min=0.0)


# ─── Mamba3SS2D ───────────────────────────────────────────────────────────────

class Mamba3SS2D(nn.Module):
    """
    四向 Mamba-3 2D SSM 分支（Path A 升级版）：

    - cross_scan_fn：Triton 四向扫描，(B,C,H,W) → (B,4,C,L)
    - Sparsemax 空间门控：对 K（= B matrix）进行硬稀疏，过滤无关像素
    - mamba3_siso_combined：Triton BF16 kernel
        * 指数梯形离散化（比 ZOH 更稳定）
        * RoPE 旋转位置编码（增强位置感知）
        * GQA（ngroups=1 即 SISO 模式）

    与 SparseSS2D 接口完全兼容（drop-in 替换），通过 ssm_cls=Mamba3SS2D 即可启用。
    """

    def __init__(
        self,
        d_model: int,
        # ── Mamba-3 核心参数 ──────────────────────────────────────────────────
        d_state: int = 64,           # headdim_qk（状态/Q/K 头维度），必须是 2 的幂
        ssm_ratio: float = 2.0,
        headdim: int = 64,           # headdim_v（V 头维度），必须是 2 的幂且整除 d_inner
        ngroups: int = 1,            # Q/K 头数（SISO: 1，GQA: >1）
        rope_fraction: float = 0.5,  # RoPE 角度占 d_state 的比例
        chunk_size: int = 64,        # kernel chunk 大小，推荐 64
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        A_floor: float = 1e-4,       # A 的下限（防止 A→0）
        # ── 共用参数（接口兼容 SparseSS2D）────────────────────────────────────
        act_layer=nn.SiLU,
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        ssm_backend=None,            # 保留接口，此处不使用
        channel_first: bool = False,
        initialize: str = "v0",      # 保留接口，此处不使用
        dt_rank=None,                # 保留接口（Mamba-1 参数），此处忽略
        **kwargs,
    ):
        if not HAS_MAMBA3:
            raise ImportError(
                "Mamba3SS2D 需要 mamba_ssm（包含 Mamba-3 Triton kernel）。\n"
                "请在服务器上执行：\n"
                "    cd <project_root>/mamba && pip install -e .\n"
                "或：pip install -e D:\\doc\\re\\main\\mamba  (本地路径)"
            )
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.channel_first = channel_first
        self.chunk_size = chunk_size
        self.A_floor = A_floor
        self.ngroups = ngroups

        self.d_inner = int(ssm_ratio * d_model)

        # 约束检查
        if self.d_inner % headdim != 0:
            raise ValueError(
                f"d_inner={self.d_inner} 不能整除 headdim={headdim}。"
                f"请调整 ssm_ratio 或 headdim，使 d_inner % headdim == 0。"
            )
        self.nheads = self.d_inner // headdim
        self.headdim = headdim

        # num_rope_angles 约束：≤ d_state//2，为偶数，至少为 2
        _raw_angles = int(d_state * rope_fraction)
        _raw_angles = (_raw_angles // 2) * 2          # 取偶数
        self.num_rope_angles = max(2, min(_raw_angles, (d_state // 2) // 2 * 2))
        if self.num_rope_angles > d_state // 2:
            raise ValueError(
                f"num_rope_angles={self.num_rope_angles} > d_state//2={d_state//2}，"
                "请减小 rope_fraction 或增大 d_state。"
            )

        # ── 1. 输入投影（与 SparseSS2D 结构相同）──────────────────────────────
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=conv_bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
        )
        self.act = act_layer()

        # ── 2. x_proj：从 d_inner 投影出 C(Q), B(K), dd_dt, dd_A, trap, angles
        # C/B: ngroups * d_state（SISO: ngroups=1）
        # dd_dt, dd_A, trap: nheads each
        # angles: num_rope_angles
        _proj_out = 2 * ngroups * d_state + 3 * self.nheads + self.num_rope_angles
        self.x_proj = nn.Linear(self.d_inner, _proj_out, bias=False)

        # ── 3. Mamba-3 特有参数 ────────────────────────────────────────────────
        # dt_bias：per-head Δt 偏置，确保 Δt 初始化在合理范围
        _dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=1e-4)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias)
        self.dt_bias._no_weight_decay = True  # type: ignore[assignment]

        # B/C 偏置（nheads, d_state）——kernel 内作用于每个 V-head
        self.B_bias = nn.Parameter(
            torch.ones(self.nheads, d_state, dtype=torch.float32)
        )
        self.C_bias = nn.Parameter(
            torch.ones(self.nheads, d_state, dtype=torch.float32)
        )

        # B/C RMSNorm（归一化最后一维 d_state）
        self.B_norm = RMSNormGated(d_state, eps=1e-5)
        self.C_norm = RMSNormGated(d_state, eps=1e-5)

        # D skip connection（per-head）
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True  # type: ignore[assignment]

        # ── 4. Sparsemax 空间门控（保留自 SparseSS2D）─────────────────────────
        # 对每个扫描方向上的序列位置做硬稀疏：不重要的像素贡献为 0
        self.sparse_gate = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 4),
            nn.ReLU(),
            nn.Linear(self.d_inner // 4, 1),
            _Sparsemax(dim=1),     # 在序列长度维 L 上做 sparsemax
        )

        # ── 5. 输出层 ──────────────────────────────────────────────────────────
        # 使用 RMSNormGated 替代 nn.LayerNorm，避免 BF16 autocast 强制 FP32 转换
        # RMSNormGated 是 Triton 实现，在 BF16 下保持原生精度，减少 aten::to cast
        self.out_norm = RMSNormGated(self.d_inner, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=conv_bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 0. channel_first 格式转换 ─────────────────────────────────────────
        if self.channel_first:
            x = x.permute(0, 2, 3, 1).contiguous()   # (B,C,H,W) → (B,H,W,C)

        B, H, W, C = x.shape
        L = H * W

        # ── 1. in_proj → split x, z ───────────────────────────────────────────
        xz = self.in_proj(x)                         # (B,H,W, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)               # each: (B,H,W, d_inner)

        # ── 2. depthwise conv + activation ───────────────────────────────────
        x_conv = x_in.permute(0, 3, 1, 2).contiguous()   # (B, d_inner, H, W)
        x_conv = self.act(self.conv2d(x_conv))             # (B, d_inner, H, W)

        # ── 3. cross_scan: (B, d_inner, H, W) → (B, 4, d_inner, L) ──────────
        y_scan = cross_scan_fn(x_conv)                     # (B, 4, d_inner, L)

        # 展平为序列：(B*4, L, d_inner)
        x_flat = (
            y_scan.permute(0, 1, 3, 2)
            .contiguous()
            .view(B * 4, L, self.d_inner)
        )
        # V（SSM 的 value）：(B*4, L, nheads, headdim)
        V = x_flat.view(B * 4, L, self.nheads, self.headdim)

        # ── 4. x_proj 投影 ────────────────────────────────────────────────────
        proj = self.x_proj(x_flat)   # (B*4, L, proj_dim)

        _ds = self.d_state
        _nh = self.nheads
        _na = self.num_rope_angles
        _ng = self.ngroups

        C_raw, B_raw, dd_dt, dd_A, trap_raw, angle_raw = torch.split(
            proj,
            [_ng * _ds, _ng * _ds, _nh, _nh, _nh, _na],
            dim=-1,
        )
        # C (Query): (B*4, L, ngroups, d_state)
        C_raw = C_raw.view(B * 4, L, _ng, _ds)
        # B (Key):   (B*4, L, ngroups, d_state)
        B_raw = B_raw.view(B * 4, L, _ng, _ds)

        # ── 5. Sparsemax 空间稀疏门控（作用于 B/K）────────────────────────────
        # sparse_gate 输入: (B*4, L, d_inner) → m_t: (B*4, L, 1)
        m_t = self.sparse_gate(x_flat)          # (B*4, L, 1)
        # 广播并门控 B：(B*4, L, 1, 1) × (B*4, L, ng, ds) → (B*4, L, ng, ds)
        B_gated = m_t.unsqueeze(-1) * B_raw

        # ── 6. RMSNorm（Mamba-3 标准归一化 B 和 C）────────────────────────────
        C_normed = self.C_norm(C_raw)            # (B*4, L, ng, d_state)
        B_normed = self.B_norm(B_gated)          # (B*4, L, ng, d_state)

        # ── 7. 计算 DT, ADT, Trap（均在 float32 保证稳定性）────────────────────
        # DT = softplus(dd_dt + dt_bias), shape: (B*4, nheads, L), fp32
        DT = F.softplus(
            dd_dt.float().transpose(1, 2)                      # (B*4, nh, L)
            + self.dt_bias.float().view(1, _nh, 1)             # (1,   nh, 1)
        )

        # _A = -softplus(dd_A), clamped ≤ -A_floor
        _A = -F.softplus(dd_A.float().transpose(1, 2))         # (B*4, nh, L)
        _A = torch.clamp(_A, max=-self.A_floor)
        ADT = _A * DT                                          # (B*4, nh, L), fp32

        # Trap ∈ (0,1): 梯形权重，(B*4, nheads, L)
        Trap = torch.sigmoid(trap_raw).transpose(1, 2)         # (B*4, nh, L)

        # ── 8. Angles：(B*4, L, num_rope_angles) → (B*4, L, nheads, na) ──────
        # 同一 angle 广播给所有 V-head（可改为 per-head angle 以增强表达力）
        Angles = angle_raw.unsqueeze(2).expand(-1, -1, _nh, -1).contiguous()
        # (B*4, L, nheads, num_rope_angles)

        # ── 9. 调用 mamba3_siso_combined（Triton BF16 kernel）─────────────────
        # 张量形状总结：
        #   Q (C_normed) : (B*4, L, ngroups=1, d_state)    [nheads_qk = ngroups]
        #   K (B_normed) : (B*4, L, ngroups=1, d_state)
        #   V            : (B*4, L, nheads,    headdim)
        #   ADT, DT, Trap: (B*4, nheads, L)     [float32]
        #   Q_bias/K_bias: (nheads, d_state)     [float32]
        #   Angles       : (B*4, L, nheads, num_rope_angles)
        #   D            : (nheads,)             [float32]
        #   Z=None        → gating 在 cross_merge 后手动执行
        #
        # kernel 内部会将 Q,K,V,Trap,Angles 强制 cast 到 BF16；
        # ADT,DT,Q_bias,K_bias,D 保持 float32 以保证精度
        y = mamba3_siso_combined(
            Q=C_normed,
            K=B_normed,
            V=V,
            ADT=ADT,
            DT=DT,
            Trap=Trap,
            Q_bias=self.C_bias,    # (nheads, d_state), fp32
            K_bias=self.B_bias,    # (nheads, d_state), fp32
            Angles=Angles,
            D=self.D.float(),      # (nheads,), fp32
            Z=None,                # gating 在外部手动应用
            chunk_size=self.chunk_size,
        )
        # y: (B*4, L, nheads, headdim)

        # ── 10. 恢复形状并 cross_merge ─────────────────────────────────────────
        # (B*4, L, nheads, headdim) → (B*4, L, d_inner) → (B, 4, d_inner, L)
        y = y.view(B * 4, L, self.d_inner)
        y_4dir = (
            y.view(B, 4, L, self.d_inner)
            .permute(0, 1, 3, 2)
            .contiguous()
        )   # (B, 4, d_inner, L)

        # cross_merge: (B, 4, d_inner, L) → (B, d_inner, H, W)
        y_merged = cross_merge_fn(y_4dir)

        # ── 11. 输出归一化 + z gating + 输出投影 ──────────────────────────────
        y_merged = y_merged.permute(0, 2, 3, 1).contiguous()   # (B, H, W, d_inner)
        y_merged = self.out_norm(y_merged)

        # z gating（与 SparseSS2D 一致：element-wise SiLU gate）
        out = y_merged * F.silu(z)
        out = self.out_proj(out)    # (B, H, W, d_model)

        if self.channel_first:
            out = out.permute(0, 3, 1, 2).contiguous()   # → (B, d_model, H, W)

        return self.dropout(out)
