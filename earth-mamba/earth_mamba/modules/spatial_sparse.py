import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import repeat

# Multi-direction scan (cross_scan / cross_merge) + selective SSM core.
from ..utils.vmamba_core import cross_scan_fn, cross_merge_fn, selective_scan_fn

class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        dim = self.dim
        number_of_logits = input.size(dim)
        input = input - input.max(dim=dim, keepdim=True)[0].expand_as(input)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        
        range_shape = [1] * input.ndim
        range_shape[dim] = number_of_logits
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, 
                             device=input.device, dtype=input.dtype).view(range_shape)
        range = range.expand_as(zs)
        
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim=dim)
        is_gt = bound > cumulative_sum_zs
        k = torch.max(is_gt * range, dim, keepdim=True)[0]
        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)
        
        self.output = torch.max(torch.zeros_like(input), input - taus)
        return self.output

class SparseSS2D(nn.Module):
    """
    Compression-aware 2D SSM 分支（Path A）：
    Sparsemax 在空间维上对输入投影 B 做硬稀疏门控，缓解有损压缩/信息稀释；
    配合四向 cross-scan（多向序列化），提升对扫描顺序与朝向的鲁棒性（scan-rotation aware，非严格几何不变）。

    SSM 核心默认走编译 selective_scan CUDA；设置环境变量 ``EARTH_MAMBA_SSM_BACKEND=torch_easy`` 或
    ``ssm_backend="torch_easy"`` 可走纯 PyTorch 路径，便于对接 Mamba-3 等新离散化实现。

    支持 ``channel_first=True`` 输入格式（即 BCHW），forward 中自动转换为 BHWC 再处理。
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        ssm_backend=None,
        channel_first=False,   # 显式声明，避免被 **kwargs 静默吞掉
        initialize="v0",       # 显式声明，保留接口兼容性（当前实现中不影响权重）
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ssm_backend = ssm_backend
        self.channel_first = channel_first
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=conv_bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias
        )
        self.act = act_layer()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.sparse_gate = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 4),
            nn.ReLU(),
            nn.Linear(self.d_inner // 4, 1),
            Sparsemax(dim=1)
        )

        self.A_log = self.A_log_init(self.d_state, self.d_inner)
        self.D = self.D_init(self.d_inner)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=conv_bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        # 支持 channel_first (B, C, H, W) 和 channel_last (B, H, W, C) 两种输入格式
        if self.channel_first:
            # (B, C, H, W) -> (B, H, W, C)
            x = x.permute(0, 2, 3, 1).contiguous()

        B, H, W, C = x.shape
        L = H * W

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        # [自动优化] 这里的 cross_scan_fn 会自动调用 Triton 版本 (如果在 GPU 上)
        y_scan = cross_scan_fn(x) # (B, 4, D, L)
        x_flat = y_scan.permute(0, 1, 3, 2).contiguous().view(-1, L, self.d_inner)

        x_dbl = self.x_proj(x_flat) 
        dt, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt) 

        # 空间稀疏门控 (m_t * B)
        m_t = self.sparse_gate(x_flat) 
        m_t = m_t.view(B, 4, L, 1) 
        B_mat = B_mat.view(B, 4, L, self.d_state)
        # [Broadcast] B_sparse = m_t * B
        B_sparse = m_t * B_mat 
        C_mat = C_mat.view(B, 4, L, self.d_state)

        # ----------------------------------------------------
        # 准备 CUDA 输入
        # ----------------------------------------------------
        u_core = y_scan.view(B, -1, L)  # y_scan 来自 Triton，天然 contiguous，view 后仍 contiguous
        
        dt_core = dt.view(B, 4, L, -1).permute(0, 1, 3, 2).contiguous().view(B, -1, L)
        
        B_core = B_sparse.permute(0, 1, 3, 2).contiguous()
        C_core = C_mat.permute(0, 1, 3, 2).contiguous()
        
        A_core = -self.A_log.float().exp() 

        # [调用核心] 
        # 不需要手动转 float() 了，vmamba_core 内部会自动处理 AMP 类型对齐
        y_global = selective_scan_fn(
            u_core, dt_core, A_core, B_core, C_core, self.D.float(),
            delta_bias=None, delta_softplus=True,
            backend=self.ssm_backend,
        )

        # ----------------------------------------------------
        # 输出处理
        # ----------------------------------------------------
        # 恢复为 (B, 4, D, L) 以适配 cross_merge_fn
        y_global = y_global.view(B, 4, -1, L)
        
        y = cross_merge_fn(y_global) # (B, D, H, W)

        y = y.permute(0, 2, 3, 1).contiguous()  # (B, D, H, W) -> (B, H, W, D)
        y = self.out_norm(y)

        y = y * F.silu(z)
        out = self.out_proj(y)  # (B, H, W, d_model)

        if self.channel_first:
            # 恢复为 (B, C, H, W)
            out = out.permute(0, 3, 1, 2).contiguous()

        return self.dropout(out)

    def A_log_init(self, d_state, d_inner, copies=4, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.expand(d_inner, d_state).clone()
        A_log = torch.log(A) 
        if copies > 1:
            A_log = A_log.expand(copies, -1, -1).clone() 
            if merge:
                A_log = A_log.flatten(0, 1) 
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    def D_init(self, d_inner, copies=4, merge=True):
        D = torch.ones(d_inner)
        if copies > 1:
            D = D.expand(copies, -1).clone()
            if merge: D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D