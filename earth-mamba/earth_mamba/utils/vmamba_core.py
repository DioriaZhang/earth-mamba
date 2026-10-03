import os
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import warnings

# ============================================================
# PyTorch 版本自适应：2.4+ 弃用了 torch.cuda.amp.custom_fwd/bwd
# ============================================================
def _parse_torch_version():
    parts = torch.__version__.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1].split("+")[0].split("a")[0].split("b")[0].split("rc")[0])
    except (IndexError, ValueError):
        major, minor = 2, 0
    return major, minor

_TORCH_MAJOR, _TORCH_MINOR = _parse_torch_version()
_USE_NEW_AMP_API = (_TORCH_MAJOR > 2) or (_TORCH_MAJOR == 2 and _TORCH_MINOR >= 4)

if _USE_NEW_AMP_API:
    _custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    _custom_bwd = torch.amp.custom_bwd(device_type="cuda")
else:
    _custom_fwd = torch.cuda.amp.custom_fwd  # type: ignore[attr-defined]
    _custom_bwd = torch.cuda.amp.custom_bwd  # type: ignore[attr-defined]

# ============================================================
# Part 1: 环境检测与后端加载
# ============================================================
try:
    import triton
    import triton.language as tl
    WITH_TRITON = True
except ImportError:
    WITH_TRITON = False
    warnings.warn("未检测到 Triton！将回退到 Python 实现，显存占用可能会很高。")

# 优先加载 H200 优化的 oflex 核心
try:
    import selective_scan_cuda_oflex
    SS_BACKEND = "oflex"
    selective_scan_cuda = selective_scan_cuda_oflex
except ImportError:
    try:
        import selective_scan_cuda
        SS_BACKEND = "mamba"
        selective_scan_cuda = selective_scan_cuda
    except ImportError:
        SS_BACKEND = None
        # 这里不直接报错，允许只用 Cross Scan 的情况，但用到 SSM 时会崩
        warnings.warn("未检测到编译好的 CUDA 核心 (selective_scan_cuda)！")

# ============================================================
# Part 2: Triton 加速算子 (移植自 VMamba，解决 OOM)
# ============================================================
if WITH_TRITON:
    @triton.jit
    def triton_cross_scan_flex(
        x: tl.tensor, y: tl.tensor, x_layout: tl.constexpr, y_layout: tl.constexpr,
        operation: tl.constexpr, onebyone: tl.constexpr, scans: tl.constexpr,
        BC: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr,
        DC: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
        NH: tl.constexpr, NW: tl.constexpr,
    ):
        i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_h, i_w = (i_hw // NW), (i_hw % NW)
        _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
        _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
        _mask_hw = _mask_h[:, None] & _mask_w[None, :]
        _for_C = min(DC - i_c * BC, BC)

        pos_h = (i_h * BH + tl.arange(0, BH)[:, None])
        pos_w = (i_w * BW + tl.arange(0, BW)[None, :])
        neg_h = (DH - i_h * BH - 1 - tl.arange(0, BH)[:, None])
        neg_w = (DW - i_w * BW - 1 - tl.arange(0, BW)[None, :])
        
        # scans=0: cross scan
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = pos_w * DH + pos_h
        HWRoute2 = neg_h * DW + neg_w
        HWRoute3 = neg_w * DH + neg_h

        _tmp1 = DC * DH * DW
        y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
        
        if y_layout == 0:
            p_y1 = y_ptr_base + HWRoute0; p_y2 = y_ptr_base + _tmp1 + HWRoute1
            p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2; p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
        else:
            p_y1 = y_ptr_base + HWRoute0 * 4 * DC; p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
            p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC; p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC       
        
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0: p_x = x_ptr_base + HWRoute0
        else: p_x = x_ptr_base + HWRoute0 * DC

        if operation == 0: # scan
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
        elif operation == 1: # merge
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

    class CrossScanTritonF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            B, C, H, W = x.shape
            B, C, H, W = int(B), int(C), int(H), int(W)
            BC, BH, BW = 1, 32, 32
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.shape = (B, C, H, W)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
            y = x.new_empty((B, 4, C, H * W))
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x.contiguous(), y, 0, 0, 0, 0, 0, BC, BH, BW, C, H, W, NH, NW
            )
            return y

        @staticmethod
        def backward(ctx, y):
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            x = y.new_empty((B, C, H, W))
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x, y.contiguous(), 0, 0, 1, 0, 0, BC, BH, BW, C, H, W, NH, NW
            )
            return x

    class CrossMergeTritonF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y):
            B, K, C, L = y.shape # (B, 4, C, L)
            H = int(L**0.5)
            W = H
            B, C, H, W = int(B), int(C), int(H), int(W)
            BC, BH, BW = 1, 32, 32
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.shape = (B, C, H, W)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
            x = y.new_empty((B, C, H, W))
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x, y.contiguous(), 0, 0, 1, 0, 0, BC, BH, BW, C, H, W, NH, NW
            )
            return x

        @staticmethod
        def backward(ctx, x):
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            y = x.new_empty((B, 4, C, H * W))
            triton_cross_scan_flex[(NH * NW, NC, B)](
                x.contiguous(), y, 0, 0, 0, 0, 0, BC, BH, BW, C, H, W, NH, NW
            )
            return y

# ============================================================
# Part 3: Python 回退实现 (仅当 Triton 不可用时使用)
# ============================================================
def cross_scan_torch(x):
    B, C, H, W = x.shape
    y = x.new_empty((B, 4, C, H * W))
    y[:, 0, :, :] = x.flatten(2, 3)
    y[:, 1, :, :] = x.transpose(2, 3).flatten(2, 3)
    y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
    return y

def cross_merge_torch(y):
    B, K, D, L = y.shape
    H = int(L**0.5)
    W = H
    y = y.view(B, K, D, -1)
    y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
    y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, D, -1)
    return y.view(B, D, H, W)

# ============================================================
# Part 4: 统一调用接口 (自动选择后端)
# ============================================================
def cross_scan_fn(x):
    """ (B, C, H, W) -> (B, 4, C, L) """
    if WITH_TRITON and x.is_cuda:
        return CrossScanTritonF.apply(x)
    return cross_scan_torch(x)

def cross_merge_fn(y):
    """ (B, 4, C, L) -> (B, C, H, W) """
    if WITH_TRITON and y.is_cuda:
        return CrossMergeTritonF.apply(y)
    return cross_merge_torch(y)

# ============================================================
# Part 5: 核心 CUDA 封装 (带 AMP 类型自动对齐)
# ============================================================
def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @_custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
        ctx.delta_softplus = delta_softplus
        
        # 1. 自动选择最佳后端
        if backend is None:
            backend = SS_BACKEND
        if backend is None:
            raise ImportError("未找到可用的 selective_scan_cuda 实现！")
        
        ctx.backend = backend
        ctx.orig_dtype = u.dtype
        ctx.ssm_fp32 = _env_truthy("EARTH_MAMBA_SELECTIVE_SCAN_FP32")
        
        # 2. [关键修复] AMP 类型对齐：强制 B, C 跟随 u 的精度
        # 解决 H200 混合精度训练时的 RuntimeError
        target_dtype = u.dtype
        if B.dtype != target_dtype: B = B.to(target_dtype)
        if C.dtype != target_dtype: C = C.to(target_dtype)
        # D 和 delta_bias 保持 FP32 或跟随 u 均可，核心通常能处理，但为了保险：
        if D is not None and D.dtype != torch.float: D = D.float() # D 通常建议 FP32
        if delta_bias is not None and delta_bias.dtype != torch.float: delta_bias = delta_bias.float()

        # 2b. 可选：整段 selective_scan 在 FP32 上算（BF16/FP16 训练时更稳，略慢）
        if ctx.ssm_fp32 and u.dtype != torch.float32:
            u = u.float()
            delta = delta.float()
            A = A.float()
            B = B.float()
            C = C.float()

        # 3. 调用底层核心
        if backend == "oflex":
            # oflex 接口: ..., nrows, backnrows, ... (通常 nrows=1)
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        elif backend == "mamba":
            # 标准 mamba 接口
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out.to(ctx.orig_dtype)
    
    @staticmethod
    @_custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        backend = ctx.backend
        
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if getattr(ctx, "ssm_fp32", False) and dout.dtype != torch.float32:
            dout = dout.float()
            
        # 4. [关键修复] Backward 阶段的类型对齐
        # 反向传播时，Pytorch 可能会传入原始精度的 Tensor，再次检查
        target_dtype = u.dtype
        if B.dtype != target_dtype: B = B.to(target_dtype)
        if C.dtype != target_dtype: C = C.to(target_dtype)
        
        if backend == "oflex":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "mamba":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus, False
            )
        od = ctx.orig_dtype
        if od != torch.float32:
            du = du.to(od)
            ddelta = ddelta.to(od)
            dA = dA.to(od)
            dB = dB.to(od)
            dC = dC.to(od)
            if dD is not None:
                dD = dD.to(od)
            if ddelta_bias is not None:
                ddelta_bias = ddelta_bias.to(od)
            
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


# --- Mamba-3 迁移占位：纯 PyTorch selective scan（与 kernels 内 SelectiveScanEasy 一致，便于替换为官方 Mamba-3 recurrence）---
_torch_easy_module = None


def _get_selective_scan_torch_easy_module():
    global _torch_easy_module
    if _torch_easy_module is None:
        root = Path(__file__).resolve().parents[2]
        path = root / "kernels" / "selective_scan" / "test_selective_scan_easy.py"
        if not path.is_file():
            raise FileNotFoundError(f"未找到参考实现: {path}")
        spec = importlib.util.spec_from_file_location("earth_mamba_selective_scan_easy", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _torch_easy_module = mod
    return _torch_easy_module


def selective_scan_torch_easy(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    """
    与编译 CUDA selective_scan 同接口的 PyTorch 实现（较慢，可反传）。
    用于验证「替换 selective_scan 调用链」；后续可在此接入官方 Mamba-3 离散化 / MIMO 形式。
    """
    mod = _get_selective_scan_torch_easy_module()
    target_dtype = u.dtype
    if B.dtype != target_dtype:
        B = B.to(target_dtype)
    if C.dtype != target_dtype:
        C = C.to(target_dtype)
    if D is not None and D.dtype != torch.float:
        D = D.float()
    if _env_truthy("EARTH_MAMBA_SELECTIVE_SCAN_FP32") and u.dtype != torch.float32:
        u = u.float()
        delta = delta.float()
        A = A.float()
        B = B.float()
        C = C.float()
    out, _ = mod.SelectiveScanEasy.apply(
        u, delta, A, B, C, D, delta_bias, delta_softplus, False, 64
    )
    return out.to(target_dtype)


def _effective_ssm_backend(explicit_backend):
    if explicit_backend:
        return explicit_backend
    env = os.environ.get("EARTH_MAMBA_SSM_BACKEND", "").strip()
    return env or None


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    delta_bias=None,
    delta_softplus=True,
    oflex=True,
    backend=None,
):
    eff = _effective_ssm_backend(backend)

    if eff == "torch_easy":
        return selective_scan_torch_easy(
            u, delta, A, B, C, D, delta_bias, delta_softplus
        )

    # [关键修复] 当 CUDA kernel 完全不可用时，自动 fallback 到 torch_easy
    # 避免在没有编译/挂载 selective_scan_cuda 时直接 ImportError 崩溃
    if SS_BACKEND is None and eff in (None, ""):
        warnings.warn(
            "selective_scan_cuda / selective_scan_cuda_oflex 均未加载，"
            "自动回退到 torch_easy (纯 PyTorch 实现，较慢但可训练)。"
            "如需加速，请编译 kernels/selective_scan 或设置 "
            "EARTH_MAMBA_SSM_BACKEND=torch_easy 以消除此警告。",
            stacklevel=2,
        )
        return selective_scan_torch_easy(
            u, delta, A, B, C, D, delta_bias, delta_softplus
        )

    cuda_backend = backend if backend not in (None, "", "torch_easy") else None
    return SelectiveScanCuda.apply(
        u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, cuda_backend
    )