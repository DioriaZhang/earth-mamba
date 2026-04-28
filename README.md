# earth-mamba

由 `dualsg_mamba` 复制的遥感向 Vision Mamba 变体：三路设计分别对应 **compression-aware** 稀疏 SSM、**noise-robust** 门控（ARMG）与 **多向扫描 + 语义图** 全局分支（非严格旋转不变）。

## 安装

在项目根目录（本目录）执行：

```bash
pip install -e ./kernels/selective_scan
pip install -e .
```

依赖见 `requirements.txt`（其中本地算子已改为 `-e ./kernels/selective_scan`）。

## SSM 后端

- 默认：已编译的 `selective_scan_cuda` / `oflex`（与 VMamba 风格四向 scan 配合）。
- 纯 PyTorch 参考（可反传，用于对接 Mamba-3 等新 recurrence 前的链路验证）：

```bash
set EARTH_MAMBA_SSM_BACKEND=torch_easy   # Windows CMD
# 或 PowerShell: $env:EARTH_MAMBA_SSM_BACKEND="torch_easy"
```

或在构造 `SparseSS2D` 时传入 `ssm_backend="torch_easy"`。

## 模型入口

```python
from earth_mamba import EarthMamba, BackboneEarthMamba, EarthMambaBlock
```

## 训练（与 `train_ddp_configs_benchmark_v2_monitor.py`）

- **BF16**：`--amp --amp_dtype bf16`（A100/H100 推荐）；FP16 时用 `--amp_dtype fp16` 且会启用 `GradScaler`。
- **selective_scan 保持 FP32**：`--ssm_fp32` 或环境变量 `EARTH_MAMBA_SELECTIVE_SCAN_FP32=1`（与 BF16 可同时开，内核内升精度）。
- **Gradient checkpoint**：`--use_grad_checkpoint`（EarthMamba 的 `use_checkpoint`）。
- **FSDP 多卡分片**：`--fsdp`（需多卡 DDP；类 ZeRO-3；**DeepSpeed ZeRO** 需单独装 deepspeed 配置）。
- **patch**：`--patch_size 16` 减少 token 数（与 `img_size` 搭配）。
- **DataLoader**：`--num_workers` / `--prefetch_factor` / `--persistent_workers`；**LMDB**：`--lmdb_path`（需 `pip install lmdb`，键 `__len__` + 索引）。
- **路径**：脚本与 `earth-mamba`、`dualsg_mamba` 同父目录时自动加入 `sys.path`；或设 `MAMBA_PROJECT_ROOT`。

## 静态逻辑核对（无需跑 GPU）

1. **数据流**：`EarthMambaBlock` 中 x → norm1 → ARMG（软门控）→ 与 z 残差混合 → 同一份 z 并行送入 SparseSS2D（Path A）与 LatentGraph（Path B）→ y_sparse + λ·y_graph → 与 x 残差 → norm2 + MLP 残差。ARMG 在 SSM/图之前，与文档「先抗干扰再进 SSM」一致。
2. **Compression-aware**：`SparseSS2D` 中 `sparse_gate` 对展平序列维做 Sparsemax，得到 `m_t` 与 `B_mat` 相乘后再送入 `selective_scan_fn`；四向 `cross_scan_fn` / `cross_merge_fn` 仅改变序列化顺序与合并，不改变该门控语义。
3. **SSM 后端分发**：`selective_scan_fn` 若 `eff == "torch_easy"`（由参数 `backend` 或环境变量 `EARTH_MAMBA_SSM_BACKEND` 决定，**显式参数优先**）则走 `selective_scan_torch_easy`（动态加载 `kernels/selective_scan/test_selective_scan_easy.py` 中的 `SelectiveScanEasy`）；否则走 `SelectiveScanCuda`，`backend=None` 时使用编译扩展探测到的 `SS_BACKEND`（`oflex` 或 `mamba`）。
4. **自定义 `ssm_cls`**：`EarthMambaBlock` 仅在 `inspect.signature(OpClass.__init__).parameters` 含 `ssm_backend` 时才传入该参数，避免 Baseline 类构造报错。
