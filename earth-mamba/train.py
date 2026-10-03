"""
Earth-Mamba 训练入口（本仓库自包含，不依赖父目录或其它对比脚本）。

等价能力来自原 `train_ddp_configs_benchmark_v2_monitor.py` 中与 Earth-Mamba 相关的部分，
实现位于 `earth_mamba/benchmark_train.py`。

单机 8 卡示例：

    export EARTH_MAMBA_SELECTIVE_SCAN_FP32=1

    torchrun --nproc_per_node=8 train.py \\
        --model earth \\
        --amp --amp_dtype bf16 \\
        --use_grad_checkpoint \\
        --fsdp \\
        --patch_size 16 \\
        --num_workers 8 \\
        --prefetch_factor 4 \\
        --epochs 300 \\
        --batch_size 128 \\
        --data_dir /path/to/dataset \\
        --output_dir /path/to/output

注意：仅支持 EarthMamba；`--model` 若写出则必须为 `earth`（默认即为 earth），便于沿用旧命令行。

编译 selective_scan CUDA 扩展（可选，见 kernels/selective_scan）：

    cd kernels/selective_scan && pip install -e .

若扩展不可用，可设置：

    export EARTH_MAMBA_SSM_BACKEND=torch_easy

（调试用途，速度较慢。）

依赖：torch、timm、torchvision、Pillow、tqdm；可选 lmdb、psutil、pynvml、matplotlib。
"""

from __future__ import annotations

import os
import sys

# 支持「在仓库根目录直接 python train.py」时找到 earth_mamba 包
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from earth_mamba.benchmark_train import main

if __name__ == "__main__":
    main()
