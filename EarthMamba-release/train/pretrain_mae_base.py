#!/usr/bin/env python3
"""
pretrain_mae_base.py  ─  EarthMamba **base** SimMIM 预训练入口
==============================================================

在 pretrain_mae.py 基础上，默认启用：
  - ``--model_size base``（depths=[2,2,27,2], dims=[128,256,512,1024]，约 161M 参数）
  - ``--flat_ckpt``：checkpoint 平铺在 ``epochs/`` 下，如 ``checkpoint_ep22.pth`` / ``backbone_ep22.pth``
  - ``--no_recon``：不生成重建对比图

其余训练逻辑、数据管道、DDP/FSDP、日志与 pretrain_mae.py 完全一致。

使用示例
─────────────────────────────────────────────────
单卡 A100（base 显存较大，batch 需酌减）::

    python pretrain_mae_base.py \\
        --data_list /data/pretrain_list.txt \\
        --output_dir /hy-tmp/pretrain_base \\
        --batch_size 8 --grad_accum 2 \\
        --epochs 800 --warmup_epochs 40 \\
        --amp_dtype bf16

8 卡 DDP::

    torchrun --standalone --nproc_per_node=8 pretrain_mae_base.py \\
        --data_list /data/pretrain_list.txt \\
        --output_dir /hy-tmp/pretrain_base \\
        --batch_size 4 --grad_accum 4 \\
        --epochs 800 --amp_dtype bf16

测速（先跑 profile_step_a100_base.py --quick）::

    CUDA_VISIBLE_DEVICES=0 python profile_step_a100_base.py \\
        --data_list /data/pretrain_list.txt --quick

4 卡 DDP 真实 step 测速::

    torchrun --standalone --nproc_per_node=4 profile_step_a100_base.py \\
        --data_list /data/pretrain_list.txt --batch_size 8 --grad_accum 2 --ddp_only

默认 save_every=3（每 3 epoch 存一次 checkpoint/backbone）

断点续训（指定 epoch）::

    python pretrain_mae_base.py \\
        --resume /hy-tmp/pretrain_base/pretrain_20260613_1200 \\
        --resume_epoch 22 \\
        --data_list /data/pretrain_list.txt \\
        --output_dir /hy-tmp/pretrain_base_resume

下游加载骨干::

    backbone.load_state_dict(
        torch.load(".../epochs/backbone_ep22.pth", map_location="cpu")
    )

输出目录（flat_ckpt）
─────────────────────────────────────────────────
  <run>/epochs/
    checkpoint_ep1.pth
    backbone_ep1.pth
    checkpoint_ep22.pth
    backbone_ep22.pth
    loss_epoch_ep22.png
  <run>/logs/          训练日志与曲线（不变）
  <run>/periodic/      周期性曲线快照（不变）
"""

from __future__ import annotations

import sys

import pretrain_mae as _pm


def _argv_has(flag: str) -> bool:
    prefix = flag + "="
    for arg in sys.argv[1:]:
        if arg == flag or arg.startswith(prefix):
            return True
    return False


def _inject_defaults() -> None:
    """在未显式传入时注入 base 预训练默认开关。"""
    defaults: list[str] = []
    if not _argv_has("--model_size"):
        defaults.extend(["--model_size", "base"])
    if not _argv_has("--ssm_version"):
        defaults.extend(["--ssm_version", "mamba3"])
    if not _argv_has("--save_every"):
        defaults.extend(["--save_every", "3"])
    if not _argv_has("--flat_ckpt"):
        defaults.append("--flat_ckpt")
    if not _argv_has("--no_recon"):
        defaults.append("--no_recon")
    if defaults:
        sys.argv[1:1] = defaults


if __name__ == "__main__":
    _inject_defaults()
    _pm.main()
