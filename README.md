# EarthMamba

[English](#english) · [中文](#中文)

---

## English

EarthMamba is a hierarchical Mamba (state-space model) backbone for remote-sensing visual representation learning. The design combines a **compression-aware** sparse SSM path, a **noise-robust** gated branch (ARMG), and a **multi-directional scan + semantic-graph** global branch.

This repository hosts the whole project in two top-level folders:

- **`earth-mamba/`** — the core model package (the original development repository): the `earth_mamba` Python package, the custom `selective_scan` CUDA kernel, Triton Mamba-3 operators, and a training entry point.
- **`EarthMamba-release/`** — the paper release package: SimMIM-style masked image modeling pretraining, the data-processing pipeline, downstream task suites (classification, semantic segmentation, horizontal-box and oriented-box detection), an ablation suite, and model-zoo / docs.

### Repository layout

```
.
├── earth-mamba/                  # Core model package (original dev repo)
│   ├── earth_mamba/              # Python package: EarthMamba / BackboneEarthMamba / EarthMambaBlock
│   ├── kernels/selective_scan/   # Custom selective-scan CUDA kernel
│   ├── train.py  setup.py  env.yml  requirements.txt
│   └── README.md
├── EarthMamba-release/           # Paper release package
│   ├── train/                    # SimMIM-style masked pretraining
│   ├── data_process/             # Data-processing pipeline
│   ├── downstream_suite/         # Classification / segmentation / HBB / OBB detection
│   ├── ablation_suite/           # Ablation studies
│   ├── checkpoints/  docs/  examples/  figure1_vis/  logs/
│   ├── DATA.md  MODEL_ZOO.md  LICENSE  CITATION.cff  THIRD_PARTY_NOTICES.md
│   └── README.md
└── README.md                     # This file
```

Model configurations: **Tiny** `[2, 2, 9, 2]` / `[96, 192, 384, 768]`, **Small** `[2, 2, 27, 2]` / `[96, 192, 384, 768]`, **Base** `[2, 2, 27, 2]` / `[128, 256, 512, 1024]`.

### Installation

Run from the repository root:

```bash
python -m pip install -r earth-mamba/requirements.txt
python -m pip install -e earth-mamba/kernels/selective_scan
python -m pip install -e earth-mamba
```

Optional downstream / data-processing dependencies:

```bash
python -m pip install -r EarthMamba-release/downstream_suite/requirements.txt
python -m pip install -r EarthMamba-release/data_process/requirements.txt
```

### Quick start

```python
import torch
from earth_mamba.models.earth_mamba import EarthMamba

model = EarthMamba(
    imgsize=224, patch_size=16, in_chans=3, num_classes=1000,
    depths=[2, 2, 27, 2], dims=[96, 192, 384, 768],
    ssm_version="mamba3", ssm_d_state=64, ssm_headdim=64,
    posembed=True, downsample_version="v3",
)
logits = model(torch.randn(1, 3, 224, 224))
```

See `EarthMamba-release/README.md` for pretraining and downstream commands; paths there are relative to the repository root.

### Status

EarthMamba was submitted to **AAAI 2027** and was **not accepted** (review scores 5 / 5 / 4 / 4). The code is released as-is for reference.

### License and third-party notice

Project-original code is released under **Apache-2.0** (`EarthMamba-release/LICENSE`). This does not relicense third-party code, kernels, dependencies, datasets, or weights. See `EarthMamba-release/THIRD_PARTY_NOTICES.md`.

---

## 中文

EarthMamba 是一个面向遥感视觉表征学习的分层 Mamba（状态空间模型）骨干网络，由三路设计组成：**compression-aware** 稀疏 SSM、**noise-robust** 门控分支（ARMG），以及**多向扫描 + 语义图**的全局分支。

本仓库以两个顶层文件夹承载整个项目：

- **`earth-mamba/`** —— 核心模型包（原开发仓库）：`earth_mamba` Python 包、自定义 `selective_scan` CUDA 内核、Triton Mamba-3 算子，以及训练入口。
- **`EarthMamba-release/`** —— 论文发布包：SimMIM 式掩码图像建模预训练、数据处理管线、下游任务套件（分类、语义分割、水平框与旋转框检测）、消融套件，以及模型动物园与文档。

### 目录结构

```
.
├── earth-mamba/                  # 核心模型包（原开发仓库）
│   ├── earth_mamba/              # Python 包：EarthMamba / BackboneEarthMamba / EarthMambaBlock
│   ├── kernels/selective_scan/   # 自定义 selective-scan CUDA 内核
│   ├── train.py  setup.py  env.yml  requirements.txt
│   └── README.md
├── EarthMamba-release/           # 论文发布包
│   ├── train/                    # SimMIM 式掩码预训练
│   ├── data_process/             # 数据处理管线
│   ├── downstream_suite/         # 分类 / 语义分割 / 水平框 / 旋转框检测
│   ├── ablation_suite/           # 消融实验
│   ├── checkpoints/  docs/  examples/  figure1_vis/  logs/
│   ├── DATA.md  MODEL_ZOO.md  LICENSE  CITATION.cff  THIRD_PARTY_NOTICES.md
│   └── README.md
└── README.md                     # 本文件
```

模型配置：**Tiny** `[2, 2, 9, 2]` / `[96, 192, 384, 768]`，**Small** `[2, 2, 27, 2]` / `[96, 192, 384, 768]`，**Base** `[2, 2, 27, 2]` / `[128, 256, 512, 1024]`。

### 安装

在仓库根目录执行：

```bash
python -m pip install -r earth-mamba/requirements.txt
python -m pip install -e earth-mamba/kernels/selective_scan
python -m pip install -e earth-mamba
```

可选的下游 / 数据处理依赖：

```bash
python -m pip install -r EarthMamba-release/downstream_suite/requirements.txt
python -m pip install -r EarthMamba-release/data_process/requirements.txt
```

### 快速开始

```python
import torch
from earth_mamba.models.earth_mamba import EarthMamba

model = EarthMamba(
    imgsize=224, patch_size=16, in_chans=3, num_classes=1000,
    depths=[2, 2, 27, 2], dims=[96, 192, 384, 768],
    ssm_version="mamba3", ssm_d_state=64, ssm_headdim=64,
    posembed=True, downsample_version="v3",
)
logits = model(torch.randn(1, 3, 224, 224))
```

预训练与下游命令见 `EarthMamba-release/README.md`，其中路径均相对仓库根目录。

### 状态

EarthMamba 投稿 **AAAI 2027**，**未录用**（评审分数 5 / 5 / 4 / 4）。代码按原样发布，供参考。

### 许可与第三方声明

项目原创代码以 **Apache-2.0** 许可发布（`EarthMamba-release/LICENSE`）。该许可不覆盖第三方代码、内核、依赖、数据集或权重。详见 `EarthMamba-release/THIRD_PARTY_NOTICES.md`。
