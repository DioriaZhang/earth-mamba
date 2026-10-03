# EarthMamba

EarthMamba is a hierarchical Mamba backbone for remote-sensing visual representation learning. This folder is the **release package** of the EarthMamba project: SimMIM-style masked image modeling pretraining code, data-processing utilities, downstream task wrappers (classification, semantic segmentation, horizontal-box and oriented-box detection), and an ablation suite. The core model package lives in the sibling [`earth-mamba/`](../earth-mamba) folder. Paths below are relative to the **repository root**.

EarthMamba was submitted to AAAI 2027 and was not accepted (review scores 5 / 5 / 4 / 4). This release is provided as-is for reference.

## Documentation

- [Data preparation](DATA.md)
- [Model zoo and checkpoints](MODEL_ZOO.md)
- [Checkpoint packaging](checkpoints/README.md)
- [Downstream suite](downstream_suite/README.md)
- [Data processing notes](data_process/Data_Preprocessing_Pipeline.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [License](LICENSE)
- [Citation metadata](CITATION.cff)
- [README command audit](docs/README_COMMAND_AUDIT.md)

## Confirmed Implementation

The core implementation is in `earth-mamba/earth_mamba/models/earth_mamba.py` and `earth-mamba/earth_mamba/models/earth_mamba_block.py`.

| Component | Evidence path |
|---|---|
| 4-stage hierarchical backbone | `earth-mamba/earth_mamba/models/earth_mamba.py` |
| ARMG | `earth-mamba/earth_mamba/modules/armg.py` |
| Sparse SSM | `earth-mamba/earth_mamba/modules/spatial_sparse.py` |
| Mamba3 SSM path | `earth-mamba/earth_mamba/modules/ss2d_mamba3.py`, `earth-mamba/earth_mamba/modules/mamba3.py` |
| Latent Graph | `earth-mamba/earth_mamba/modules/latent_graph.py` |
| Patch embedding and patch merging | `earth-mamba/earth_mamba/models/earth_mamba.py` |
| Multi-scale downstream output | `BackboneEarthMamba` in `earth-mamba/earth_mamba/models/earth_mamba.py` |

Confirmed release model configurations:

| Size | Depths | Dimensions |
|---|---|---|
| Tiny | `[2, 2, 9, 2]` | `[96, 192, 384, 768]` |
| Small | `[2, 2, 27, 2]` | `[96, 192, 384, 768]` |
| Base | `[2, 2, 27, 2]` | `[128, 256, 512, 1024]` |

Tiny is a configuration-only entry in this anonymous package. Small and Base checkpoints are described in [MODEL_ZOO.md](MODEL_ZOO.md).

Downstream adapters are engineering interfaces for task heads and should not be described as separate paper contributions unless the manuscript explicitly supports that claim.

## Installation

Use a CUDA/PyTorch environment compatible with the custom selective-scan and Triton code. CPU-only full training or full forward execution is not claimed by this release.

Install the core package and custom kernel from the repository root:

```bash
python -m pip install -r earth-mamba/requirements.txt
python -m pip install -e earth-mamba/kernels/selective_scan
python -m pip install -e earth-mamba
```

Optional downstream dependencies:

```bash
python -m pip install -r downstream_suite/requirements.txt
```

Optional data-processing dependencies:

```bash
python -m pip install -r data_process/requirements.txt
```

## Minimal Model Construction

```python
import torch
from earth_mamba.models.earth_mamba import EarthMamba

model = EarthMamba(
    imgsize=224,
    patch_size=16,
    in_chans=3,
    num_classes=1000,
    depths=[2, 2, 27, 2],
    dims=[96, 192, 384, 768],
    ssm_version="mamba3",
    ssm_d_state=64,
    ssm_headdim=64,
    posembed=True,
    downsample_version="v3",
)

x = torch.randn(1, 3, 224, 224)
logits = model(x)
```

For downstream multi-scale features, use the wrappers registered through `downstream_suite/backbone.py`.

## Data

The pretraining manifest for the release target contains approximately 3.67M image entries. The real manifest is not included because it is large and can contain private absolute paths. See [DATA.md](DATA.md) and [examples/train_list.example.txt](examples/train_list.example.txt).

## Pretraining

Pretraining uses SimMIM-style masked image modeling/reconstruction: masked pixel regions are replaced with a learnable mask value, the full masked image is encoded, and normalized L1 reconstruction loss is computed only over masked pixels.

Small:

```bash
python train/pretrain_mae.py \
  --data_list examples/train_list.example.txt \
  --output_dir ${OUTPUT_DIR}/pretrain_small \
  --model_size small \
  --patch_size 16 \
  --batch_size 16 \
  --grad_accum 2 \
  --epochs 800 \
  --warmup_epochs 40 \
  --amp_dtype bf16
```

Base:

```bash
python train/pretrain_mae_base.py \
  --data_list examples/train_list.example.txt \
  --output_dir ${OUTPUT_DIR}/pretrain_base \
  --batch_size 4 \
  --grad_accum 4 \
  --epochs 800 \
  --warmup_epochs 40 \
  --amp_dtype bf16
```

Archived Base records indicate a staged 40 + 30 + 30 weights-only continuation recipe. Document it as staged continuation, not as one uninterrupted cosine run.

## Data Processing Templates

Run each script with `--help` in your environment for the complete argument list.

```bash
python data_process/preprocess_downstream.py \
  /path/to/downstream_dataset \
  --output_root ${DATA_ROOT}/Downstream-datasets
```

```bash
python data_process/pipeline_cluster_1_build.py \
  --globe230k /path/to/Globe230k \
  --luojia /path/to/LuoJia-CDKD \
  --downstream_dirs ${DATA_ROOT}/Downstream-datasets \
  --feature_root ${OUTPUT_DIR}/feature \
  --weight_path /path/to/satmae-pretrain-vit-base-e199.pth
```

```bash
python data_process/pipeline_cluster_2_filter.py \
  --pool_dir /path/to/candidate_pool \
  --feature_root ${OUTPUT_DIR}/feature \
  --weight_path /path/to/satmae-pretrain-vit-base-e199.pth \
  --output_dir ${OUTPUT_DIR}/pool_filter
```

## Downstream Templates

Set paths:

```bash
export DATA_ROOT=/path/to/datasets
export OUTPUT_DIR=/path/to/outputs
export CKPT_SMALL=/path/to/earthmamba-small/backbone.pth
export CKPT_BASE=/path/to/earthmamba-base/backbone.pth
```

Classification:

```bash
python downstream_suite/run_dfc15.py \
  --backbone earth-mamba \
  --data_dir ${DATA_ROOT}/DFC15 \
  --ckpt ${CKPT_SMALL} \
  --output_dir ${OUTPUT_DIR}/dfc15_small
```

```bash
python downstream_suite/run_patternnet.py \
  --backbone earth-mamba \
  --data_dir ${DATA_ROOT}/PatternNet \
  --ckpt ${CKPT_SMALL} \
  --output_dir ${OUTPUT_DIR}/patternnet_small
```

Semantic segmentation:

```bash
python downstream_suite/run_inria.py \
  --backbone earth-mamba \
  --data_dir ${DATA_ROOT}/INRIA \
  --ckpt ${CKPT_SMALL} \
  --output_dir ${OUTPUT_DIR}/inria_small
```

```bash
python downstream_suite/run_deepglobe.py \
  --backbone earth-mamba \
  --data_dir ${DATA_ROOT}/DeepGlobe \
  --ckpt ${CKPT_SMALL} \
  --output_dir ${OUTPUT_DIR}/deepglobe_small
```

Oriented detection:

```bash
python downstream_suite/run_dior_r.py \
  --data-dir ${DATA_ROOT}/DIOR-R \
  --backbone earth-mamba \
  --ckpt ${CKPT_SMALL} \
  --project-root /path/to/project-root \
  train \
  --work-dir ${OUTPUT_DIR}/dior_r_small
```

Use `--backbone earth-mamba-b` with `${CKPT_BASE}` for Base runs. The DIOR horizontal-box entry and archived result are documented in [MODEL_ZOO.md](MODEL_ZOO.md); the README command is intentionally omitted pending a dispatcher/parser consistency cleanup in the downstream wrapper.

## Results and Checkpoints

See [MODEL_ZOO.md](MODEL_ZOO.md) for model configurations, checkpoint placeholders, and archived downstream results. The values are consolidated from archived experiment records; complete step-level logs are not included in this release.

## License

EarthMamba project-original code is released under [Apache-2.0](LICENSE). This does not relicense third-party code, kernels, dependencies, datasets, or weights. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
