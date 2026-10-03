# Model Zoo

EarthMamba provides Tiny, Small, and Base model configurations. The anonymous submission package documents checkpoint slots for Small and Base, but checkpoint binaries are not included in this archive. Tiny is listed for configuration completeness only.

Checkpoints are large binary artifacts and should not be committed as ordinary Git files. In anonymous submission mode, release assets are referenced with anonymous placeholders or local package-relative paths. Public URLs should be added only after de-anonymization.

## Model Configurations

| Model | Depths | Dimensions | Pretrained weight | Status |
|---|---|---|---|---|
| EarthMamba-Tiny | `[2, 2, 9, 2]` | `[96, 192, 384, 768]` | Checkpoint not provided | Config only |
| EarthMamba-Small | `[2, 2, 27, 2]` | `[96, 192, 384, 768]` | `small/backbone.pth` | Checkpoint slot; binary not included |
| EarthMamba-Base | `[2, 2, 27, 2]` | `[128, 256, 512, 1024]` | `base/backbone.pth` | Checkpoint slot; binary not included |

Parameter counts are intentionally omitted because classification models, backbone-only modules, and dense downstream adapters have different counting scopes in the repository records.

## Checkpoint Package

Checkpoint binaries are not included in this archive. The paths below document the expected local layout if reviewers place checkpoints manually.

Expected asset layout:

```text
CKPT-earthmamba/
  small/
    checkpoint.pth
    backbone.pth
  base/
    backbone.pth
    checkpoint.pth
```

`backbone.pth` contains backbone-only weights intended for downstream loading. `checkpoint.pth` contains the full pretraining checkpoint for resume workflows. Ordinary downstream users should start from `backbone.pth`; the full resume checkpoint is not required for downstream evaluation.

## Weight Table

| Variant | File | Type | Intended use | Download | SHA256 |
|---|---|---|---|---|---|
| Small | `CKPT-earthmamba/small/backbone.pth` | Backbone-only weights | Downstream classification, segmentation, and detection | Not included in this archive | Placeholder |
| Small | `CKPT-earthmamba/small/checkpoint.pth` | Full pretraining checkpoint | Pretraining resume and audit | Not included in this archive | Placeholder |
| Base | `CKPT-earthmamba/base/backbone.pth` | Backbone-only weights | Downstream classification, segmentation, and detection | Not included in this archive | Placeholder |
| Base | `CKPT-earthmamba/base/checkpoint.pth` | Full pretraining checkpoint | Pretraining resume and audit | Not included in this archive | Placeholder |

SHA256 values are intentionally left as placeholders because checkpoint binaries are not part of this repository tree.

## Downstream Results

The values are consolidated from archived experiment records. Complete step-level logs are not included in this release.

| Task | Metric | EarthMamba-Small | EarthMamba-Base |
|---|---|---:|---:|
| DFC15 | macro mAP | 97.54 | 98.21 |
| PatternNet | validation accuracy | 99.87 | 99.92 |
| INRIA | validation mIoU | 86.64 | 87.43 |
| DeepGlobe | validation mIoU-6 | 76.23 | 76.84 |
| DIOR | mAP@0.5 | 66.27 | 67.32 |
| DIOR-R | mAP@0.5 | 61.60 | 62.86 |

No mean/std values are reported because the archived records do not establish repeated random-seed experiments.

## Loading

The verified public loading interface is through the downstream scripts:

```bash
python downstream_suite/run_patternnet.py \
  --backbone earth-mamba \
  --data_dir ${DATA_ROOT}/PatternNet \
  --ckpt /path/to/earthmamba-small/backbone.pth \
  --output_dir ${OUTPUT_DIR}/patternnet_small
```

Use `--backbone earth-mamba` for Small and `--backbone earth-mamba-b` for Base. No public `earthmamba_small(pretrained=True)` factory was found in the repository.
