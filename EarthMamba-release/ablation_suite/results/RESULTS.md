# Ablation Results (archived)

> Compositional module ablation on EarthMamba-small (same checkpoint, gated branches).  
> Archive date: **2026-06-26**. Source: `raw_archive.json` and `raw/{task}_{variant}.json` in this folder.

## Module flags

| ID | Sparse SSM (A) | Graph (B) | ARMG (C) | Description |
|----|:--------------:|:---------:|:--------:|-------------|
| ID1 | | | | Pure baseline — all paths off |
| ID3 | ✓ | | | A only |
| ID4 | | ✓ | | B only |
| ID5 | | | ✓ | C only |
| ID6 | ✓ | ✓ | | A+B |
| ID7 | ✓ | | ✓ | A+C |
| ID8 | | ✓ | ✓ | B+C |
| ID9 | ✓ | ✓ | ✓ | Full model (**main-table reference**, not re-trained) |

## Summary table

| ID | DFC15 macro-mAP | INRIA val mIoU | DIOR mAP@0.5 |
|----|----------------:|---------------:|-------------:|
| ID1 | 88.17% | 83.36% | 48.81% |
| ID3 | 95.75% | **85.65%** | 61.63% |
| ID4 | 88.06% | 83.00% | 49.10% |
| ID5 | 89.94% | 83.89% | 49.15% |
| ID6 | 96.04% | 84.94% | 61.80% |
| ID7 | **96.16%** | 85.59% | **62.29%** |
| ID8 | 88.15% | 83.16% | 49.61% |
| ID9 | 97.54% † | 86.64% † | 66.27% † |

† ID9: cited from the six-model downstream main table (full EarthMamba-S), not from this ablation training schedule.

## Per-task best epochs

### DFC15 (80 epochs, test macro-mAP)

| ID | Best epoch | macro-mAP | micro-mAP |
|----|----------|----------:|----------:|
| ID1 | 52 | 88.17% | 95.32% |
| ID3 | 51 | 95.75% | 98.59% |
| ID4 | 76 | 88.06% | 95.90% |
| ID5 | 75 | 89.94% | 96.43% |
| ID6 | 54 | 96.04% | 98.57% |
| ID7 | 55 | **96.16%** | 98.66% |
| ID8 | 79 | 88.15% | 95.91% |

### INRIA (50 epochs, val mIoU)

| ID | Best epoch | val mIoU | train mIoU @ best |
|----|----------|----------:|------------------:|
| ID1 | 40 | 83.36% | 83.17% |
| ID3 | 50 | **85.65%** | 84.53% |
| ID4 | 39 | 83.00% | 83.34% |
| ID5 | 40 | 83.89% | 82.98% |
| ID6 | 37 | 84.94% | 83.66% |
| ID7 | 40 | 85.59% | 83.96% |
| ID8 | 40 | 83.16% | 83.00% |

### DIOR (20 epochs, val mAP@0.5)

| ID | Best epoch | mAP@0.5 |
|----|----------|--------:|
| ID1 | 20 | 48.81% |
| ID3 | 17 | 61.63% |
| ID4 | 19 | 49.10% |
| ID5 | 20 | 49.15% |
| ID6 | 16 | 61.80% |
| ID7 | 17 | **62.29%** |
| ID8 | 17 | 49.61% |

## Observations

1. **Module A (Sparse SSM)** drives most downstream gains: ID3 / ID6 / ID7 are far above ID1 on DFC15 and DIOR; INRIA peaks at ID3 (85.65%).
2. **B or C alone** (ID4, ID5, ID8) stay near ID1 — graph and ARMG add little without the sparse SSM path.
3. **A is necessary** for strong transfer: A+B (ID6) and A+C (ID7) clearly beat B+C (ID8).
4. **ID9 vs sub-combinations:** Full model reference scores exceed the best trained sub-combination on each task (DFC15 97.54% vs 96.16%; INRIA 86.64% vs 85.65%; DIOR 66.27% vs 62.29%). Sub-combinations do not reproduce Full numbers under this ablation protocol — expected for compositional design, not strict leave-one-out causality.

## Paper wording (suggested)

- Avoid claiming *strong synergistic effects* from a single ablation row.
- Prefer: *each component contributes; their combination achieves the best overall transfer in the main benchmark.*
- Baseline is **EarthMamba with all three paths gated off** (same stage, channels, pretrain, downstream protocol) — not VMamba or an external backbone.

## Raw files

| File | Content |
|------|---------|
| `raw_archive.json` | Full archive of best-epoch records |
| `raw/dfc15_ID*.json` | DFC15 per-variant snapshot |
| `raw/inria_ID*.json` | INRIA per-variant snapshot |
| `raw/dior_ID*.json` | DIOR per-variant snapshot |

Re-generate tables after new runs:

```bash
python collect_results.py --results_root "$RESULTS_ROOT"
python archive_raw_results.py --results_root "$RESULTS_ROOT"
```
