# EarthMamba Ablation Suite

Self-contained package for **compositional ablation** of three EarthMamba blocks at inference time:

| Code | Module | Flag |
|------|--------|------|
| A | Sparse SSM path | `use_sparse_ssm` |
| B | Latent Graph branch | `use_graph` |
| C | ARMG denoising gate | `use_armg` |

All training and collection scripts live **inside this folder**. Datasets, checkpoints, and the EarthMamba Python package are external — pass paths via CLI or environment variables.

## What is inside this folder

```
.
├── README.md
├── requirements.txt
├── .env.example
├── env.example.sh
├── variant_flags.py       # ID1–ID9 module combinations
├── run_variant.py         # single job launcher
├── run_matrix.sh          # full matrix (ID1–ID8 × tasks)
├── collect_results.py     # aggregate training outputs → results/
├── archive_raw_results.py # snapshot best-epoch records → results/raw/
├── preflight.py           # path / import checks
├── validate.py            # dry-run command builder
├── task_dfc15.py
├── task_inria.py
├── task_dior.py
├── configs/
│   ├── variants.yaml
│   ├── dfc15.yaml
│   ├── inria.yaml
│   └── dior.yaml
├── lib/                   # task trainers + earth_mamba hooks
└── results/
    ├── RESULTS.md         # archived metrics (English)
    ├── raw_archive.json
    └── raw/               # per-run JSON snapshots
```

## Variant design

Eight variants are **trained** (ID1, ID3–ID8). **ID9 (Full)** is **not re-trained** — metrics are copied from the main downstream benchmark table.

| ID | Sparse (A) | Graph (B) | ARMG (C) | Label |
|----|:----------:|:---------:|:--------:|-------|
| ID1 | | | | Pure baseline (all off) |
| ID3 | ✓ | | | A only |
| ID4 | | ✓ | | B only |
| ID5 | | | ✓ | C only |
| ID6 | ✓ | ✓ | | A+B |
| ID7 | ✓ | | ✓ | A+C |
| ID8 | | ✓ | ✓ | B+C |
| ID9 | ✓ | ✓ | ✓ | Full (reference only) |

**Tasks:** DFC15 (macro_mAP), INRIA (val_mIoU), DIOR (mAP@0.5).

**Job count:** 7 variants × 3 tasks = **21 training jobs** (ID9 excluded).

Gates are toggled at **inference** (modules are built and weights loaded; branches are enabled/disabled via flags). ID1 is **not** a separately trained VMamba backbone — it is the same checkpoint with all three paths disabled.

## How to use (portable)

### 1. Environment

```bash
cd ablation_suite
pip install -r requirements.txt
```

### 2. External paths

| Variable | Purpose |
|----------|---------|
| `EARTH_MAMBA_ROOT` | Directory containing the `earth_mamba/` Python package |
| `CKPT` | Pretrained EarthMamba-small checkpoint (`.pth`) |
| `DATA_INRIA`, `DATA_DFC15`, `DATA_DIOR` | Dataset roots |
| `RESULTS_ROOT` | Writable directory for new training outputs (default: `./outputs`) |

Copy `.env.example` → `.env`, export variables, or `source env.example.sh` after editing paths.

### 3. Preflight

```bash
python preflight.py
python validate.py
```

### 4. Run one variant

```bash
python run_variant.py --task dfc15 --variant ID3 \
  --data_dir "$DATA_DFC15" --ckpt "$CKPT"

python run_variant.py --task inria --variant ID7 \
  --data_dir "$DATA_INRIA" --ckpt "$CKPT"
```

### 5. Run full matrix

```bash
bash run_matrix.sh
# optional DIOR only: RUN_DIOR=1 bash run_matrix.sh
```

### 6. Collect metrics

After training, each job writes `{RESULTS_ROOT}/{task}/{variant}/results.json`.

```bash
python collect_results.py --results_root "$RESULTS_ROOT"
python archive_raw_results.py --results_root "$RESULTS_ROOT"
```

Outputs land in `results/` (`ablation_summary.json`, `ablation_table.md`, `raw/*.json`).

### 7. Read archived results

See **`results/RESULTS.md`** for the completed 2026-06-26 run (tables + interpretation). Raw per-run JSON is under `results/raw/`.

## Implementation notes

- Same pretrained checkpoint and downstream protocol for all variants; only module gates differ.
- Sparse SSM (A) dominates transfer gains; B/C alone add little without A.
- ID9 numbers are **references** from the six-model main table, not reproduced inside this ablation protocol (epochs / tuning may differ slightly from ID7).

---

## Appendix: example server layout (optional)

```bash
export DATA_DFC15=/path/to/DFC15
export DATA_INRIA=/path/to/INRIA
export DATA_DIOR=/path/to/DIOR
export CKPT=/path/to/earth-mamba-small.pth
export RESULTS_ROOT=/path/to/ablation_outputs
export EARTH_MAMBA_ROOT=/path/to/earth-mamba
```

Always pass explicit paths after upload; do not rely on hard-coded defaults inside the code.
