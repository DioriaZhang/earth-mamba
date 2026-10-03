# Downstream Benchmark Suite

Self-contained downstream evaluation package. **Everything you need to run benchmarks lives inside this folder** (`run_*.py`, `tasks/`, `lib/`). Datasets, pretrained weights, and the EarthMamba source tree are **external** — you point to them with CLI flags or environment variables.

Anonymous supplement note: binary pretrained weights are not included in this folder. The `.pth` checkpoint slots are intentionally empty; pass local checkpoints with `--ckpt` when running experiments.

## What is inside this folder

```
.
├── README.md
├── requirements.txt
├── environment.yml
├── .env.example
├── bootstrap.py          # adds this folder to PYTHONPATH
├── backbone.py           # unified --backbone registry
├── common.py
├── check_setup.py
├── run_dfc15.py
├── run_patternnet.py
├── run_inria.py
├── run_deepglobe.py
├── run_dior.py
├── run_dior_r.py
├── tasks/                # per-dataset training loops
└── lib/
    ├── earthmamba_adapter.py
    ├── earth_mamba_*.py  # small / base variants
    ├── deepglobe_core.py
    ├── dior_r/           # oriented detection
    └── vendor/
        ├── baselines/    # satmae, roma, rsmamba registries
        └── baselines2/   # rvsa, satlas registries
```

Do not expect sibling folders from an old monorepo layout. After upload, only this tree is guaranteed to exist.

## Models

| `--backbone` | Scale | ~Params | Role |
|--------------|-------|---------|------|
| `earth-mamba` | small | 91M | Main-table EarthMamba-S (`dims=[96,192,384,768]`) |
| `earth-mamba-b` | base | 161M | Base-scale ablation (`dims=[128,256,512,1024]`) |
| `rvsa` | ViT-B | 86M | Baseline |
| `satlas_aerial` | Swin-B | 88M | Baseline (`satlas` alias OK) |
| `satmae` | ViT-B | 86M | Baseline |
| `roma` | Mamba | 85M | Baseline |
| `rsmamba` | ViM-h | 81M | Baseline |

The published **six-model** comparison uses `earth-mamba` (small). `earth-mamba-b` is an optional larger encoder for scale studies.

## Tasks

| Task | Script | Metric | Input |
|------|--------|--------|-------|
| DFC15 multilabel | `run_dfc15.py` | macro_mAP | 224×224 |
| PatternNet | `run_patternnet.py` | val_acc | 224×224, encoder frozen by default |
| INRIA | `run_inria.py` | val_mIoU | 512×512 patch |
| DeepGlobe | `run_deepglobe.py` | val_mIoU_6 | 512×512 patch |
| DIOR (HBB) | `run_dior.py` | mAP@0.5 | 512×512 |
| DIOR-R (OBB) | `run_dior_r.py train` | mAP@0.5 | 512×512 |

## How to use (portable)

### 1. Environment

```bash
cd downstream_suite          # or whatever you renamed this folder on the server
pip install -r requirements.txt
# optional: conda env create -f environment.yml
```

### 2. External dependencies (you provide paths)

| Dependency | How the suite finds it |
|------------|------------------------|
| **EarthMamba code** | `EARTH_MAMBA_ROOT` env var, or a clone next to this folder named `earth-mamba` |
| **Pretrained weights** | always pass `--ckpt /path/to/weights.pth` (recommended after upload) |
| **Datasets** | always pass `--data_dir` / `--data-dir` |
| **Run outputs** | pass `--output_dir` or `--work-dir` (any writable directory) |

Copy `.env.example` → `.env`, set variables, then `export` them in your shell or job script.

### 3. Smoke test

```bash
python check_setup.py
```

### 4. Run any task

Same CLI for every backbone — only `--backbone` and `--ckpt` change:

```bash
python run_dfc15.py \
  --backbone earth-mamba \
  --data_dir "$DATA_ROOT/DFC15" \
  --ckpt "$CKPT/earth-mamba-small.pth" \
  --output_dir "$OUTPUT/DFC15/earth-mamba"
```

```bash
python run_patternnet.py \
  --backbone rvsa \
  --data_dir "$DATA_ROOT/PatternNet" \
  --ckpt "$CKPT/rvsa-vit-b.pth" \
  --freeze_encoder --batch_size 64
```

```bash
python run_inria.py \
  --backbone earth-mamba \
  --data_dir "$DATA_ROOT/INRIA" \
  --ckpt "$CKPT/earth-mamba-small.pth"
```

```bash
python run_deepglobe.py \
  --backbone satmae \
  --data_dir "$DATA_ROOT/DeepGlobe" \
  --ckpt "$CKPT/satmae-vit-base.pth" \
  --output_dir "$OUTPUT/DeepGlobe/satmae"
```

```bash
python run_dior.py \
  --backbone roma \
  --data_dir "$DATA_ROOT/DIOR" \
  --ckpt "$CKPT/roma-mamba-base.pth"
```

```bash
python run_dior_r.py train \
  --backbone earth-mamba \
  --data-dir "$DATA_ROOT/DIOR" \
  --ckpt "$CKPT/earth-mamba-small.pth" \
  --work-dir "$OUTPUT/DIOR_R/earth-mamba"
```

**EarthMamba-B (base):**

```bash
python run_dfc15.py \
  --backbone earth-mamba-b \
  --data_dir "$DATA_ROOT/DFC15" \
  --ckpt "$CKPT/earth-mamba-base.pth" \
  --batch_size 32 --encoder_lr 5e-6

python run_deepglobe.py \
  --backbone earth-mamba-b \
  --data_dir "$DATA_ROOT/DeepGlobe" \
  --ckpt "$CKPT/earth-mamba-base.pth" \
  --batch_size 2
```

`--model_size auto` infers small vs base from checkpoint `embed_dim` (96 → small, 128 → base). For `earth-mamba-b`, auto resolves to **base**.

### 5. Outputs

Each run writes `results.json` (and checkpoints) under the `--output_dir` or `--work-dir` you choose.

## Implementation notes

- **Classification** (DFC15, PatternNet): all backbones go through `backbone.build_encoder()`.
- **Dense tasks** (INRIA, DeepGlobe, DIOR, DIOR-R): EarthMamba variants attach a gated pyramid adapter under `lib/`; baselines use their native multi-scale features.
- Baseline weight lookup helpers live in `lib/vendor/`; you still normally invoke only the top-level `run_*.py` scripts.

---

## Appendix: example server layout (optional)

The snippets below are **one possible deployment** on a GPU server. Paths are **not** part of this package — replace them with your own.

```bash
export DATA_ROOT=/path/to/datasets
export CKPT=/path/to/pretrained_weights
export OUTPUT=/path/to/run_outputs
export EARTH_MAMBA_ROOT=/path/to/earth-mamba   # Python package root
```

Example concrete paths (historical internal setup):

```bash
export DATA_ROOT=/hy-tmp/task
export CKPT=/hy-tmp/CKPT
export OUTPUT=/hy-tmp/downstream_results
export EARTH_MAMBA_ROOT=/hy-tmp/earth-mamba
```

If you omit `--ckpt`, the code may try built-in default paths from an old server layout. **After upload, always pass `--ckpt` explicitly** so runs do not depend on missing directories.
