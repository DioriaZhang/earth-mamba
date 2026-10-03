# Checkpoints

Pretrained checkpoints are large binary artifacts and should not be committed as ordinary Git files. This repository tree keeps only checkpoint documentation and placeholders.

## Expected Asset Contents

Static archive listing:

```text
CKPT-earthmamba/
  small/
    checkpoint.pth
    backbone.pth
  base/
    backbone.pth
    checkpoint.pth
```

`backbone.pth` is intended for downstream loading. `checkpoint.pth` contains full pretraining state for resume/audit workflows and is not required for ordinary downstream evaluation.

No download URL is included in this anonymous archive.

SHA256 values are placeholders until checkpoint binaries are packaged separately outside this 50MB code supplement.

## Loading

Use the downstream script interface:

```bash
python downstream_suite/run_patternnet.py \
  --backbone earth-mamba \
  --data_dir ${DATA_ROOT}/PatternNet \
  --ckpt /path/to/earthmamba-small/backbone.pth \
  --output_dir ${OUTPUT_DIR}/patternnet_small
```

Use `--backbone earth-mamba` for Small and `--backbone earth-mamba-b` for Base. See [../MODEL_ZOO.md](../MODEL_ZOO.md) for the full model table and archived results.
