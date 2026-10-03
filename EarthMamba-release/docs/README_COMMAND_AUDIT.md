# README Command Audit

Scope: top-level `README.md` only. Commands were checked against current repository files and static `argparse` definitions. No training, pretraining, downstream fine-tuning, or model import was executed.

`cffconvert` was not found in PATH, so citation validation is reported in `docs/FINAL_RELEASE_DOCS_REPORT.md` as static/manual YAML structure verification.

| Section | Command purpose | Script/path | Path exists | Arguments verified | Execution status | Action |
|---|---|---|---|---|---|---|
| Installation | Install core requirements | `earth-mamba/requirements.txt` | Yes | Path verified | STATICALLY VERIFIED | Rewritten to root-relative path |
| Installation | Install selective-scan kernel | `earth-mamba/kernels/selective_scan` | Yes | Editable install path verified | STATICALLY VERIFIED | Rewritten to root-relative path |
| Installation | Install EarthMamba package | `earth-mamba/setup.py` | Yes | Editable install path verified | STATICALLY VERIFIED | Rewritten to root-relative path |
| Installation | Install downstream requirements | `downstream_suite/requirements.txt` | Yes | Path verified | STATICALLY VERIFIED | Added as optional dependency command |
| Installation | Install data-processing requirements | `data_process/requirements.txt` | Yes | Path verified | STATICALLY VERIFIED | Added as optional dependency command |
| Minimal model construction | Instantiate `EarthMamba` | `earth-mamba/earth_mamba/models/earth_mamba.py` | Yes | Constructor keyword names statically checked | STATICALLY VERIFIED | Kept |
| Pretraining | Small pretraining template | `train/pretrain_mae.py` | Yes | `--data_list`, `--output_dir`, `--model_size`, `--patch_size`, `--batch_size`, `--grad_accum`, `--epochs`, `--warmup_epochs`, `--amp_dtype` verified | STATICALLY VERIFIED | Kept with placeholder paths |
| Pretraining | Base pretraining template | `train/pretrain_mae_base.py` | Yes | Wrapper injects Base defaults; forwarded arguments verified in `train/pretrain_mae.py` | STATICALLY VERIFIED | Kept with placeholder paths |
| Data processing | Downstream preprocessing template | `data_process/preprocess_downstream.py` | Yes | positional `input_dirs` and `--output_root` verified | STATICALLY VERIFIED | Added |
| Data processing | Cluster feature build template | `data_process/pipeline_cluster_1_build.py` | Yes | `--globe230k`, `--luojia`, `--downstream_dirs`, `--feature_root`, `--weight_path` verified | STATICALLY VERIFIED | Added |
| Data processing | Pool filtering template | `data_process/pipeline_cluster_2_filter.py` | Yes | `--pool_dir`, `--feature_root`, `--weight_path`, `--output_dir` verified | STATICALLY VERIFIED | Added |
| Downstream | Environment variable placeholders | shell variables only | N/A | Placeholder syntax reviewed | STATICALLY VERIFIED | Kept |
| Downstream | DFC15 template | `downstream_suite/run_dfc15.py` | Yes | `--backbone`, `--data_dir`, `--ckpt`, `--output_dir` verified | STATICALLY VERIFIED | Kept |
| Downstream | PatternNet template | `downstream_suite/run_patternnet.py` | Yes | `--backbone`, `--data_dir`, `--ckpt`, `--output_dir` verified | STATICALLY VERIFIED | Kept |
| Downstream | INRIA template | `downstream_suite/run_inria.py` -> `tasks/inria_earth.py` | Yes | dispatcher needs `--backbone`; task parser arguments `--data_dir`, `--ckpt`, `--output_dir` verified | STATICALLY VERIFIED | Kept |
| Downstream | DeepGlobe template | `downstream_suite/run_deepglobe.py` | Yes | `--backbone`, `--data_dir`, `--ckpt`, `--output_dir` verified | STATICALLY VERIFIED | Kept |
| Downstream | DIOR-R train template | `downstream_suite/run_dior_r.py` -> `lib/dior_r/run.py` | Yes | global `--data-dir`, `--backbone`, `--ckpt`, `--project-root`; subcommand `train`; `--work-dir` verified | STATICALLY VERIFIED | Kept |
| Downstream | DIOR HBB template | `downstream_suite/run_dior.py` -> `tasks/dior_earth.py` | Yes | dispatcher reads `--backbone`, but earth-mamba task parser does not register `--backbone` | REMOVED | Removed from README command templates pending wrapper cleanup |

## Summary

- README command entries audited: 18
- HELP EXECUTED: 0
- STATICALLY VERIFIED: 17
- REMOVED: 1
- NOT VERIFIED retained in README: 0

## Notes

- No `python <script> --help` command was executed because the scripts import project dependencies and some entry points may initialize heavy modules.
- The DIOR horizontal-box result remains documented in `MODEL_ZOO.md`, but its README command was removed because the current wrapper/parser combination is not statically clean.
- All public command paths use repository-relative files plus `/path/to/...`, `${DATA_ROOT}`, `${OUTPUT_DIR}`, or `${CKPT_*}` placeholders.
