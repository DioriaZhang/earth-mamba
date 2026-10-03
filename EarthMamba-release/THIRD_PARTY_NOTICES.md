# Third-Party Notices

EarthMamba project-original code is licensed under Apache-2.0 through the top-level [LICENSE](LICENSE). That license does not relicense third-party code, kernels, dependencies, datasets, or weights.

This file records third-party components identified by static repository inspection. It is not a complete legal review.

## Core Dependencies and Components

| Component | Evidence in repository | License status |
|---|---|---|
| PyTorch | requirements/setup metadata | External dependency; verify upstream license |
| torchvision / torchaudio | requirements | External dependency; verify upstream license |
| timm | requirements | External dependency; verify upstream license |
| Triton | requirements and custom ops | External dependency; verify upstream license |
| einops | requirements | External dependency; verify upstream license |
| selective scan kernels | `earth-mamba/kernels/selective_scan` | BSD classifier present in setup metadata; keep upstream notices |
| Mamba / Mamba3 code paths | `earth-mamba/earth_mamba/modules/mamba3.py`, `earth-mamba/earth_mamba/ops/triton/mamba3` | Needs manual license verification |
| Dao AI Lab / GoombaLab code | copyright notices in Mamba3/Triton files | Needs manual license verification |
| Tri Dao / Albert Gu code | selective-scan and Triton notices | Needs manual license verification beyond detected notices |
| NVIDIA helper code | selective-scan CUDA header | Keep NVIDIA notice and verify redistribution conditions |
| VMamba-style utilities | `earth-mamba/earth_mamba/utils/vmamba_core.py` and related SSM code | Needs manual license verification |

## Downstream and Baseline Code

| Component | Evidence in repository | License status |
|---|---|---|
| MMDetection-style detection code | downstream detection wrappers and task code | Needs manual license verification |
| MMSegmentation-style decoder/task conventions | segmentation wrappers and UPerNet naming | Needs manual license verification |
| SatMAE baseline | downstream vendor/baseline references | Needs manual license verification |
| RVSA baseline | downstream vendor/baseline references | Needs manual license verification |
| Satlas baseline | downstream vendor/baseline references | Needs manual license verification |
| RoMA baseline | downstream vendor/baseline references | Needs manual license verification |
| RSMamba baseline | downstream vendor/baseline references | Needs manual license verification |
| Vendored baseline weights | `downstream_suite/lib/vendor/**/weights/*.pth` if retained | Do not redistribute unless license and provenance are confirmed |

## Data and Weights

- Raw datasets are not redistributed by this repository.
- The 3.67M pretraining manifest is not redistributed.
- Third-party weights are not redistributed by default.
- EarthMamba Small/Base checkpoints are release assets and should be accompanied by explicit terms and checksums.

## Release Policy

- Keep upstream copyright headers intact.
- Add upstream license files or links for every vendored code subtree before non-anonymous public release.
- Do not commit large binary weights, datasets, LMDBs, tar shards, or private training manifests.
- Remove or replace private paths in examples, docs, logs, and saved command lines before public release.
