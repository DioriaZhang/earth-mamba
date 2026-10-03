# Final Release Docs Report

Release settings:

```text
PROJECT_NAME = EarthMamba
PROJECT_LICENSE = Apache-2.0
RELEASE_MODE = anonymous_submission
RELEASE_VERSION = 0.1.0
```

## Created

New files:

- `LICENSE`
- `MODEL_ZOO.md`
- `CITATION.cff`
- `docs/README_COMMAND_AUDIT.md`
- `docs/FINAL_RELEASE_DOCS_REPORT.md`
- `examples/train_list.example.txt`

Deleted files:

- `LICENSE_PENDING.md`

Modified files:

- `README.md`
- `DATA.md`
- `THIRD_PARTY_NOTICES.md`
- `checkpoints/README.md`

No model, forward, pretraining, data-processing, or downstream Python source file was modified.

## License

Top-level license: Apache-2.0.

The top-level `LICENSE` uses the standard Apache License 2.0 text. It is intended for EarthMamba project-original code and does not relicense third-party code, kernels, dependencies, datasets, or weights.

Static license review found no clear top-level conflict. Package metadata was not modified:

- `earth-mamba/setup.py` has no license field to correct.
- `earth-mamba/kernels/selective_scan/setup.py` belongs to the third-party selective-scan component and has its own BSD classifier; it was not overwritten.

Third-party components still requiring manual verification include selective-scan, Mamba/Mamba3, Triton custom ops, VMamba-style utilities, downstream vendor baselines, MMDetection/MMSegmentation-style code, SatMAE, RVSA, Satlas, RoMA, RSMamba, and any vendored baseline weights.

## Model Zoo

`MODEL_ZOO.md` includes:

- EarthMamba-Tiny: config only, checkpoint not provided.
- EarthMamba-Small: Small config and checkpoint package entries.
- EarthMamba-Base: Base config and checkpoint package entries.

Checkpoint binaries are not included in this repository tree. The expected external asset layout is:

```text
CKPT-earthmamba/small/checkpoint.pth
CKPT-earthmamba/small/backbone.pth
CKPT-earthmamba/base/backbone.pth
CKPT-earthmamba/base/checkpoint.pth
```

SHA256 values are placeholders because checkpoint binaries are not included in the 50MB code supplement. No checkpoint download URL is included in the anonymous archive.

## Citation

`CITATION.cff` uses CFF 1.2.0-style fields and remains anonymous:

- authors are `Anonymous Authors`;
- no email is present;
- no affiliation is present;
- no repository URL is present;
- no DOI is present;
- no publication date is asserted.

`cffconvert` was not available in PATH, and no package was installed. A static YAML structure check was performed by inspection of the generated file. Formal public release should replace anonymous authors and add the public repository/DOI only after de-anonymization.

## README Command Audit

Command audit report: `docs/README_COMMAND_AUDIT.md`.

Summary:

- README command entries audited: 18
- HELP EXECUTED: 0
- STATICALLY VERIFIED: 17
- REMOVED: 1
- NOT VERIFIED retained in README: 0

Corrected or removed items:

- Installation commands were rewritten to repository-root-relative paths such as `earth-mamba/requirements.txt` and `earth-mamba/setup.py`.
- README links were updated from `LICENSE_PENDING.md` to `LICENSE`.
- Full checkpoint/result tables were moved to `MODEL_ZOO.md`.
- The DIOR horizontal-box README command was removed because the current `run_dior.py` dispatcher reads `--backbone`, but the earth-mamba task parser does not register that argument.

No `python <script> --help` command was executed because the entry points import project dependencies and could initialize heavy modules. Verification was static against files and `argparse` definitions.

## Privacy

Anonymous/privacy scan over the created or modified release files found no EarthMamba author names, affiliations, emails, local drive-letter paths, server-root paths, private usernames, non-anonymous GitHub links, cloud-share links, checkpoint save paths, or real data directories.

The only scan hits were `mAP@0.5` metric names in `MODEL_ZOO.md`; these are not emails or identity leaks.

Third-party author names in notices are allowed because they are required provenance for third-party code.

## Consistency Checks

- Top-level `LICENSE` exists.
- Top-level `LICENSE_PENDING.md` does not exist.
- Top-level `MODEL_ZOO.md` exists.
- Top-level `CITATION.cff` exists and is anonymous.
- README relative links checked in this pass exist.
- README Python command script paths checked in this pass exist.
- README command arguments were statically checked against current code.
- README does not contain private absolute paths.
- README does not contain real author identity.
- MODEL_ZOO result values match the archived records supplied for Small and Base.
- MODEL_ZOO does not invent public download links and keeps checkpoint SHA256 fields as placeholders.
- THIRD_PARTY_NOTICES remains present.
- Third-party source headers were not modified.
- Core experiment code was not modified.

`git diff --stat` could not be used because `project/` is not a Git repository in this workspace.

## Final Verdict

READY WITH DOCUMENTATION WARNINGS

Warnings:

- DIOR horizontal-box training/evaluation command is intentionally omitted from README pending a downstream wrapper/parser cleanup.
- `CITATION.cff` was not validated with `cffconvert` because the tool is not installed.
- Third-party license provenance still needs manual legal verification before non-anonymous public release.
