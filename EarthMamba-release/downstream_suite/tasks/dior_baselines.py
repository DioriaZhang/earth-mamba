from __future__ import annotations

import argparse

from tasks.dior_baselines_backbones import BASELINE_DIOR_BACKBONES, build_baseline_encoder
from tasks.dior_earth_task import add_dior_args, run_dior_training


def main() -> None:
    p = argparse.ArgumentParser("downstream_suite — DIOR baselines")
    p.add_argument("--backbone", default="satmae", choices=list(BASELINE_DIOR_BACKBONES))
    add_dior_args(p)
    p.add_argument("--strict_load_report", action="store_true")
    p.add_argument("--ssm_version", default="mamba3")
    p.add_argument("--encoder_embed_dim", type=int, default=None)
    p.add_argument("--encoder_depth", type=int, default=None)
    p.add_argument("--encoder_num_heads", type=int, default=None)
    p.add_argument("--encoder_patch_size", type=int, default=None)
    p.add_argument("--rsmamba_size", default="base", choices=["base", "large", "huge"])
    p.set_defaults(amp_dtype="none")
    args = p.parse_args()

    def build_encoder_fn(a):
        return build_baseline_encoder(
            a.backbone,
            a.img_size,
            a.ckpt,
            ssm_version=a.ssm_version,
            strict_load_report=a.strict_load_report,
            embed_dim=a.encoder_embed_dim,
            depth=a.encoder_depth,
            num_heads=a.encoder_num_heads,
            patch_size=a.encoder_patch_size,
            rsmamba_size=a.rsmamba_size,
        )

    run_dior_training(
        args,
        build_encoder_fn,
        task_name=f"run_dior.py [{args.backbone}]",
        earthmamba_param_groups=False,
    )


if __name__ == "__main__":
    main()
