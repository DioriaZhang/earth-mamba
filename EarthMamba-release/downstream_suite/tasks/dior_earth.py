from __future__ import annotations

import argparse
import sys

import backbone as bb
from lib.earthmamba_adapter import build_earthmamba_encoder, ensure_registered
from lib.earth_mamba_variants import resolve_model_size
from tasks.dior_earth_task import add_dior_args, run_dior_training


def _argv_backbone() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--backbone" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "earth-mamba"


def main() -> None:
    p = argparse.ArgumentParser("downstream_suite — DIOR earth-mamba")
    add_dior_args(p)
    p.add_argument(
        "--dense_adapter",
        default="semantic_pyramid",
        choices=["semantic_pyramid", "gated_pyramid", "detail_pyramid", "none"],
    )
    p.add_argument("--debug_shapes", action="store_true")
    p.add_argument("--ssm_version", default="mamba3")
    p.set_defaults(
        amp_dtype="none",
        adapter_lr=2e-4,
        warmup_epochs=2,
        eval_interval=5,
        score_thresh=0.05,
    )
    args = p.parse_args()
    ensure_registered()

    backbone_name = bb.normalize_backbone(_argv_backbone())
    if not args.ckpt:
        args.ckpt = bb.resolve_ckpt(backbone_name, None)
    model_size = resolve_model_size(backbone_name, "auto")

    def build_encoder_fn(a):
        return build_earthmamba_encoder(
            a.ckpt,
            a.img_size,
            model_size=model_size,
            ssm_version=a.ssm_version,
            dense_adapter=a.dense_adapter,
            debug_shapes=a.debug_shapes,
        )

    run_dior_training(
        args,
        build_encoder_fn,
        task_name=f"run_dior.py [{backbone_name}]",
        earthmamba_param_groups=True,
    )


if __name__ == "__main__":
    main()
