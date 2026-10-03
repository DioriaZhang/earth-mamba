"""EarthMamba classification encoder (small / base) for DFC15 & PatternNet."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.earth_mamba_variants import MODEL_PRESETS, resolve_model_size
from lib.earthmamba_adapter import load_checkpoint_state_dict, infer_embed_dim_from_state_dict


def _ensure_earthmamba_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for cand in (
        root.parent / "earth-mamba",
        root.parent.parent / "earth-mamba",  # 仓库根 earth-mamba（两文件夹布局）
        Path("/hy-tmp/earth-mamba"),
        Path("/root/earth-mamba"),
    ):
        if (cand / "earth_mamba").is_dir():
            s = str(cand.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            return


def _resolve_dims_depths(backbone: str, ckpt: Optional[str], model_size: str) -> tuple[list[int], list[int], str]:
    size = resolve_model_size(backbone, model_size)
    if size in MODEL_PRESETS:
        depths, dims = MODEL_PRESETS[size]
        return list(dims), list(depths), size
    if ckpt:
        sd = load_checkpoint_state_dict(ckpt)
        embed_dim = infer_embed_dim_from_state_dict(sd)
        for tag, (_, dims) in MODEL_PRESETS.items():
            if embed_dim == dims[0]:
                depths, _ = MODEL_PRESETS[tag]
                return list(dims), list(depths), tag
    depths, dims = MODEL_PRESETS["small"]
    return list(dims), list(depths), "small"


class EarthMambaClassifierEncoder(nn.Module):
    """4-stage encoder with ``out_dims`` for GAP + linear head."""

    out_dims: List[int]

    def __init__(
        self,
        img_size: int,
        dims: List[int],
        depths: List[int],
        *,
        ssm_version: str = "mamba3",
        use_armg: bool = True,
        use_graph: bool = True,
    ):
        super().__init__()
        _ensure_earthmamba_path()
        from earth_mamba.models.earth_mamba import EarthMamba
        from earth_mamba.models.earth_mamba_block import LayerNorm as EM_LayerNorm

        self.backbone = EarthMamba(
            patch_size=16,
            in_chans=3,
            num_classes=1,
            depths=depths,
            dims=dims,
            ssm_d_state=64 if ssm_version == "mamba3" else 16,
            ssm_ratio=2.0,
            ssm_version=ssm_version,
            ssm_headdim=64,
            mlp_ratio=4.0,
            drop_path_rate=0.0,
            norm_layer="ln",
            posembed=True,
            imgsize=img_size,
            downsample_version="v3",
            use_armg=use_armg,
            use_graph=use_graph,
        )
        self.channel_first = self.backbone.channel_first
        self.outnorms = nn.ModuleList([
            EM_LayerNorm(dims[i], channel_first=self.channel_first) for i in range(4)
        ])
        self.out_dims = list(dims)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed is not None:
            pe = self.backbone.pos_embed
            if not self.channel_first:
                pe = pe.permute(0, 2, 3, 1)
            x = x + pe
        outs = []
        for i, layer in enumerate(self.backbone.layers):
            x = layer.blocks(x)
            o = self.outnorms[i](x)
            if not self.channel_first:
                o = o.permute(0, 3, 1, 2)
            outs.append(o.contiguous())
            x = layer.downsample(x)
        return outs


def _load_pretrained(encoder: EarthMambaClassifierEncoder, ckpt: str) -> None:
    print(f"  [EarthMamba] loading backbone: {ckpt}", flush=True)
    sd = load_checkpoint_state_dict(ckpt)
    model_sd = encoder.backbone.state_dict()
    cleaned = {}
    skipped_shape: list[str] = []
    for raw_key, value in sd.items():
        key = str(raw_key)
        for prefix in ("module.", "encoder_without_ddp.", "encoder.", "backbone.", "core."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key.startswith(("decoder.", "classifier.", "mask_token")) or key == "mask_value":
            continue
        if key == "pos_embed" and key in model_sd and tuple(value.shape) != tuple(model_sd[key].shape):
            if value.dim() == 4 and model_sd[key].dim() == 4 and value.shape[1] == model_sd[key].shape[1]:
                value = F.interpolate(
                    value.float(),
                    size=model_sd[key].shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).to(value.dtype)
        if key in model_sd and tuple(model_sd[key].shape) == tuple(value.shape):
            cleaned[key] = value
        elif key in model_sd:
            skipped_shape.append(f"{key}: ckpt{tuple(value.shape)} model{tuple(model_sd[key].shape)}")
    missing, unexpected = encoder.backbone.load_state_dict(cleaned, strict=False)
    ratio = len(cleaned) / max(1, len(model_sd))
    print(
        f"    loaded_keys={len(cleaned)}/{len(model_sd)} ({ratio:.1%}) "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if ratio < 0.85:
        print("    *** WARN: low load ratio — check small vs base ckpt ***", flush=True)


def build_classifier_encoder(
    backbone: str,
    img_size: int,
    ckpt: Optional[str],
    *,
    model_size: str = "auto",
    ssm_version: str = "mamba3",
):
    dims, depths, tag = _resolve_dims_depths(backbone, ckpt, model_size)
    print(f"  [EarthMamba] classifier encoder size={tag} dims={dims}", flush=True)
    enc = EarthMambaClassifierEncoder(
        img_size, dims, depths, ssm_version=ssm_version,
    )
    if ckpt:
        _load_pretrained(enc, ckpt)
    else:
        print("  [EarthMamba] random init", flush=True)
    return enc
