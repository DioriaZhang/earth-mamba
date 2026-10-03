"""Earth-Mamba dense feature adapter for DIOR-R (earth-mamba only).

Ports the v1 / verify_dior detail+gated pyramid idea: fuse a CNN stem (stride
4/8/16/32) with native Mamba stage features so small OBB objects get usable P2–P5.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.earth_mamba_variants import MODEL_PRESETS
from lib.earthmamba_adapter import load_checkpoint_state_dict, infer_embed_dim_from_state_dict

SMALL_DEPTHS, SMALL_DIMS = MODEL_PRESETS["small"]
BASE_DEPTHS, BASE_DIMS = MODEL_PRESETS["base"]
ADAPTER_CHOICES = ("gated_pyramid", "detail_pyramid", "semantic_pyramid", "none")


def _ensure_earthmamba_path(project_root: Path) -> None:
    candidates = [
        project_root / "earth-mamba",
        project_root.parent / "earth-mamba",  # 仓库根 earth-mamba（两文件夹布局）
        project_root / "downstream_code" / "earth-mamba",
        Path("/hy-tmp/earth-mamba"),
        Path("/root/earth-mamba"),
    ]
    for cand in candidates:
        if (cand / "earth_mamba").is_dir():
            root = str(cand.resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            return


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    for groups in (max_groups, 16, 8, 4, 2, 1):
        if groups <= channels and channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _conv_gn_act(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        _gn(out_ch),
        nn.SiLU(inplace=True),
    )


class GatedFusion(nn.Module):
    def __init__(self, detail_ch: int, semantic_ch: int, out_ch: int):
        super().__init__()
        self.detail = nn.Conv2d(detail_ch, out_ch, 1, bias=False)
        self.semantic = nn.Conv2d(semantic_ch, out_ch, 1, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(out_ch * 2, out_ch, 1), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        if semantic.shape[-2:] != detail.shape[-2:]:
            semantic = F.interpolate(
                semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False,
            )
        detail_out = self.detail(detail)
        semantic_out = self.semantic(semantic)
        gate = self.gate(torch.cat([detail_out, semantic_out], dim=1))
        return self.out(detail_out + gate * semantic_out)


class EarthMambaFeatureAdapter(nn.Module):
    """CNN detail pyramid + Mamba semantics → dense P2–P5 (channels 96/192/384/768)."""

    out_dims = list(SMALL_DIMS)

    def __init__(self, mode: str = "gated_pyramid"):
        super().__init__()
        self.mode = mode
        if mode == "none":
            return
        if mode not in ADAPTER_CHOICES:
            raise ValueError(f"unknown earth_adapter={mode!r}; choices={ADAPTER_CHOICES}")

        if mode != "semantic_pyramid":
            self.stem_s2 = _conv_gn_act(3, 48, stride=2)
            self.stem_s4 = _conv_gn_act(48, SMALL_DIMS[0], stride=2)
            self.stem_s8 = _conv_gn_act(SMALL_DIMS[0], SMALL_DIMS[1], stride=2)
            self.stem_s16 = _conv_gn_act(SMALL_DIMS[1], SMALL_DIMS[2], stride=2)
            self.stem_s32 = _conv_gn_act(SMALL_DIMS[2], SMALL_DIMS[3], stride=2)

        self.sem_p2 = nn.Conv2d(SMALL_DIMS[0], SMALL_DIMS[0], 1, bias=False)
        self.sem_p3 = nn.ModuleList([
            nn.Conv2d(SMALL_DIMS[0], SMALL_DIMS[1], 1, bias=False),
            nn.Conv2d(SMALL_DIMS[1], SMALL_DIMS[1], 1, bias=False),
        ])
        self.sem_p4 = nn.ModuleList([
            nn.Conv2d(SMALL_DIMS[0], SMALL_DIMS[2], 1, bias=False),
            nn.Conv2d(SMALL_DIMS[2], SMALL_DIMS[2], 1, bias=False),
        ])
        self.sem_p5 = nn.ModuleList([
            nn.Conv2d(SMALL_DIMS[1], SMALL_DIMS[3], 1, bias=False),
            nn.Conv2d(SMALL_DIMS[2], SMALL_DIMS[3], 1, bias=False),
            nn.Conv2d(SMALL_DIMS[3], SMALL_DIMS[3], 1, bias=False),
        ])

        if mode == "gated_pyramid":
            self.fuse_p2 = GatedFusion(SMALL_DIMS[0], SMALL_DIMS[0], SMALL_DIMS[0])
            self.fuse_p3 = GatedFusion(SMALL_DIMS[1], SMALL_DIMS[1], SMALL_DIMS[1])
            self.fuse_p4 = GatedFusion(SMALL_DIMS[2], SMALL_DIMS[2], SMALL_DIMS[2])
            self.fuse_p5 = GatedFusion(SMALL_DIMS[3], SMALL_DIMS[3], SMALL_DIMS[3])
        else:
            self.fuse_p2 = _conv_gn_act(SMALL_DIMS[0], SMALL_DIMS[0])
            self.fuse_p3 = _conv_gn_act(SMALL_DIMS[1], SMALL_DIMS[1])
            self.fuse_p4 = _conv_gn_act(SMALL_DIMS[2], SMALL_DIMS[2])
            self.fuse_p5 = _conv_gn_act(SMALL_DIMS[3], SMALL_DIMS[3])
        if mode == "semantic_pyramid":
            self.sem_only_p2 = _conv_gn_act(SMALL_DIMS[0], SMALL_DIMS[0])
            self.sem_only_p3 = _conv_gn_act(SMALL_DIMS[1], SMALL_DIMS[1])
            self.sem_only_p4 = _conv_gn_act(SMALL_DIMS[2], SMALL_DIMS[2])
            self.sem_only_p5 = _conv_gn_act(SMALL_DIMS[3], SMALL_DIMS[3])

    def _sum_to(self, parts: Sequence[torch.Tensor], size: torch.Size) -> torch.Tensor:
        out = None
        for part in parts:
            if part.shape[-2:] != tuple(size):
                part = F.interpolate(part, size=size, mode="bilinear", align_corners=False)
            out = part if out is None else out + part
        return out

    def forward(self, image: torch.Tensor, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.mode == "none":
            return feats
        if self.mode == "semantic_pyramid":
            h, w = image.shape[-2:]
            p2_size = (max(1, (h + 3) // 4), max(1, (w + 3) // 4))
            p3_size = (max(1, (h + 7) // 8), max(1, (w + 7) // 8))
            p4_size = (max(1, (h + 15) // 16), max(1, (w + 15) // 16))
            p5_size = (max(1, (h + 31) // 32), max(1, (w + 31) // 32))
            p2 = F.interpolate(self.sem_p2(feats[0]), size=p2_size, mode="bilinear", align_corners=False)
            p3 = self._sum_to([self.sem_p3[0](feats[0]), self.sem_p3[1](feats[1])], p3_size)
            p4 = self._sum_to([self.sem_p4[0](feats[0]), self.sem_p4[1](feats[2])], p4_size)
            p5 = self._sum_to(
                [self.sem_p5[0](feats[1]), self.sem_p5[1](feats[2]), self.sem_p5[2](feats[3])],
                p5_size,
            )
            return [
                self.sem_only_p2(p2),
                self.sem_only_p3(p3),
                self.sem_only_p4(p4),
                self.sem_only_p5(p5),
            ]

        s2 = self.stem_s2(image)
        p2_detail = self.stem_s4(s2)
        p3_detail = self.stem_s8(p2_detail)
        p4_detail = self.stem_s16(p3_detail)
        p5_detail = self.stem_s32(p4_detail)

        p2_sem = self.sem_p2(feats[0])
        p3_sem = self._sum_to(
            [self.sem_p3[0](feats[0]), self.sem_p3[1](feats[1])], p3_detail.shape[-2:],
        )
        p4_sem = self._sum_to(
            [self.sem_p4[0](feats[0]), self.sem_p4[1](feats[2])], p4_detail.shape[-2:],
        )
        p5_sem = self._sum_to(
            [self.sem_p5[0](feats[1]), self.sem_p5[1](feats[2]), self.sem_p5[2](feats[3])],
            p5_detail.shape[-2:],
        )

        if self.mode == "gated_pyramid":
            return [
                self.fuse_p2(p2_detail, p2_sem),
                self.fuse_p3(p3_detail, p3_sem),
                self.fuse_p4(p4_detail, p4_sem),
                self.fuse_p5(p5_detail, p5_sem),
            ]
        return [
            self.fuse_p2(
                p2_detail + F.interpolate(p2_sem, p2_detail.shape[-2:], mode="bilinear", align_corners=False)
            ),
            self.fuse_p3(p3_detail + p3_sem),
            self.fuse_p4(p4_detail + p4_sem),
            self.fuse_p5(p5_detail + p5_sem),
        ]


class EarthMambaDenseEncoder(nn.Module):
    """Earth-Mamba backbone + optional dense adapter (DIOR-R earth-mamba only)."""

    out_dims: List[int]

    def __init__(
        self,
        img_size: int,
        project_root: Path,
        dims: Sequence[int],
        depths: Sequence[int],
        ssm_version: str = "mamba3",
        dense_adapter: str = "gated_pyramid",
        debug_shapes: bool = False,
    ):
        super().__init__()
        self.dims = [int(d) for d in dims]
        self.depths = [int(d) for d in depths]
        _ensure_earthmamba_path(project_root)
        from earth_mamba.models.earth_mamba import EarthMamba
        from earth_mamba.models.earth_mamba_block import LayerNorm as EM_LayerNorm

        self.core = EarthMamba(
            patch_size=16,
            in_chans=3,
            num_classes=1,
            depths=self.depths,
            dims=self.dims,
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
        )
        self.channel_first = self.core.channel_first
        self.outnorms = nn.ModuleList([
            EM_LayerNorm(self.dims[i], channel_first=self.channel_first) for i in range(4)
        ])
        self.dense_adapter = EarthMambaFeatureAdapter(dense_adapter)
        self.out_dims = list(self.dense_adapter.out_dims)
        self.adapter_mode = dense_adapter
        self.debug_shapes = debug_shapes
        self._shape_logged = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        image = x
        x = self.core.patch_embed(x)
        if self.core.pos_embed is not None:
            pe = self.core.pos_embed
            if not self.channel_first:
                pe = pe.permute(0, 2, 3, 1)
            x = x + pe
        feats: List[torch.Tensor] = []
        for i, layer in enumerate(self.core.layers):
            x = layer.blocks(x)
            y = self.outnorms[i](x)
            if not self.channel_first:
                y = y.permute(0, 3, 1, 2)
            feats.append(y.contiguous())
            x = layer.downsample(x)
        outs = self.dense_adapter(image, feats)
        if self.debug_shapes and not self._shape_logged:
            shapes = [tuple(tensor.shape) for tensor in outs]
            print(f"  [earth-adapter] mode={self.adapter_mode} feature_shapes={shapes}", flush=True)
            self._shape_logged = True
        return outs


def load_earthmamba_weights(encoder: EarthMambaDenseEncoder, ckpt: str) -> None:
    print(f"  [EarthMamba+adapter] 加载预训练权重: {ckpt}")
    try:
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(ckpt, map_location="cpu")
    sd = raw.get("model", raw.get("state_dict", raw))
    model_sd = encoder.core.state_dict()
    cleaned: dict = {}
    skipped_shape: list[str] = []
    for raw_key, value in sd.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "encoder_without_ddp.", "encoder.", "backbone.", "core."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        if key.startswith(("decoder.", "classifier.", "mask_token")) or key == "mask_value":
            continue
        if key == "pos_embed" and key in model_sd and tuple(value.shape) != tuple(model_sd[key].shape):
            if value.dim() == 4 and model_sd[key].dim() == 4 and value.shape[1] == model_sd[key].shape[1]:
                value = F.interpolate(
                    value.float(), size=model_sd[key].shape[-2:],
                    mode="bicubic", align_corners=False,
                ).to(value.dtype)
        if key in model_sd:
            if tuple(model_sd[key].shape) != tuple(value.shape):
                skipped_shape.append(f"{key}: ckpt{tuple(value.shape)} model{tuple(model_sd[key].shape)}")
                continue
            cleaned[key] = value
    missing, unexpected = encoder.core.load_state_dict(cleaned, strict=False)
    print(
        f"    loaded_keys={len(cleaned)} missing={len(missing)} unexpected={len(unexpected)} "
        f"skipped_shape={len(skipped_shape)}"
    )


def _resolve_dims_depths(model_size: str, ckpt: str | Path) -> tuple[list[int], list[int], str]:
    size = (model_size or "auto").strip().lower()
    if size in MODEL_PRESETS:
        depths, dims = MODEL_PRESETS[size]
        return list(dims), list(depths), size
    sd = load_checkpoint_state_dict(str(ckpt))
    embed_dim = infer_embed_dim_from_state_dict(sd)
    for tag, (_, dims) in MODEL_PRESETS.items():
        if embed_dim == dims[0]:
            depths, _ = MODEL_PRESETS[tag]
            return list(dims), list(depths), tag
    depths, dims = MODEL_PRESETS["small"]
    return list(dims), list(depths), "small"


def build_earth_mamba_encoder(
    checkpoint: str | Path,
    project_root: Path,
    image_size: int,
    *,
    model_size: str = "auto",
    ssm_version: str = "mamba3",
    earth_adapter: str = "gated_pyramid",
    debug_shapes: bool = False,
) -> EarthMambaDenseEncoder:
    dims, depths, tag = _resolve_dims_depths(model_size, checkpoint)
    print(f"  [EarthMamba+adapter] size={tag} dims={dims}", flush=True)
    encoder = EarthMambaDenseEncoder(
        img_size=image_size,
        project_root=project_root,
        dims=dims,
        depths=depths,
        ssm_version=ssm_version,
        dense_adapter=earth_adapter,
        debug_shapes=debug_shapes,
    )
    load_earthmamba_weights(encoder, str(checkpoint))
    return encoder


def _add_decay_groups(
    groups: list,
    params: Iterable[tuple[str, nn.Parameter]],
    lr: float,
    weight_decay: float,
) -> None:
    decay, no_decay = [], []
    for name, param in params:
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "bn" in name.lower()
            or "gn" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})


def build_earth_mamba_param_groups(
    model: nn.Module,
    *,
    lr: float,
    encoder_lr: float,
    adapter_lr: float,
    weight_decay: float,
) -> tuple[list, list[float]]:
    """Return (optimizer param groups, base_lrs) for cosine schedule."""
    core, adapter, detector = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".core." in f".{name}." or name.startswith("core."):
            core.append((name, param))
        elif "dense_adapter" in name or "outnorms" in name:
            adapter.append((name, param))
        else:
            detector.append((name, param))
    groups: list = []
    _add_decay_groups(groups, core, encoder_lr, weight_decay)
    _add_decay_groups(groups, adapter, adapter_lr, weight_decay)
    _add_decay_groups(groups, detector, lr, weight_decay)
    print(
        f"  [earth-adapter] lr groups: encoder={encoder_lr:g} adapter={adapter_lr:g} "
        f"detector={lr:g}",
        flush=True,
    )
    base_lrs: list[float] = []
    for group in groups:
        base_lrs.append(float(group["lr"]))
    return groups, base_lrs


def uses_earth_adapter(backbone: str, earth_adapter: str) -> bool:
    key = normalize_backbone_name(backbone)
    return key in ("earth-mamba", "earth-mamba-b") and earth_adapter != "none"


def normalize_backbone_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")
