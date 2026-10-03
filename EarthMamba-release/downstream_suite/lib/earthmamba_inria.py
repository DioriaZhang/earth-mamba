from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

SMALL_DEPTHS = [2, 2, 27, 2]
SMALL_DIMS = [96, 192, 384, 768]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_earthmamba_path() -> None:
    root = _project_root()
    candidates = [
        root / "earth-mamba",
        root.parent / "earth-mamba",
        root.parent.parent / "earth-mamba",  # 仓库根 earth-mamba（两文件夹布局）
        Path("/hy-tmp/earth-mamba"),
    ]
    for cand in candidates:
        if (cand / "earth_mamba").is_dir():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
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
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        if semantic.shape[-2:] != detail.shape[-2:]:
            semantic = F.interpolate(semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        d = self.detail(detail)
        s = self.semantic(semantic)
        gate = self.gate(torch.cat([d, s], dim=1))
        return self.out(d + gate * s)


class EarthMambaFeatureAdapter(nn.Module):
    """Convert native EarthMamba stage features into dense P2-P5 features."""

    out_dims = list(SMALL_DIMS)
    out_strides = [4, 8, 16, 32]

    def __init__(self, mode: str = "gated_pyramid"):
        super().__init__()
        self.mode = mode
        if mode == "none":
            self.out_strides = [16, 32, 64, 128]
            return
        if mode not in ("detail_pyramid", "gated_pyramid", "semantic_pyramid"):
            raise ValueError(f"unknown dense_adapter={mode!r}")

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
        for p in parts:
            if p.shape[-2:] != tuple(size):
                p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
            out = p if out is None else out + p
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
        p3_sem = self._sum_to([self.sem_p3[0](feats[0]), self.sem_p3[1](feats[1])], p3_detail.shape[-2:])
        p4_sem = self._sum_to([self.sem_p4[0](feats[0]), self.sem_p4[1](feats[2])], p4_detail.shape[-2:])
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
            self.fuse_p2(p2_detail + F.interpolate(p2_sem, p2_detail.shape[-2:], mode="bilinear", align_corners=False)),
            self.fuse_p3(p3_detail + p3_sem),
            self.fuse_p4(p4_detail + p4_sem),
            self.fuse_p5(p5_detail + p5_sem),
        ]


class EarthMambaDenseEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        ssm_version: str = "mamba3",
        dense_adapter: str = "gated_pyramid",
        debug_shapes: bool = False,
        use_armg: bool = True,
        use_graph: bool = True,
    ):
        super().__init__()
        _ensure_earthmamba_path()
        from earth_mamba.models.earth_mamba import EarthMamba
        from earth_mamba.models.earth_mamba_block import LayerNorm as EM_LayerNorm

        self.core = EarthMamba(
            patch_size=16,
            in_chans=3,
            num_classes=1,
            depths=SMALL_DEPTHS,
            dims=SMALL_DIMS,
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
        self.use_armg = use_armg
        self.use_graph = use_graph
        self.channel_first = self.core.channel_first
        self.outnorms = nn.ModuleList([
            EM_LayerNorm(SMALL_DIMS[i], channel_first=self.channel_first)
            for i in range(4)
        ])
        self.dense_adapter = EarthMambaFeatureAdapter(dense_adapter)
        self.out_dims = list(self.dense_adapter.out_dims)
        self.out_strides = list(self.dense_adapter.out_strides)
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
            print(f"  feature_shapes={[tuple(o.shape) for o in outs]} strides={self.out_strides}")
            self._shape_logged = True
        return outs


def load_earthmamba_backbone(encoder: EarthMambaDenseEncoder, ckpt: Optional[str]) -> None:
    if not ckpt:
        print("  [EarthMamba] random init")
        return
    print(f"  [EarthMamba] loading backbone: {ckpt}")
    try:
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(ckpt, map_location="cpu")
    sd = raw.get("model", raw.get("state_dict", raw))
    model_sd = encoder.core.state_dict()
    cleaned = {}
    skipped_shape = []
    ignored = 0
    for raw_key, v in sd.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "encoder_without_ddp.", "encoder.", "backbone.", "core."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        if key.startswith(("decoder.", "classifier.", "mask_token")) or key == "mask_value":
            ignored += 1
            continue
        if key == "pos_embed" and key in model_sd and tuple(v.shape) != tuple(model_sd[key].shape):
            if v.dim() == 4 and model_sd[key].dim() == 4 and v.shape[1] == model_sd[key].shape[1]:
                v = F.interpolate(v.float(), size=model_sd[key].shape[-2:], mode="bicubic", align_corners=False).to(v.dtype)
        if key in model_sd:
            if tuple(model_sd[key].shape) != tuple(v.shape):
                skipped_shape.append(f"{key}: ckpt{tuple(v.shape)} model{tuple(model_sd[key].shape)}")
                continue
            cleaned[key] = v
    missing, unexpected = encoder.core.load_state_dict(cleaned, strict=False)
    print(
        f"    loaded_keys={len(cleaned)} missing={len(missing)} unexpected={len(unexpected)} "
        f"skipped_shape={len(skipped_shape)} ignored={ignored}"
    )
    if skipped_shape:
        for item in skipped_shape[:8]:
            print(f"      skip shape: {item}")
    if missing:
        print(f"      missing sample: {list(missing)[:8]}")
    if unexpected:
        print(f"      unexpected sample: {list(unexpected)[:8]}")


def build_earthmamba_encoder(
    ckpt: Optional[str],
    img_size: int,
    *,
    ssm_version: str = "mamba3",
    dense_adapter: str = "gated_pyramid",
    debug_shapes: bool = False,
    use_armg: bool = True,
    use_graph: bool = True,
) -> EarthMambaDenseEncoder:
    enc = EarthMambaDenseEncoder(
        img_size=img_size,
        ssm_version=ssm_version,
        dense_adapter=dense_adapter,
        debug_shapes=debug_shapes,
        use_armg=use_armg,
        use_graph=use_graph,
    )
    load_earthmamba_backbone(enc, ckpt)
    return enc


def _add_decay_groups(groups: list, params: Iterable[tuple[str, nn.Parameter]], lr: float, weight_decay: float) -> None:
    decay = []
    no_decay = []
    for name, p in params:
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower() or "gn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})


def build_earthmamba_param_groups(
    model: nn.Module,
    *,
    lr: float,
    encoder_lr: float,
    adapter_lr: Optional[float],
    weight_decay: float,
) -> list:
    adapter_lr = lr if adapter_lr is None else adapter_lr
    core, adapter, other = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".core." in f".{name}." or name.startswith("core."):
            core.append((name, p))
        elif "dense_adapter" in name or "outnorms" in name:
            adapter.append((name, p))
        else:
            other.append((name, p))
    groups: list = []
    _add_decay_groups(groups, core, encoder_lr, weight_decay)
    _add_decay_groups(groups, adapter, adapter_lr, weight_decay)
    _add_decay_groups(groups, other, lr, weight_decay)
    print(f"  lr groups: encoder={encoder_lr:g} adapter={adapter_lr:g} task={lr:g}")
    return groups
