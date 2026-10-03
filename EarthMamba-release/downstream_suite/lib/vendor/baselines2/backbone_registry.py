"""baselines2 backbone 注册中心。

支持 2 个模型（与 baselines/shared/backbone_registry.py 独立）：
  rvsa           — RVSA ViT-B，MAE on MillionAID (~86M)
  satlas_aerial  — SatlasPretrain Swin-v2-B，Aerial_SwinB_SI (~88M)

统一接口：
  encoder = build_encoder(backbone, img_size, ckpt)
  feats: List[Tensor(B,C,H,W)] = encoder(x)
  encoder.out_dims: List[int]  # 各骨干原生通道，动态传入 UPerNet
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

BASELINES2_ROOT = Path(__file__).resolve().parent
SCALEMAE_ROOT = BASELINES2_ROOT.parent / "v3_scalemae"

from shared.weights_manifest import CKPT_NAME_WARNINGS, PREFERRED_CKPT_NAMES


class BaseEncoder(nn.Module):
    out_dims: List[int]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError


class RVSAEncoder(BaseEncoder):
    """RVSA ViT-B encoder。

    使用 timm 构建标准 ViT-B/16，加载 MillionAID MAE 预训练权重。
    在层 [2, 5, 8, 11] 抽取中间特征。
    out_dims = [768, 768, 768, 768]

    注意：须使用 ViT-B 权重，不能用 ViTAE-B 权重（架构不同）。
    """

    EMBED_DIM = 768
    DEPTH = 12
    NUM_HEADS = 12
    PATCH_SIZE = 16
    EXTRACT_LAYERS = [2, 5, 8, 11]

    def __init__(self, img_size: int = 224, ckpt: Optional[str] = None):
        super().__init__()
        self.embed_dim = self.EMBED_DIM
        self.patch_size = self.PATCH_SIZE
        self.out_dims = [self.EMBED_DIM] * 4

        try:
            import timm  # type: ignore
        except ImportError:
            raise ImportError("RVSA encoder 需要 timm: pip install timm")

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            img_size=img_size,
            embed_dim=self.EMBED_DIM,
            depth=self.DEPTH,
            num_heads=self.NUM_HEADS,
            patch_size=self.PATCH_SIZE,
            num_classes=0,
            global_pool="",
        )

        if ckpt:
            _warn_ckpt_name("RVSA", ckpt)
            self._load_ckpt(ckpt)

        n = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"  [RVSA] ViT-B/16 img_size={img_size} params={n:.1f}M ckpt={ckpt or 'random'}")

    def _load_ckpt(self, path: str) -> None:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            for key in ("model", "state_dict", "encoder", "backbone"):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break

        state: Dict[str, torch.Tensor] = {
            k: v for k, v in obj.items()
            if not any(k.startswith(p) for p in ("decoder", "mask_token", "head"))
        }

        # ViTAE-B 权重含 PCM 等 ViT-B 没有的 key
        vitae_hints = sum(1 for k in state if "pcm" in k.lower() or "reduction" in k.lower())
        if vitae_hints > 3:
            warnings.warn(
                f"[RVSA] 权重 '{Path(path).name}' 疑似 ViTAE-B，"
                "当前 encoder 是标准 ViT-B，请换 MillionAID MAE ViT-B 权重。"
            )

        if "pos_embed" in state:
            state = _resize_pos_embed(state, self.vit)

        missing, unexpected = self.vit.load_state_dict(state, strict=False)
        print(f"  [RVSA] load: {len(state)} keys; missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 20:
            warnings.warn(f"[RVSA] missing keys 过多 ({len(missing)})，权重可能不匹配。")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, _, H, W = x.shape
        h_tok = H // self.patch_size
        w_tok = W // self.patch_size

        extracted: Dict[int, torch.Tensor] = {}
        hooks = []

        for idx in self.EXTRACT_LAYERS:
            def _hook(module, inp, out, _i=idx):
                t = out[0] if isinstance(out, tuple) else out
                if t.ndim == 3:
                    n_sp = h_tok * w_tok
                    if t.shape[1] == n_sp + 1:
                        t = t[:, 1:, :]
                    extracted[_i] = (
                        t.reshape(B, h_tok, w_tok, self.embed_dim)
                        .permute(0, 3, 1, 2).contiguous()
                    )

            hooks.append(self.vit.blocks[idx].register_forward_hook(_hook))

        _ = self.vit(x)

        for h in hooks:
            h.remove()

        return [extracted[i] for i in self.EXTRACT_LAYERS]


class SatlasAerialEncoder(BaseEncoder):
    """SatlasPretrain Swin-v2-B encoder。

    架构：torchvision swin_v2_b，4 尺度输出 [128,256,512,1024]。
    支持 aerial_swinb_si.pth 与 sentinel2_swinb_si_rgb.pth（结构相同，预训练数据不同）。
    """

    OUT_DIMS = [128, 256, 512, 1024]

    def __init__(self, ckpt: Optional[str] = None, num_channels: int = 3):
        super().__init__()
        self.out_dims = list(self.OUT_DIMS)

        import torchvision  # type: ignore
        self.backbone = torchvision.models.swin_v2_b(weights=None)
        out_ch = self.backbone.features[0][0].out_channels
        self.backbone.features[0][0] = nn.Conv2d(
            num_channels, out_ch, kernel_size=(4, 4), stride=(4, 4)
        )

        if ckpt:
            _warn_ckpt_name("SatlasPretrain_Aerial", ckpt)
            self._load_ckpt(ckpt)

        n = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"  [SatlasAerial] Swin-v2-B params={n:.1f}M ckpt={ckpt or 'random'}")

    def _load_ckpt(self, path: str) -> None:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            if "model" in obj:
                obj = obj["model"]

        state: Dict[str, torch.Tensor] = {}
        for k, v in obj.items():
            if "backbone" not in k:
                continue
            while k.count("backbone.") > 1:
                k = k.replace("backbone.", "", 1)
            k = k.removeprefix("backbone.")
            state[k] = v

        if not state:
            state = {k: v for k, v in obj.items() if not k.startswith("head")}

        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        print(f"  [SatlasAerial] load: {len(state)} keys; missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 10:
            warnings.warn(f"[SatlasAerial] missing keys 过多 ({len(missing)})，权重可能不匹配。")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


def _resize_pos_embed(state: dict, model: nn.Module) -> dict:
    """将 MAE checkpoint 中的 pos_embed 双三次插值到当前模型分辨率。"""
    pe_ckpt = state["pos_embed"]
    pe_model = model.pos_embed.data
    if pe_ckpt.shape == pe_model.shape:
        return state

    has_cls = getattr(model, "cls_token", None) is not None

    if has_cls:
        cls_token = pe_ckpt[:, :1, :]
        pe_ckpt_sp = pe_ckpt[:, 1:, :]
        n_model_sp = pe_model.shape[1] - 1
    else:
        cls_token = None
        pe_ckpt_sp = pe_ckpt
        n_model_sp = pe_model.shape[1]

    n_ckpt_sp = pe_ckpt_sp.shape[1]
    h_ckpt = int(n_ckpt_sp ** 0.5)
    h_model = int(n_model_sp ** 0.5)

    if h_ckpt == h_model:
        return state

    pe_s = pe_ckpt_sp.reshape(1, h_ckpt, h_ckpt, -1).permute(0, 3, 1, 2).float()
    pe_s = F.interpolate(pe_s, size=(h_model, h_model), mode="bicubic", align_corners=False)
    pe_s = pe_s.permute(0, 2, 3, 1).reshape(1, h_model * h_model, -1).to(pe_ckpt.dtype)

    if cls_token is not None:
        state["pos_embed"] = torch.cat([cls_token, pe_s], dim=1)
    else:
        state["pos_embed"] = pe_s

    print(f"  [pos_embed] 插值 {h_ckpt}²→{h_model}²")
    return state


def _warn_ckpt_name(model_folder: str, ckpt_path: str) -> None:
    name = Path(ckpt_path).name.lower()
    for hint in CKPT_NAME_WARNINGS.get(model_folder, []):
        if hint in name:
            preferred = PREFERRED_CKPT_NAMES.get(model_folder, [])
            warnings.warn(
                f"[{model_folder}] 权重文件名含 '{hint}'，可能下错了。"
                f" 期望: {preferred or '见 README'}"
            )
            return


BACKBONE_CHOICES = ["rvsa", "satlas_aerial", "scalemae"]


def build_encoder(
    backbone: str,
    img_size: int,
    ckpt: Optional[str] = None,
    **kwargs,
) -> BaseEncoder:
    name = backbone.lower().replace("-", "_")

    if name in ("scalemae", "scale_mae"):
        if not (SCALEMAE_ROOT / "scalemae_backbone.py").is_file():
            raise FileNotFoundError(f"Cannot locate {SCALEMAE_ROOT / 'scalemae_backbone.py'}")
        scale_root = str(SCALEMAE_ROOT)
        if scale_root not in sys.path:
            sys.path.insert(0, scale_root)
        importlib.invalidate_caches()
        from scalemae_backbone import build_encoder as build_scalemae

        return build_scalemae("scalemae", img_size, ckpt)

    if name == "rvsa":
        return RVSAEncoder(img_size=img_size, ckpt=ckpt)
    if name in ("satlas_aerial", "satlas"):
        return SatlasAerialEncoder(ckpt=ckpt)
    raise ValueError(f"未知 backbone: {backbone}，可选: {BACKBONE_CHOICES}")


def resolve_ckpt(
    backbone: str,
    ckpt: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
) -> Optional[str]:
    """优先 --ckpt，其次 --ckpt_dir，最后各模型默认 weights/ 目录。"""
    from shared.paths import backbone_weights_dir, find_ckpt

    if ckpt:
        return ckpt
    if ckpt_dir:
        found = find_ckpt(Path(ckpt_dir))
        if found:
            return found

    name_map = {
        "rvsa": "RVSA",
        "satlas_aerial": "SatlasPretrain_Aerial",
    }
    norm = backbone.lower().replace("-", "_")
    if norm in ("scalemae", "scale_mae"):
        default_ckpt = SCALEMAE_ROOT / "scalemae-vitlarge-800.pth"
        if default_ckpt.is_file():
            return str(default_ckpt)
        return None
    folder = name_map.get(norm)
    if not folder:
        return None

    weights_dir = backbone_weights_dir(folder)
    preferred = PREFERRED_CKPT_NAMES.get(folder, [])
    for name in preferred:
        p = weights_dir / name
        if p.is_file():
            return str(p)

    return find_ckpt(weights_dir)
