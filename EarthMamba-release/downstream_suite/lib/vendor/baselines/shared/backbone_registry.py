"""
统一 Backbone 注册表：7 个遥感预训练模型的标准化 Encoder 适配器。

所有 Encoder 接口::

  encoder = build_encoder(backbone_name, img_size, ckpt, **kwargs)
  feats   = encoder(x)      # List[Tensor(B, C_i, H_i, W_i)]，长度 = 4
  encoder.out_dims          # List[int]，4 个通道数

backbone_name（不区分大小写）::
  earthmamba | skysense | satmae | dofa | clay | roma | rsmamba

多尺度模型（Swin/EarthMamba）自然输出 4 个不同分辨率特征图。
单尺度 ViT 类模型在 {depth//4, depth//2, depth*3//4, depth} 层抽取特征并
reshape 成 (B, embed_dim, H/patch, W/patch)，out_dims=[embed_dim]*4。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 目录定位（/hy-tmp/baselines + /hy-tmp/downstream_code）
_THIS_DIR = Path(__file__).resolve().parent        # baselines/shared/
_BASELINES_DIR = _THIS_DIR.parent                  # baselines/
_HYTMP_ROOT = _BASELINES_DIR.parent               # /hy-tmp/
_DOWNSTREAM_DIR = _HYTMP_ROOT / "downstream_code"

# earth-mamba：downstream_code/earth-mamba 或 /hy-tmp/earth-mamba
for _em_candidate in (_DOWNSTREAM_DIR / "earth-mamba", _HYTMP_ROOT / "earth-mamba"):
    if _em_candidate.is_dir():
        _p = str(_em_candidate)
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break


# ──────────────────────────────────────────────────────────────────────────────
# 公共工具
# ──────────────────────────────────────────────────────────────────────────────

def _safe_load(ckpt_path: str, map_location: str = "cpu") -> dict:
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except Exception:
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)


def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _strip_checkpoint_prefixes(sd: dict) -> dict:
    """剥去 checkpoint 中常见的 module/encoder/backbone 前缀。"""
    for prefix in ("module.", "encoder_without_ddp.", "encoder.", "backbone.", "model."):
        if any(k.startswith(prefix) for k in sd):
            return _strip_prefix(sd, prefix)
    return sd


def _filter_state_dict_for_model(
    model: nn.Module,
    sd: dict,
    *,
    skip_prefixes: tuple[str, ...] = (),
) -> tuple[dict, list[str]]:
    """只保留 model 中 shape 一致的 key，跳过 shape mismatch（strict=False 也会报错）。"""
    model_sd = model.state_dict()
    filtered: dict = {}
    skipped: list[str] = []
    for k, v in sd.items():
        if any(k.startswith(p) for p in skip_prefixes):
            continue
        if k in model_sd and model_sd[k].shape != v.shape:
            skipped.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(model_sd[k].shape)}")
            continue
        filtered[k] = v
    return filtered, skipped


def _load_state_dict_relaxed(model: nn.Module, sd: dict, label: str = "") -> tuple[list, list]:
    skip = (
        "decoder.", "head.", "mask_token", "decoder_pos_embed",
        "decoder_embed", "decoder_pred", "fc_norm", "norm.",
    )
    filtered, skipped = _filter_state_dict_for_model(model, sd, skip_prefixes=skip)
    if skipped:
        tag = f"[{label}] " if label else ""
        print(f"    {tag}跳过 {len(skipped)} 个 shape 不匹配的 key")
        for s in skipped[:8]:
            print(f"      {s}")
        if len(skipped) > 8:
            print(f"      ... 等 {len(skipped)} 个")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"    missing={len(missing)}  unexpected={len(unexpected)}")
    return list(missing), list(unexpected)


def _count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"total={total/1e6:.1f}M  trainable={trainable/1e6:.1f}M"


def _extract_state_dict(raw: dict) -> dict:
    """从常见 checkpoint 包装格式中提取 state_dict。"""
    if not isinstance(raw, dict):
        return {}
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    if "model" in raw and isinstance(raw["model"], dict):
        return raw["model"]
    return raw if isinstance(raw, dict) else {}


def _is_decoder_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("decoder", "mask_token", "encoder_pred", "lm_head"))


def _encoder_pos_embed_key(sd: dict) -> Optional[str]:
    """优先选 encoder 的 pos_embed，避免 MAE ckpt 里 decoder_pos_embed 被误读。"""
    exact = [k for k in sd if k in ("pos_embed", "pos_embed.weight")]
    if exact:
        return exact[0]
    candidates = [
        k for k in sd
        if not _is_decoder_key(k)
        and (k.endswith("pos_embed") or k.endswith("pos_embed.weight"))
        and getattr(sd[k], "ndim", 0) == 3
    ]
    return candidates[0] if candidates else None


def _encoder_patch_key(sd: dict) -> Optional[str]:
    candidates = [
        k for k in sd
        if not _is_decoder_key(k)
        and "patch_embed" in k
        and k.endswith(".weight")
        and getattr(sd[k], "ndim", 0) == 4
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda key: (".proj." not in key, key.count(".")))
    return candidates[0]


def _peek_vit_config(ckpt_path: Optional[str]) -> Dict[str, int]:
    """从 checkpoint 推断 ViT 结构（embed_dim / depth / patch_size / 预训练分辨率）。"""
    if not ckpt_path or not Path(ckpt_path).is_file():
        return {}
    sd = _extract_state_dict(_safe_load(ckpt_path))
    if not sd:
        return {}
    sd = _strip_checkpoint_prefixes(sd)

    cfg: Dict[str, int] = {}

    patch_key = _encoder_patch_key(sd)
    if patch_key is not None:
        pw = int(sd[patch_key].shape[-1])
        cfg["patch_size"] = pw
        cfg["embed_dim"] = int(sd[patch_key].shape[0])  # 比 pos_embed 更可靠

    pe_key = _encoder_pos_embed_key(sd)
    if pe_key is not None:
        pe = sd[pe_key]
        if hasattr(pe, "shape") and len(pe.shape) == 3:
            if "embed_dim" not in cfg:
                cfg["embed_dim"] = int(pe.shape[-1])
            n_tok = int(pe.shape[1])
            n_patch = n_tok
            has_cls = 0
            side = int(n_patch ** 0.5)
            if side * side != n_patch:
                n_patch = n_tok - 1
                side = int(n_patch ** 0.5)
                if side * side == n_patch:
                    has_cls = 1
            if side * side == n_patch:
                cfg["has_cls_token"] = has_cls
                cfg["pretrain_patches"] = n_patch
                pw = cfg.get("patch_size")
                if pw:
                    cfg["pretrain_img_size"] = side * pw

    block_ids: set[int] = set()
    for k in sd:
        if _is_decoder_key(k):
            continue
        parts = k.split(".")
        if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
            block_ids.add(int(parts[1]))
    if block_ids:
        cfg["depth"] = max(block_ids) + 1

    return cfg


def _clean_mae_encoder_state_dict(sd: dict, embed_dim: int) -> dict:
    """去掉 MAE decoder/head 权重，并确保 pos_embed 来自 encoder。"""
    cleaned: dict = {}
    for key, value in sd.items():
        if _is_decoder_key(key):
            continue
        if key.startswith(("head.", "fc_norm.")):
            continue
        cleaned[key] = value

    pos_candidates = {
        key: value
        for key, value in cleaned.items()
        if "pos_embed" in key and getattr(value, "ndim", 0) == 3
    }
    if pos_candidates:
        best_key = next(
            (key for key, value in pos_candidates.items() if int(value.shape[-1]) == embed_dim),
            None,
        )
        if best_key is None:
            best_key, best_value = max(pos_candidates.items(), key=lambda item: item[1].shape[-1])
        else:
            best_value = pos_candidates[best_key]
        for key in list(pos_candidates):
            if key != best_key:
                cleaned.pop(key, None)
        cleaned["pos_embed"] = best_value
    return cleaned


def _vit_model_has_cls(model_pe: torch.Tensor, h_tok: int, w_tok: int) -> bool:
    """判断 timm ViT 的 pos_embed 是否含 CLS token。"""
    n_patch = h_tok * w_tok
    n = model_pe.shape[1]
    if n == n_patch + 1:
        return True
    if n == n_patch:
        return False
    # 非标准：按是否完全平方猜测
    side = int(round(n ** 0.5))
    return side * side != n


def _split_pos_embed_tokens(
    pos_embed: torch.Tensor,
) -> tuple[Optional[torch.Tensor], torch.Tensor, int, int]:
    """拆分 pos_embed 为 (cls_or_none, patch_tokens, h_old, w_old)。"""
    n = pos_embed.shape[1]
    side = int(round(n ** 0.5))
    if side * side == n:
        return None, pos_embed, side, side
    side = int(round((n - 1) ** 0.5))
    if side * side == n - 1:
        return pos_embed[:, :1, :], pos_embed[:, 1:, :], side, side
    raise ValueError(f"pos_embed token 数 {n} 无法解析为 grid+可选CLS")


def _resize_pos_embed_for_vit(
    stored_pe: torch.Tensor,
    *,
    h_tok: int,
    w_tok: int,
    model_has_cls: bool,
) -> torch.Tensor:
    """将 ckpt 的 pos_embed 双三次插值到目标 patch 网格 (h_tok × w_tok)。"""
    cls_tok, patch_pe, h_old, w_old = _split_pos_embed_tokens(stored_pe)
    C = patch_pe.shape[-1]
    pe = patch_pe.reshape(1, h_old, w_old, C).permute(0, 3, 1, 2).float()
    pe = F.interpolate(pe, size=(h_tok, w_tok), mode="bicubic", align_corners=False)
    pe = pe.flatten(2).permute(0, 2, 1).to(dtype=stored_pe.dtype, device=stored_pe.device)
    if model_has_cls:
        if cls_tok is not None:
            pe = torch.cat([cls_tok.to(pe.device, pe.dtype), pe], dim=1)
        else:
            pe = torch.cat([torch.zeros(1, 1, C, dtype=pe.dtype, device=pe.device), pe], dim=1)
    return pe


def _prepare_vit_pos_embed_in_sd(
    sd: dict,
    vit: nn.Module,
    h_tok: int,
    w_tok: int,
    label: str = "",
) -> dict:
    """加载前调整 state_dict 中的 pos_embed（分辨率 / CLS 不一致时自动插值）。"""
    if "pos_embed" not in sd or not hasattr(vit, "pos_embed"):
        return sd
    stored_pe: torch.Tensor = sd["pos_embed"]
    model_pe: torch.Tensor = vit.pos_embed
    if stored_pe.shape == model_pe.shape:
        return sd
    sd = dict(sd)
    tag = f"[{label}] " if label else ""
    if stored_pe.shape[-1] != model_pe.shape[-1]:
        print(f"    {tag}pos_embed embed_dim 不匹配 {stored_pe.shape} vs {model_pe.shape}，跳过 pos_embed")
        del sd["pos_embed"]
        return sd
    model_has_cls = _vit_model_has_cls(model_pe, h_tok, w_tok)
    print(
        f"    {tag}pos_embed 插值: ckpt{tuple(stored_pe.shape)} → "
        f"grid {h_tok}×{w_tok} (+cls={model_has_cls})"
    )
    try:
        sd["pos_embed"] = _resize_pos_embed_for_vit(
            stored_pe, h_tok=h_tok, w_tok=w_tok, model_has_cls=model_has_cls,
        )
    except ValueError as e:
        print(f"    {tag}pos_embed 调整失败: {e}，跳过 pos_embed")
        del sd["pos_embed"]
    return sd


# ──────────────────────────────────────────────────────────────────────────────
# BaseEncoder 基类
# ──────────────────────────────────────────────────────────────────────────────

class BaseEncoder(nn.Module):
    """所有 Encoder 适配器的基类。
    子类必须设置 self.out_dims 并实现 forward()。
    """
    out_dims: List[int]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """输入 (B, 3, H, W)，输出 4 个特征图列表。"""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# 1. EarthMamba 适配器
# ──────────────────────────────────────────────────────────────────────────────

_EM_DEPTHS = [2, 2, 27, 2]
_EM_DIMS   = [96, 192, 384, 768]


class EarthMambaAdapter(BaseEncoder):
    """EarthMamba（VMamba-Small 变体）多尺度 Encoder。
    out_dims = [96, 192, 384, 768]
    """

    def __init__(
        self,
        img_size: int,
        ssm_version: str = "mamba3",
        use_armg: bool = True,
        use_graph: bool = True,
    ):
        super().__init__()
        from earth_mamba.models.earth_mamba import EarthMamba
        from earth_mamba.models.earth_mamba_block import LayerNorm as EM_LayerNorm

        self.backbone = EarthMamba(
            patch_size=16, in_chans=3, num_classes=1,
            depths=_EM_DEPTHS, dims=_EM_DIMS,
            ssm_d_state=64 if ssm_version == "mamba3" else 16,
            ssm_ratio=2.0, ssm_version=ssm_version, ssm_headdim=64,
            mlp_ratio=4.0, drop_path_rate=0.0,
            norm_layer="ln", posembed=True, imgsize=img_size,
            downsample_version="v3",
            use_armg=use_armg,
            use_graph=use_graph,
        )
        self.use_armg = use_armg
        self.use_graph = use_graph
        self.channel_first = self.backbone.channel_first
        for i in range(4):
            self.add_module(
                f"outnorm{i}",
                EM_LayerNorm(_EM_DIMS[i], channel_first=self.channel_first),
            )
        self.out_dims = list(_EM_DIMS)

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
            o = getattr(self, f"outnorm{i}")(x)
            if not self.channel_first:
                o = o.permute(0, 3, 1, 2)
            outs.append(o.contiguous())
            x = layer.downsample(x)
        return outs

    def load_pretrained(self, ckpt_path: str) -> None:
        print(f"  [EarthMamba] 加载预训练权重: {ckpt_path}")
        raw = _safe_load(ckpt_path)
        sd = raw.get("model", raw.get("state_dict", raw))
        cleaned: dict = {}
        for k, v in sd.items():
            if k.startswith("encoder."):
                k = k[len("encoder."):]
            if k.startswith("decoder.") or k == "mask_value":
                continue
            if k == "pos_embed" and v.shape != self.backbone.pos_embed.shape:
                v = F.interpolate(
                    v, size=self.backbone.pos_embed.shape[2:],
                    mode="bicubic", align_corners=False,
                )
            cleaned[k] = v
        m, u = self.backbone.load_state_dict(cleaned, strict=False)
        print(f"    missing={len(m)}  unexpected={len(u)}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Swin-V2 适配器（用于 SkySense）
# ──────────────────────────────────────────────────────────────────────────────

# SkySense 官方 SwinTransformerV2 配置（见 models/swin_transformer_v2.py arch_zoo + README Usage）
# HR backbone: SwinTransformerV2() 即 arch='huge'，无需传 embed_dim/depths
_SKYSENSE_ARCH: Dict[str, str] = {
    "huge": "huge", "large": "large", "base": "base", "small": "small",
}
# 各 stage 输出通道（embed_dims × 2^stage）
_SKYSENSE_OUT_DIMS: Dict[str, List[int]] = {
    "huge":  [352, 704, 1408, 2816],   # arch_zoo huge embed_dims=352
    "large": [192, 384, 768, 1536],
    "base":  [128, 256, 512, 1024],
    "small": [96,  192, 384, 768],
}


class SwinV2Adapter(BaseEncoder):
    """Swin Transformer V2 多尺度 Encoder（用于 SkySense HR RGB backbone）。

    官方用法（README）::
        from swin_transformer_v2 import SwinTransformerV2
        model = SwinTransformerV2()   # arch='huge', window_size=8
        ckpt = {k.replace('backbone.', ''): v for k, v in ckpt.items()
                if k.startswith('backbone.')}

    依赖（pip 安装，无需克隆整个 SkySense 仓库）::
        openmim → mmcv-full==1.7.1, mmcls==0.25.0

    只需 baselines/SkySense/swin_transformer_v2.py + weights/ 即可。
    mmcv/mmcls 不可用时才 fallback 到 timm。
    """

    @staticmethod
    def _check_skysense_deps() -> Optional[str]:
        """逐步检查 SkySense 官方依赖，返回 None=OK，否则返回错误说明。"""
        try:
            import mmcv
            ver = getattr(mmcv, "__version__", "0")
            if ver.split(".")[0].isdigit() and int(ver.split(".")[0]) >= 2:
                py = f"{sys.version_info.major}.{sys.version_info.minor}"
                return (
                    f"检测到 mmcv {ver}（2.x），SkySense 需要 mmcv-full 1.x + mmcls 0.25.0。\n"
                    f"  当前 Python {py}，与主环境 mmcv2 不兼容。\n"
                    f"  请单独建 SkySense 环境：bash baselines/SkySense/setup_skysense_env.sh\n"
                    f"  然后 conda activate skysense 再跑 --backbone skysense。"
                )
        except ImportError:
            pass

        checks = [
            ("mmcv",       "from mmcv.cnn import build_norm_layer"),
            ("mmcls",      "from mmcls.models.utils import WindowMSAV2"),
            ("mmcls.base", "from mmcls.models.backbones.base_backbone import BaseBackbone"),
        ]
        for name, stmt in checks:
            try:
                exec(stmt, {})  # noqa: S102
            except Exception as e:
                return f"{name} 不可用: {e}"
        return None

    def __init__(self, img_size: int, model_size: str = "huge"):
        super().__init__()
        skysense_dir = _BASELINES_DIR / "SkySense"
        skysense_str = str(skysense_dir)
        swin_file = skysense_dir / "swin_transformer_v2.py"
        if skysense_str not in sys.path:
            sys.path.insert(0, skysense_str)

        self._img_size = img_size

        if not swin_file.is_file():
            raise FileNotFoundError(
                f"未找到 {swin_file}\n"
                f"请从 SkySense 官方仓库复制 models/swin_transformer_v2.py 到该路径。"
            )

        dep_err = self._check_skysense_deps()
        if dep_err:
            raise RuntimeError(f"SkySense 官方模型依赖检查失败：\n{dep_err}")

        try:
            from swin_transformer_v2 import SwinTransformerV2  # type: ignore
        except Exception as e:
            raise ImportError(
                f"swin_transformer_v2.py 存在但 import 失败: {e}\n"
                f"文件路径: {swin_file}"
            ) from e

        size_key = model_size.lower()
        arch = _SKYSENSE_ARCH.get(size_key, "huge")
        try:
            self.backbone = SwinTransformerV2(
                arch=arch,
                img_size=img_size,
                out_indices=(0, 1, 2, 3),
                window_size=8,
            )
        except Exception as e:
            raise RuntimeError(
                f"SwinTransformerV2 实例化失败 (arch={arch}, img_size={img_size}): {e}"
            ) from e

        self.out_dims = list(_SKYSENSE_OUT_DIMS.get(size_key, _SKYSENSE_OUT_DIMS["huge"]))
        print(f"  [SkySense] 官方 SwinTransformerV2  arch={arch}  out_dims={self.out_dims}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = self.backbone(x)
        return list(outs) if isinstance(outs, tuple) else [outs]

    def load_pretrained(self, ckpt_path: str) -> None:
        print(f"  [SkySense/SwinV2] 加载预训练权重: {ckpt_path}")
        raw = _safe_load(ckpt_path)
        # 官方格式：顶层 dict，key 为 backbone.xxx
        if isinstance(raw, dict) and any(k.startswith("backbone.") for k in raw):
            sd = {k.replace("backbone.", "", 1): v for k, v in raw.items()
                  if k.startswith("backbone.")}
        else:
            sd = _extract_state_dict(raw)
            if any(k.startswith("backbone.") for k in sd):
                sd = _strip_prefix(sd, "backbone.")
            sd = _strip_checkpoint_prefixes(sd)

        # 官方：strict=False；模型内部 hook 会自动处理 relative_position 等 key
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        print(f"    missing={len(missing)}  unexpected={len(unexpected)}")
        if unexpected[:5]:
            print(f"    unexpected 示例: {unexpected[:5]}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. 通用 ViT 适配器（SatMAE / DOFA / Clay 等）
# ──────────────────────────────────────────────────────────────────────────────

class ViTAdapter(BaseEncoder):
    """单尺度 ViT 适配器：在均匀分割的 4 个层索引处抽取特征，
    reshape 成 (B, embed_dim, H/patch, W/patch)，out_dims=[embed_dim]*4。

    底层模型优先从 timm 导入，若失败尝试 baselines/{model_type}/ 目录下的代码。
    """

    def __init__(
        self,
        model_type: str,         # "satmae" | "dofa" | "clay" | "satmamba" | ...
        img_size:   int   = 224,
        embed_dim:  int   = 768,
        depth:      int   = 12,
        num_heads:  int   = 12,
        patch_size: int   = 16,
        has_cls_token: bool = True,
    ):
        super().__init__()
        self.model_type    = model_type.lower()
        self.embed_dim     = embed_dim
        self.depth         = depth
        self.patch_size    = patch_size
        self.has_cls_token = has_cls_token
        self.h_tok = img_size  // patch_size
        self.w_tok = img_size  // patch_size

        # 抽取层索引（4 段均分）
        self.extract_indices = [
            max(0, depth // 4 - 1),
            max(0, depth // 2 - 1),
            max(0, depth * 3 // 4 - 1),
            depth - 1,
        ]
        self.out_dims = [embed_dim] * 4

        # 构建底层 ViT
        self.vit = self._build_vit(model_type, img_size, embed_dim, depth, num_heads, patch_size)
        self._hooks: list = []

    # ── 底层 ViT 构建 ──────────────────────────────────────────────────────

    def _build_vit(self, model_type, img_size, embed_dim, depth, num_heads, patch_size):
        if model_type == "dofa":
            return self._build_dofa(img_size, embed_dim, depth, num_heads, patch_size)
        if model_type == "clay":
            return self._build_clay(img_size, embed_dim, depth, num_heads, patch_size)
        # satmae / generic：优先 timm
        return self._build_timm_vit(img_size, embed_dim, depth, num_heads, patch_size)

    def _build_timm_vit(self, img_size, embed_dim, depth, num_heads, patch_size):
        try:
            import timm  # type: ignore
            # 统一用 vit_base 作为模板，所有结构参数以关键字覆盖
            model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False, img_size=img_size,
                embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                patch_size=patch_size, num_classes=0, global_pool="",
            )
            return model
        except Exception:
            pass
        # fallback：尝试 baselines/SatMAE/models_vit.py
        satmae_dir = str(_BASELINES_DIR / "SatMAE")
        if satmae_dir not in sys.path:
            sys.path.insert(0, satmae_dir)
        try:
            if embed_dim == 768:
                from models_vit import vit_base_patch16 as vit_fn  # type: ignore
            else:
                from models_vit import vit_large_patch16 as vit_fn  # type: ignore
            model = vit_fn()
            return model
        except ImportError:
            pass
        raise ImportError(
            "ViT 模型构建失败：请安装 timm（pip install timm）"
            " 或将 SatMAE 的 models_vit.py 放到 baselines/SatMAE/"
        )

    def _build_dofa(self, img_size, embed_dim, depth, num_heads, patch_size):
        dofa_dir = str(_BASELINES_DIR / "DOFA")
        if dofa_dir not in sys.path:
            sys.path.insert(0, dofa_dir)
        try:
            from models_dwv import vit_base_patch16 as dofa_base  # type: ignore
            model = dofa_base()
            # 记录 wavelength 处理函数
            self._dofa_wavelengths = [650.0, 550.0, 450.0]  # R, G, B (nm)
            return model
        except ImportError:
            warnings.warn(
                "[DOFA] 找不到 models_dwv.py，使用 timm ViT-B 替代（无波长嵌入）。\n"
                f"请将 DOFA 仓库的 models_dwv.py 复制到 {dofa_dir}/"
            )
            return self._build_timm_vit(img_size, embed_dim, depth, num_heads, patch_size)

    def _build_clay(self, img_size, embed_dim, depth, num_heads, patch_size):
        # Clay 使用 patch_size=8，embed_dim=1024，深度=24
        return self._build_timm_vit(img_size, embed_dim, depth, num_heads, patch_size)

    # ── Forward（hook 抽取中间层特征） ─────────────────────────────────────

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, _, H, W = x.shape
        h_tok = H // self.patch_size
        w_tok = W // self.patch_size

        extracted: Dict[int, torch.Tensor] = {}

        def _make_hook(idx: int) -> Callable:
            def hook(module, inp, out):
                # timm block 输出：(B, N_tokens, C) 或 (B, C)
                t = out[0] if isinstance(out, tuple) else out
                if t.ndim == 3:  # (B, N, C)
                    n_spatial = h_tok * w_tok
                    # timm ViT 始终带 CLS；RoMA 等无 CLS 模型 ckpt 也不带 CLS
                    if t.shape[1] == n_spatial + 1:
                        t = t[:, 1:, :]
                    elif t.shape[1] != n_spatial:
                        return
                    extracted[idx] = t.permute(0, 2, 1).reshape(B, -1, h_tok, w_tok)
            return hook

        # 获取 transformer blocks list
        blocks = self._get_blocks()

        handles = []
        for i, target_idx in enumerate(self.extract_indices):
            if target_idx < len(blocks):
                h = blocks[target_idx].register_forward_hook(_make_hook(i))
                handles.append((i, h))

        # DOFA 需要额外的 wavelength 参数
        if self.model_type == "dofa" and hasattr(self, "_dofa_wavelengths"):
            try:
                import torch
                wl = torch.tensor(
                    self._dofa_wavelengths, dtype=x.dtype, device=x.device
                ).unsqueeze(0).expand(B, -1)
                self.vit(x, wl)
            except Exception:
                self.vit(x)
        else:
            self.vit(x)

        for _, h in handles:
            h.remove()

        # 补全缺失层（若 depth 很小）
        result = []
        last: Optional[torch.Tensor] = None
        for i in range(4):
            if i in extracted:
                last = extracted[i]
            result.append(last if last is not None else torch.zeros(
                B, self.embed_dim, h_tok, w_tok, device=x.device, dtype=x.dtype
            ))
        return result

    def _get_blocks(self) -> nn.ModuleList:
        """从底层模型中提取 transformer blocks list。"""
        vit = self.vit
        # timm VisionTransformer
        if hasattr(vit, "blocks"):
            return vit.blocks
        # timm 某些版本
        if hasattr(vit, "transformer") and hasattr(vit.transformer, "layers"):
            return vit.transformer.layers
        # SatMAE style
        if hasattr(vit, "blocks"):
            return vit.blocks
        # 通用递归找最长的 ModuleList
        best: Optional[nn.ModuleList] = None
        for m in vit.modules():
            if isinstance(m, nn.ModuleList) and len(m) >= self.depth:
                best = m
                break
        if best is None:
            raise AttributeError(f"无法从 {type(vit).__name__} 中找到 transformer blocks")
        return best

    # ── 权重加载 ───────────────────────────────────────────────────────────

    def load_from_state_dict(self, sd: dict, label: str = "") -> None:
        """从已清洗的 state_dict 加载权重（含 pos_embed 插值 + shape 安全过滤）。"""
        tag = label or self.model_type.upper()
        sd = _prepare_vit_pos_embed_in_sd(
            sd, self.vit, self.h_tok, self.w_tok, label=tag,
        )
        _load_state_dict_relaxed(self.vit, sd, label=tag)

    def load_pretrained(self, ckpt_path: str) -> None:
        print(f"  [{self.model_type.upper()}] 加载预训练权重: {ckpt_path}")
        raw = _safe_load(ckpt_path)

        # Clay Lightning checkpoint
        if "state_dict" in raw and "hyper_parameters" in raw:
            sd = {k.replace("model.encoder.", "").replace("encoder.", ""):
                  v for k, v in raw["state_dict"].items()
                  if "encoder" in k and "decoder" not in k}
        else:
            sd = _extract_state_dict(raw)
            sd = _strip_checkpoint_prefixes(sd)
            if self.model_type == "satmae":
                sd = _clean_mae_encoder_state_dict(sd, self.embed_dim)

        self.load_from_state_dict(sd)


# ──────────────────────────────────────────────────────────────────────────────
# 4. RoMA 适配器（NeurIPS 2025，Mamba-Base 85M）
# ──────────────────────────────────────────────────────────────────────────────

class RoMAAdapter(BaseEncoder):
    """RoMA (Rotation-aware Multi-scale Autoregressive) Mamba-Base 适配器。

    架构参数（论文 Table 1）：
      embed_dim=768, depth=12, patch_size=16, 85M 参数
    权重文件：baselines/RoMA/weights/mamba-base.pth
    External pretrained-weight link omitted in the anonymous supplement.

    底层使用 ARM 的 Mamba 骨干（models_mamba.py），与 ViT 接口兼容。
    若官方代码不可用，fallback 为 timm ViT-B/16 代理（仅用于调试）。
    """

    # Mamba-Base 参数（与 ViT-Base 对齐）
    EMBED_DIM = 768
    DEPTH = 12
    PATCH_SIZE = 16

    def __init__(self, img_size: int):
        super().__init__()
        roma_dir = str(_BASELINES_DIR / "RoMA")
        if roma_dir not in sys.path:
            sys.path.insert(0, roma_dir)

        self._inner: ViTAdapter
        loaded_official = False

        # 优先尝试加载 RoMA 官方的 Mamba 模型代码
        for module_name in ("models_mamba", "models.mamba", "mamba"):
            try:
                import importlib
                mod = importlib.import_module(module_name)
                # ARM Mamba-Base 的工厂函数名通常为 arm_base_pz16
                for fn_name in ("arm_base_pz16", "mamba_base", "MambaEncoder"):
                    if hasattr(mod, fn_name):
                        fn = getattr(mod, fn_name)
                        backbone = fn(img_size=img_size) if callable(fn) else fn
                        # 包装为 ViTAdapter，替换其内部 vit 为官方 Mamba
                        inner = ViTAdapter(
                            "roma", img_size=img_size,
                            embed_dim=self.EMBED_DIM, depth=self.DEPTH,
                            num_heads=self.EMBED_DIM // 64,
                            patch_size=self.PATCH_SIZE,
                            has_cls_token=False,  # Mamba 无 cls_token
                        )
                        inner.vit = backbone
                        self._inner = inner
                        loaded_official = True
                        break
                if loaded_official:
                    break
            except (ImportError, Exception):
                continue

        if not loaded_official:
            warnings.warn(
                "[RoMA] 官方 Mamba 模型代码不可用，使用 timm ViT-B/16 作为代理（仅调试）。\n"
                f"请将 RoMA 仓库的 models_mamba.py 复制到 {roma_dir}/\n"
                "  obtain RoMA source from its upstream project page\n"
                f"  cp /tmp/RoMA/RoMA/models_mamba.py {roma_dir}/"
            )
            # timm ViT fallback 始终含 CLS token；hook 会自动剥离
            self._inner = ViTAdapter(
                "roma", img_size=img_size,
                embed_dim=self.EMBED_DIM, depth=self.DEPTH,
                num_heads=self.EMBED_DIM // 64,
                patch_size=self.PATCH_SIZE,
                has_cls_token=True,
            )

        self.out_dims = self._inner.out_dims

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self._inner(x)

    def load_pretrained(self, ckpt_path: str) -> None:
        """加载 RoMA mamba-base.pth 权重。

        checkpoint 格式（ARM 风格）：
          {'model': state_dict, 'epoch': int, 'args': ...}
        或直接是 state_dict。
        常见前缀：'encoder.'，直接剥去；保留 patch_embed、blocks 等。
        """
        print(f"  [RoMA] 加载预训练权重: {ckpt_path}")
        raw = _safe_load(ckpt_path)
        sd = raw.get("model", raw.get("state_dict", raw))

        # 剥去常见前缀
        for prefix in ("encoder.", "backbone.", "model."):
            if any(k.startswith(prefix) for k in sd):
                sd = _strip_prefix(sd, prefix)
                break

        # 过滤 decoder / head 等无关键
        sd = {k: v for k, v in sd.items()
              if not any(k.startswith(p) for p in ("decoder.", "head.", "mask_token"))}

        if hasattr(self._inner, "vit"):
            self._inner.load_from_state_dict(sd, label="RoMA")
        else:
            _load_state_dict_relaxed(self._inner, sd, label="RoMA")


# ──────────────────────────────────────────────────────────────────────────────
# 5. RSMamba 适配器
# ──────────────────────────────────────────────────────────────────────────────

_RSMAMBA_CONFIGS: Dict[str, dict] = {
    "base":  dict(embed_dim=192, depth=24),
    "large": dict(embed_dim=256, depth=36),
    "huge":  dict(embed_dim=320, depth=48),
}


class RSMambaAdapter(BaseEncoder):
    """RSMamba 适配器。若官方代码不可用，fallback 为 timm ViT 代理。"""

    def __init__(self, img_size: int, model_size: str = "base"):
        super().__init__()
        rsmamba_dir = str(_BASELINES_DIR / "RSMamba")
        if rsmamba_dir not in sys.path:
            sys.path.insert(0, rsmamba_dir)

        cfg = _RSMAMBA_CONFIGS.get(model_size.lower(), _RSMAMBA_CONFIGS["base"])
        embed_dim, depth = cfg["embed_dim"], cfg["depth"]
        self._embed_dim = embed_dim
        self._depth = depth

        self._inner: BaseEncoder
        try:
            from rsmamba import RSMamba  # type: ignore
            backbone = RSMamba(
                img_size=img_size, embed_dim=embed_dim, depth=depth,
            )
            self._inner = _RSMambaRawAdapter(backbone, embed_dim, depth, img_size)
        except ImportError:
            warnings.warn(
                f"[RSMamba] 官方代码不可用，使用 timm ViT 代理 (embed_dim={embed_dim})。\n"
                f"请将 RSMamba 代码放到 {rsmamba_dir}/"
            )
            self._inner = ViTAdapter(
                "rsmamba", img_size=img_size,
                embed_dim=embed_dim, depth=depth,
                num_heads=max(1, embed_dim // 64),
                patch_size=16,
            )
        self.out_dims = self._inner.out_dims

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self._inner(x)

    def load_pretrained(self, ckpt_path: str) -> None:
        print(f"  [RSMamba] 加载权重: {ckpt_path}")
        raw = _safe_load(ckpt_path)
        sd = raw.get("model", raw.get("state_dict", raw))
        # MMPretrain 格式
        if any(k.startswith("backbone.") for k in sd):
            sd = _strip_prefix(sd, "backbone.")
        if hasattr(self._inner, "load_state_dict"):
            self._inner.load_state_dict(sd, strict=False)


class _RSMambaRawAdapter(BaseEncoder):
    """包装 RSMamba 官方模型为 BaseEncoder 接口。"""
    def __init__(self, backbone, embed_dim, depth, img_size):
        super().__init__()
        self.vit_adapter = ViTAdapter(
            "rsmamba", img_size=img_size,
            embed_dim=embed_dim, depth=depth,
            num_heads=max(1, embed_dim // 64),
        )
        self.vit_adapter.vit = backbone
        self.out_dims = self.vit_adapter.out_dims

    def forward(self, x):
        return self.vit_adapter(x)


# ──────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────────────────────

def _infer_rsmamba_size(ckpt: Optional[str], default: str = "base") -> str:
    """根据权重文件名推断 RSMamba 规模（b/l/h）。"""
    if not ckpt:
        return default
    stem = Path(ckpt).stem.lower()
    if (
        "-h_" in stem or "_h_" in stem or stem.endswith("-h") or "-h-" in stem
        or "-h" in stem.split("_")[0]  # e.g. RSMamba-h_UC
    ):
        return "huge"
    if "-l_" in stem or "_l_" in stem or stem.endswith("-l") or "-l-" in stem:
        return "large"
    if "-b_" in stem or "_b_" in stem or stem.endswith("-b") or "-b-" in stem:
        return "base"
    return default


def build_encoder(
    backbone: str,
    img_size: int,
    ckpt: Optional[str] = None,
    *,
    # EarthMamba
    ssm_version: str = "mamba3",
    use_armg: bool = True,
    use_graph: bool = True,
    # SkySense/Swin
    swin_size: str = "huge",
    # ViT 系列
    embed_dim: Optional[int] = None,
    depth: Optional[int] = None,
    num_heads: Optional[int] = None,
    patch_size: Optional[int] = None,
    # RSMamba 版本
    rsmamba_size: str = "base",
) -> BaseEncoder:
    """构建并返回指定 backbone 的 Encoder 适配器，可选加载预训练权重。

    Args:
        backbone:      模型名称（不区分大小写）
        img_size:      输入图片分辨率
        ckpt:          预训练权重路径（None = 随机初始化）
        ssm_version:   EarthMamba SSM 版本（mamba2/mamba3）
        swin_size:     SkySense Swin 尺寸（huge/large/base）
        embed_dim:     ViT embed_dim（None = 各模型默认值）
        depth:         ViT depth
        num_heads:     ViT num_heads
        patch_size:    ViT patch_size
        rsmamba_size:  RSMamba 版本（base/large/huge）

    Returns:
        BaseEncoder 实例（未 .to(device) / .train()）
    """
    name = backbone.lower().replace("-", "").replace("_", "")

    encoder: BaseEncoder

    if name in ("earthmamba", "em", "mamba3"):
        encoder = EarthMambaAdapter(
            img_size,
            ssm_version=ssm_version,
            use_armg=use_armg,
            use_graph=use_graph,
        )

    elif name in ("skysense", "swin", "swinv2"):
        encoder = SwinV2Adapter(img_size, model_size=swin_size)

    elif name in ("satmae",):
        peek = _peek_vit_config(ckpt) if ckpt else {}
        _ed = embed_dim or peek.get("embed_dim") or 768
        if (
            embed_dim is not None
            and peek.get("embed_dim")
            and embed_dim != peek["embed_dim"]
        ):
            warnings.warn(
                f"[SatMAE] 显式 embed_dim={embed_dim} 与 ckpt 推断 "
                f"{peek['embed_dim']} 不一致，以显式参数为准"
            )
            _ed = embed_dim
        _d  = depth or peek.get("depth") or 12
        _nh = num_heads or max(1, _ed // 64)
        _ps = patch_size or peek.get("patch_size") or 16
        if peek.get("pretrain_img_size") and peek["pretrain_img_size"] != img_size:
            print(
                f"  [SatMAE] ckpt 预训练分辨率 {peek['pretrain_img_size']}，"
                f"下游 img_size={img_size}（pos_embed 将自动插值）"
            )
        encoder = ViTAdapter("satmae", img_size=img_size,
                             embed_dim=_ed, depth=_d, num_heads=_nh, patch_size=_ps)

    elif name in ("dofa", "dofav2"):
        peek = _peek_vit_config(ckpt) if ckpt else {}
        _ed = peek.get("embed_dim") or embed_dim or 768
        _d  = depth or peek.get("depth") or (12 if name == "dofa" else 24)
        _nh = num_heads or max(1, _ed // 64)
        _ps = patch_size or peek.get("patch_size") or 16
        if peek.get("pretrain_img_size") and peek["pretrain_img_size"] != img_size:
            print(
                f"  [DOFA] ckpt 预训练分辨率 {peek['pretrain_img_size']}，"
                f"下游 img_size={img_size}（pos_embed 将自动插值）"
            )
        encoder = ViTAdapter("dofa", img_size=img_size,
                             embed_dim=_ed, depth=_d, num_heads=_nh, patch_size=_ps)

    elif name in ("clay",):
        _ed = embed_dim or 1024
        _d  = depth     or 24
        _nh = num_heads or 16
        _ps = patch_size or 8
        encoder = ViTAdapter("clay", img_size=img_size,
                             embed_dim=_ed, depth=_d, num_heads=_nh, patch_size=_ps)

    elif name in ("roma",):
        encoder = RoMAAdapter(img_size)

    elif name in ("rsmamba",):
        size = rsmamba_size if rsmamba_size != "base" or not ckpt else _infer_rsmamba_size(ckpt, rsmamba_size)
        if ckpt and size != rsmamba_size:
            print(f"  [RSMamba] 从权重文件名推断 model_size={size}")
        encoder = RSMambaAdapter(img_size, model_size=size)

    else:
        raise ValueError(
            f"未知 backbone: {backbone!r}。"
            f"可选: earthmamba / skysense / satmae / dofa / clay / roma / rsmamba"
        )

    if ckpt:
        if hasattr(encoder, "load_pretrained"):
            encoder.load_pretrained(ckpt)
        else:
            print(f"  [{backbone}] 警告: 无 load_pretrained 方法，跳过权重加载")
    else:
        print(f"  [{backbone}] 随机初始化（无预训练权重）")

    print(f"  [{backbone}] 参数量: {_count_params(encoder)}  out_dims={encoder.out_dims}")
    return encoder
