#!/usr/bin/env python3
"""检查 baselines 目录下各对比模型的权重与模型代码是否齐全。"""

from __future__ import annotations

import sys
from pathlib import Path

BASELINES_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINES_ROOT))

from shared.paths import BACKBONE_DIR_NAMES, DOWNSTREAM_ROOT, default_weights_dir

# 各模型期望的权重文件名（任一存在即可；* 表示 glob）
EXPECTED_WEIGHTS: dict[str, list[str]] = {
    "SkySense": ["skysense_model_backbone_hr.pth", "*.pth"],
    "SatMAE": ["pretrain-vit-base-e199.pth", "*.pth"],
    "DOFA": ["DOFA_ViT_base_e100.pth", "*.pth"],
    "Clay": ["clay-v1.5.ckpt", "*.ckpt", "*.pth"],
    "RoMA": ["mamba-base.pth", "*.pth"],
    "RSMamba": ["RSMamba-h_UC.pth", "RSMamba*.pth", "*.pth"],
}

# 各模型需要的 Python 代码（相对模型目录）
EXPECTED_CODE: dict[str, list[str]] = {
    "SkySense": ["swin_transformer_v2.py"],
    "SatMAE": [],  # timm fallback
    "DOFA": [],    # timm fallback（models_dwv.py 可选）
    "Clay": [],    # timm fallback
    "RoMA": [
        "models_mamba.py",
        "mamba_simple.py",
        "utils_mamba/pos_embed.py",
    ],
    "RSMamba": [],  # MMPretrain 结构；无独立 rsmamba.py 时用 timm fallback
}

OPTIONAL_CODE: dict[str, list[str]] = {
    "DOFA": ["models_dwv.py"],
    "SatMAE": ["models_vit.py"],
}


def _glob_any(weights_dir: Path, patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        if "*" in pat:
            found.extend(weights_dir.glob(pat))
        else:
            p = weights_dir / pat
            if p.is_file():
                found.append(p)
    return sorted(set(found))


def _fmt_size(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / 1024:.1f} KB"


def check_model(folder: str) -> tuple[bool, list[str]]:
    model_dir = BASELINES_ROOT / folder
    weights_dir = model_dir / "weights"
    lines: list[str] = []
    ok = True

    lines.append(f"\n=== {folder} ===")

    if not model_dir.is_dir():
        lines.append("  [FAIL] 模型目录不存在")
        return False, lines

    # 权重
    w_patterns = EXPECTED_WEIGHTS.get(folder, ["*.pth"])
    w_files = _glob_any(weights_dir, w_patterns) if weights_dir.is_dir() else []
    if w_files:
        for w in w_files:
            lines.append(f"  [OK] 权重: {w.name} ({_fmt_size(w.stat().st_size)})")
    else:
        lines.append(f"  [FAIL] 权重缺失: {weights_dir}/")
        ok = False

    # 必需代码
    for rel in EXPECTED_CODE.get(folder, []):
        p = model_dir / rel
        if p.is_file():
            lines.append(f"  [OK] 代码: {rel}")
        else:
            lines.append(f"  [FAIL] 代码缺失: {rel}")
            ok = False

    # 权重文件名提示（非 FAIL）
    if folder == "RSMamba":
        w = w_files[0].name.lower() if w_files else ""
        if "h_" in w or "-h" in w:
            lines.append("  [INFO] 权重为 Huge 版；build_encoder 会自动推断 model_size=huge")
        if not (model_dir / "rsmamba.py").is_file():
            lines.append("  [INFO] 无 rsmamba.py 时将使用 timm ViT 代理 + 加载 MMPretrain 权重（strict=False）")

    # 可选代码
    for rel in OPTIONAL_CODE.get(folder, []):
        p = model_dir / rel
        if p.is_file():
            lines.append(f"  [OK] 可选代码: {rel}")
        else:
            lines.append(f"  [WARN] 可选代码未放置: {rel}（将使用 timm fallback）")

    return ok, lines


def main() -> int:
    print(f"baselines 根目录: {BASELINES_ROOT}")
    print(f"downstream_code:  {DOWNSTREAM_ROOT}")

    all_ok = True
    from shared.paths import resolve_earth_mamba_root
    em = resolve_earth_mamba_root()
    if em:
        print(f"earth-mamba:      {em}")
    else:
        print("[WARN] 未找到 earth-mamba（需要 /hy-tmp/earth-mamba 或 /hy-tmp/downstream_code/earth-mamba）")
        all_ok = False

    if not DOWNSTREAM_ROOT.is_dir():
        print(f"[FAIL] downstream_code 不在预期位置: {DOWNSTREAM_ROOT}")
        all_ok = False

    for folder in sorted({v for v in BACKBONE_DIR_NAMES.values() if v}):
        ok, lines = check_model(folder)
        all_ok &= ok
        print("\n".join(lines))

    # 任务脚本
    print("\n=== 任务脚本 ===")
    for name in (
        "baselines_dfc15.py",
        "baselines_inria.py",
        "baselines_second.py",
        "baselines_dior.py",
    ):
        p = BASELINES_ROOT / name
        tag = "[OK]" if p.is_file() else "[FAIL]"
        print(f"  {tag} {name}")

    print("\n" + ("全部检查通过。" if all_ok else "存在缺失项，请按上方 [FAIL] 补齐。"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
