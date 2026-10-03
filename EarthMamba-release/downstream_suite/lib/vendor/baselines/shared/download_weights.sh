#!/usr/bin/env bash
# =============================================================================
#  下载 6 个对比模型权重脚本
#  用法: bash download_weights.sh [MODEL_NAME|all]
#  示例: bash download_weights.sh satmae
#         bash download_weights.sh all
#
#  默认目标路径：project_mamba/baselines/（与 downstream_code 同级）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"

download() {
    local url="$1" dst="$2"
    echo "  → $dst"
    mkdir -p "$(dirname "$dst")"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dst" "$url"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$dst" "$url"
    else
        echo "错误: 需要 wget 或 curl"; exit 1
    fi
}

# ──────────────────────────────────────────────────
# SatMAE  (ViT-Base, fMoW-Sentinel, ~86M, 最接近 89M)
# 来源: official SatMAE upstream project page
# ──────────────────────────────────────────────────
download_satmae() {
    echo "[SatMAE] ViT-Base pretrain (fMoW-Sentinel, 200ep) ..."
    download \
        "ANONYMOUS_EXTERNAL_WEIGHT_PLACEHOLDER/satmae/pretrain-vit-base-e199.pth" \
        "$BASELINES_DIR/SatMAE/weights/pretrain-vit-base-e199.pth"
    echo "[SatMAE] 完成 (~350MB)"
}

# ──────────────────────────────────────────────────
# DOFA  (ViT-Base, ~86M)
# 来源: omitted in anonymous supplement
# ──────────────────────────────────────────────────
download_dofa() {
    echo "[DOFA] ViT-Base pretrain (100ep) ..."
    download \
        "ANONYMOUS_EXTERNAL_WEIGHT_PLACEHOLDER/dofa/DOFA_ViT_base_e100.pth" \
        "$BASELINES_DIR/DOFA/weights/DOFA_ViT_base_e100.pth"
    echo "[DOFA] 完成 (~448MB)"
}

# ──────────────────────────────────────────────────
# DOFA-v2  (ViT-Large, 307M) 可选
# ──────────────────────────────────────────────────
download_dofav2() {
    echo "[DOFA-v2] ViT-Large pretrain (150ep) ..."
    download \
        "ANONYMOUS_EXTERNAL_WEIGHT_PLACEHOLDER/dofa/dofav2_vit_large_e150.pth" \
        "$BASELINES_DIR/DOFA/weights/dofav2_vit_large_e150.pth"
    echo "[DOFA-v2] 完成 (~1.3GB)"
}

# ──────────────────────────────────────────────────
# Clay  (ViT-Large encoder 311M / total 632M)
# 来源: omitted in anonymous supplement
# ──────────────────────────────────────────────────
download_clay() {
    echo "[Clay] v1.5 checkpoint (~1.25GB encoder) ..."
    download \
        "ANONYMOUS_EXTERNAL_WEIGHT_PLACEHOLDER/clay/clay-v1.5.ckpt" \
        "$BASELINES_DIR/Clay/weights/clay-v1.5.ckpt"
    echo "[Clay] 完成"
}

# ──────────────────────────────────────────────────
# SkySense  (Swin-V2-Huge HR encoder, 需申请)
# External request link omitted in anonymous supplement.
# ──────────────────────────────────────────────────
download_skysense() {
    echo "[SkySense] 注意: 权重需要通过官方渠道申请下载"
    echo "  官方项目页: obtain from the upstream SkySense project page"
    echo "  Checkpoint request link omitted in the anonymous supplement."
    echo "  下载后将文件放到: $BASELINES_DIR/SkySense/weights/"
    echo "  期望文件名: skysense_model_backbone_hr.pth"
    echo "  另需将 SkySense 仓库中的 models/ 目录复制到 $BASELINES_DIR/SkySense/"
}

# ──────────────────────────────────────────────────
# RoMA  (Mamba-Base, 85M, NeurIPS 2025)
# 来源: omitted in anonymous supplement
# ──────────────────────────────────────────────────
download_roma() {
    echo "[RoMA] Mamba-Base pretrain (~1.13GB) ..."
    download \
        "ANONYMOUS_EXTERNAL_WEIGHT_PLACEHOLDER/roma/mamba-base.pth" \
        "$BASELINES_DIR/RoMA/weights/mamba-base.pth"
    echo "[RoMA] 完成 (~1.13GB)"
    echo "  注: 还需从 RoMA upstream project 复制 models_mamba.py 到 $BASELINES_DIR/RoMA/"
}

# ──────────────────────────────────────────────────
# RSMamba  (Mamba-Base, N=24, HS=192)
# 来源: omitted in anonymous supplement
# ──────────────────────────────────────────────────
download_rsmamba() {
    echo "[RSMamba] 权重从 HuggingFace 获取"
    echo "  External weight link omitted in the anonymous supplement."
    echo "  下载 Base 版本权重后放到: $BASELINES_DIR/RSMamba/weights/"
    echo "  另需将 RSMamba GitHub 中的模型代码复制到 $BASELINES_DIR/RSMamba/"
    # 若 HF hub 已安装可用以下命令:
    # Place rsmamba_base.pth in $BASELINES_DIR/RSMamba/weights/ when available.
}

# ──────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────
MODEL="${1:-all}"
case "$MODEL" in
    satmae)    download_satmae ;;
    dofa)      download_dofa ;;
    dofav2)    download_dofav2 ;;
    clay)      download_clay ;;
    skysense)  download_skysense ;;
    roma)      download_roma ;;
    rsmamba)   download_rsmamba ;;
    all)
        download_satmae
        download_dofa
        download_clay
        download_roma
        download_skysense
        download_rsmamba
        echo ""
        echo "=== 完成 ==="
        echo "注: SkySense / RSMamba 需手动获取，详见上方提示。RoMA 已自动下载。"
        ;;
    *)
        echo "用法: bash download_weights.sh [satmae|dofa|dofav2|clay|skysense|roma|rsmamba|all]"
        exit 1
        ;;
esac
