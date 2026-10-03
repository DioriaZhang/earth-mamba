#!/usr/bin/env bash
# A100 single-GPU env — source before ablation runs
_strip_cr() { printf '%s' "$1" | tr -d '\r'; }

RUN_DIR_RAW="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export RUN_DIR="$(_strip_cr "${RUN_DIR_RAW}")"
export PROJECT_ROOT="$(_strip_cr "${PROJECT_ROOT:-$(cd "${RUN_DIR}/../.." && pwd)}")"
export EARTH_MAMBA_ROOT="$(_strip_cr "${EARTH_MAMBA_ROOT:-${PROJECT_ROOT}/earth-mamba}")"

export CKPT="$(_strip_cr "${CKPT:-/hy-tmp/CKPT/PN-log13-ep14/checkpoint.pth}")"
export DATA_INRIA="$(_strip_cr "${DATA_INRIA:-/hy-tmp/task/INRIA}")"
export DATA_DFC15="$(_strip_cr "${DATA_DFC15:-/hy-tmp/task/DFC15}")"
export DATA_DIOR="$(_strip_cr "${DATA_DIOR:-/hy-tmp/task/DIOR}")"
export RESULTS_ROOT="$(_strip_cr "${RESULTS_ROOT:-/hy-tmp/downstream_results/ablation_study}")"
export CUDA_VISIBLE_DEVICES="$(_strip_cr "${CUDA_VISIBLE_DEVICES:-0}")"

export NUM_WORKERS="$(_strip_cr "${NUM_WORKERS:-8}")"
export PREFETCH_FACTOR="$(_strip_cr "${PREFETCH_FACTOR:-4}")"
export PYTORCH_CUDA_ALLOC_CONF="$(_strip_cr "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}")"

export PYTHONPATH="${RUN_DIR}:${EARTH_MAMBA_ROOT}:${PYTHONPATH:-}"

echo "[ablation] RUN_DIR=${RUN_DIR}"
echo "[ablation] EARTH_MAMBA_ROOT=${EARTH_MAMBA_ROOT}"
echo "[ablation] CKPT=${CKPT}"
echo "[ablation] A100 perf: NUM_WORKERS=${NUM_WORKERS} PREFETCH=${PREFETCH_FACTOR}"
