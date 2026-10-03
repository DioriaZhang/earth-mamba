#!/usr/bin/env bash
# Example environment — edit paths, then: source env.example.sh

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export RUN_DIR

export EARTH_MAMBA_ROOT="${EARTH_MAMBA_ROOT:-/path/to/earth-mamba}"
export CKPT="${CKPT:-/path/to/earth-mamba-small.pth}"

export DATA_INRIA="${DATA_INRIA:-/path/to/datasets/INRIA}"
export DATA_DFC15="${DATA_DFC15:-/path/to/datasets/DFC15}"
export DATA_DIOR="${DATA_DIOR:-/path/to/datasets/DIOR}"

export RESULTS_ROOT="${RESULTS_ROOT:-${RUN_DIR}/outputs}"
export PYTHONPATH="${RUN_DIR}:${EARTH_MAMBA_ROOT}:${PYTHONPATH:-}"

echo "[ablation_suite] RUN_DIR=${RUN_DIR}"
echo "[ablation_suite] RESULTS_ROOT=${RESULTS_ROOT}"
