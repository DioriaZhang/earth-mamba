#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env_a100.sh"

VARIANTS=(ID1 ID3 ID4 ID5 ID6 ID7 ID8)

run_one() {
  python "${SCRIPT_DIR}/run_variant.py" --task "$1" --variant "$2" --data_dir "$3" \
    --ckpt "${CKPT}" --output_dir "${RESULTS_ROOT}/$1/$2"
}

for v in "${VARIANTS[@]}"; do run_one dfc15 "$v" "${DATA_DFC15}"; done
for v in "${VARIANTS[@]}"; do run_one inria "$v" "${DATA_INRIA}"; done
for v in "${VARIANTS[@]}"; do run_one dior  "$v" "${DATA_DIOR}"; done

echo "Done (21 jobs). Collect: python ${SCRIPT_DIR}/collect_results.py --results_root ${RESULTS_ROOT}"
