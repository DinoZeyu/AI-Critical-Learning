#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-22 42 62}"
STAGES="${STAGES:-all}"

for seed in ${SEEDS}; do
  echo
  echo "========== seed_${seed} =========="
  bash "${SCRIPT_DIR}/run_seed_experiments.sh" "${seed}" "${STAGES}"
done
