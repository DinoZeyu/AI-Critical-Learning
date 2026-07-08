#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEST_ROOT="${1:-seed_42}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

copy_dir() {
  local source="$1"
  local target="$2"

  if [[ ! -e "${source}" ]]; then
    echo "==> Source missing, skipping: ${source}"
    return
  fi

  if [[ -e "${target}" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" ]]; then
      echo "==> Target exists, skipping: ${target}"
      return
    fi
    echo "Refusing to overwrite existing target: ${target}" >&2
    echo "Set SKIP_EXISTING=1 to skip it, or choose another destination root." >&2
    exit 1
  fi

  mkdir -p "$(dirname "${target}")"
  echo "==> Copying ${source} -> ${target}"
  cp -a "${source}" "${target}"
}

mkdir -p "${DEST_ROOT}"

copy_dir "Image_Data/Train_Noise_Data" "${DEST_ROOT}/Train_Noise_Data"
copy_dir "Gold_Evaluators" "${DEST_ROOT}/Gold_Evaluators"
copy_dir \
  "Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning" \
  "${DEST_ROOT}/Experiments_Results"
copy_dir \
  "Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp" \
  "${DEST_ROOT}/Noise_Baseline"
copy_dir "Supplemental_Ablation_Results" "${DEST_ROOT}/Supplemental_Ablation_Results"

echo
echo "Done. Legacy seed42 artifacts were copied under: ${DEST_ROOT}"
