#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/seeded_runs/run_seed_experiments.sh <seed> [stages]" >&2
  echo "Stages: all noise-data gold-evaluators noise-baselines ggcl supplemental-ablation clean-baselines" >&2
  exit 1
fi

SEED="$1"
REQUESTED_STAGES="${2:-all}"

case "${SEED}" in
  ''|*[!0-9]*)
    echo "Seed must be an integer: ${SEED}" >&2
    exit 1
    ;;
esac

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE="${FORCE:-0}"
REPLACE_EXISTING="${REPLACE_EXISTING:-0}"
OVERWRITE_NOISE="${OVERWRITE_NOISE:-${FORCE}}"

SEED_ROOT="seed_${SEED}"
NOISE_ROOT="${SEED_ROOT}/Train_Noise_Data"
EVALUATORS_ROOT="${SEED_ROOT}/Gold_Evaluators"
RESULTS_ROOT="${SEED_ROOT}/Experiments_Results"
RAW_GGCL_RESULTS_ROOT="${SEED_ROOT}/.ggcl_raw"
NOISE_BASELINE_ROOT="${SEED_ROOT}/Noise_Baseline"
RAW_NOISE_BASELINE_ROOT="${SEED_ROOT}/.noise_baseline_raw"
ABLATION_ROOT="${SEED_ROOT}/Supplemental_Ablation_Results"
RAW_ABLATION_ROOT="${SEED_ROOT}/.supplemental_ablation_raw"
RAW_CLEAN_BASELINE_ROOT=".clean_baseline_raw"
GGCL_MANIFEST="${SCRIPT_DIR}/manifests/ggcl_seed42_runs.tsv"
SUPPLEMENTAL_MANIFEST="${SCRIPT_DIR}/manifests/supplemental_ablation_seed42_runs.tsv"

ALLOWED_STAGES="all noise-data gold-evaluators noise-baselines ggcl supplemental-ablation clean-baselines"

validate_stages() {
  local stage
  for stage in ${REQUESTED_STAGES}; do
    case " ${ALLOWED_STAGES} " in
      *" ${stage} "*) ;;
      *)
        echo "Unknown stage: ${stage}" >&2
        echo "Allowed stages: ${ALLOWED_STAGES}" >&2
        exit 1
        ;;
    esac
  done
}

stage_enabled() {
  local stage="$1"
  if [[ " ${REQUESTED_STAGES} " == *" ${stage} "* ]]; then
    return 0
  fi
  [[ " ${REQUESTED_STAGES} " == *" all "* && "${stage}" != "clean-baselines" ]]
}

ensure_can_write() {
  local path="$1"
  local label="$2"

  if [[ ! -e "${path}" ]]; then
    return 0
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    echo "==> Skipping existing ${label}: ${path}"
    return 1
  fi

  if [[ "${FORCE}" == "1" ]]; then
    echo "==> FORCE=1, reusing existing ${label}: ${path}"
    return 0
  fi

  echo "Refusing to overwrite existing ${label}: ${path}" >&2
  echo "Set SKIP_EXISTING=1 to skip it, or FORCE=1 if overwrite is intentional." >&2
  exit 1
}

should_run_final_dir() {
  local final_dir="$1"
  local marker="$2"
  local label="$3"

  if [[ -e "${marker}" ]]; then
    if [[ "${REPLACE_EXISTING}" == "1" ]]; then
      echo "==> REPLACE_EXISTING=1, will replace existing ${label}: ${final_dir}"
      return 0
    fi
    if [[ "${SKIP_EXISTING}" == "1" ]]; then
      echo "==> Skipping existing ${label}: ${marker}"
      return 1
    fi
    echo "Refusing to overwrite existing ${label}: ${final_dir}" >&2
    echo "Set SKIP_EXISTING=1 to skip it, or REPLACE_EXISTING=1 to replace it after a successful rerun." >&2
    exit 1
  fi

  if [[ -e "${final_dir}" ]]; then
    if [[ "${REPLACE_EXISTING}" == "1" ]]; then
      echo "==> REPLACE_EXISTING=1, will replace incomplete existing ${label}: ${final_dir}"
      return 0
    fi
    echo "Refusing to write into existing incomplete ${label}: ${final_dir}" >&2
    echo "Move it aside or set REPLACE_EXISTING=1 to replace it after a successful rerun." >&2
    exit 1
  fi
}

install_generated_dir() {
  local generated_dir="$1"
  local final_dir="$2"
  local label="$3"

  if [[ ! -d "${generated_dir}" ]]; then
    echo "Expected generated ${label} directory was not found: ${generated_dir}" >&2
    exit 1
  fi

  if [[ -e "${final_dir}" ]]; then
    if [[ "${REPLACE_EXISTING}" != "1" ]]; then
      echo "Refusing to replace existing ${label}: ${final_dir}" >&2
      echo "Set REPLACE_EXISTING=1 if replacing this canonical result is intentional." >&2
      exit 1
    fi
    rm -rf -- "${final_dir}"
  fi

  mkdir -p "$(dirname "${final_dir}")"
  mv "${generated_dir}" "${final_dir}"
}

run_noisy_dataset() {
  local dataset="$1"
  local group="$2"
  local noise_type="$3"
  local value="$4"
  local output_name="$5"
  local target_dir="${NOISE_ROOT}/${dataset}/${group}/${output_name}"
  local -a overwrite_args=()

  if ! ensure_can_write "${target_dir}" "noisy data"; then
    return
  fi
  if [[ "${OVERWRITE_NOISE}" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi

  echo
  echo "==> make_noisy_data dataset=${dataset} seed=${SEED} ${group}/${output_name}"
  "${PYTHON_BIN}" scripts/Data_Prepration/make_noisy_data.py \
    --dataset "${dataset}" \
    --split train \
    --noise-type "${noise_type}" \
    --value "${value}" \
    --seed "${SEED}" \
    --noise-data-root "${NOISE_ROOT}" \
    --output-name "${output_name}" \
    "${overwrite_args[@]}"
}

run_hybrid_noisy_dataset() {
  local dataset="$1"
  local feature_noise_type="$2"
  local feature_value="$3"
  local output_name="$4"
  local target_dir="${NOISE_ROOT}/${dataset}/hybrid_noise/${output_name}"
  local -a overwrite_args=()

  if ! ensure_can_write "${target_dir}" "hybrid noisy data"; then
    return
  fi
  if [[ "${OVERWRITE_NOISE}" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi

  echo
  echo "==> make_hybrid_noisy_data dataset=${dataset} seed=${SEED} hybrid_noise/${output_name}"
  "${PYTHON_BIN}" scripts/Data_Prepration/make_hybrid_noisy_data.py \
    --dataset "${dataset}" \
    --split train \
    --feature-noise-type "${feature_noise_type}" \
    --feature-value "${feature_value}" \
    --label-shuffle-fraction 0.2 \
    --seed "${SEED}" \
    --noise-data-root "${NOISE_ROOT}" \
    --output-name "${output_name}" \
    "${overwrite_args[@]}"
}

run_noise_data_stage() {
  local dataset
  for dataset in STL Flower_102; do
    run_noisy_dataset "${dataset}" feature_noise blur 3 blur_3p0
    run_noisy_dataset "${dataset}" feature_noise brightness 0.75 brightness_0p75
    run_noisy_dataset "${dataset}" feature_noise gaussian 30 gaussian_30p0
    run_noisy_dataset "${dataset}" label_noise label_shuffle 0.2 label_shuffle_0p2
    run_hybrid_noisy_dataset "${dataset}" blur 3 blur_3p0_label_shuffle_0p2
    run_hybrid_noisy_dataset "${dataset}" brightness 0.75 brightness_0p75_label_shuffle_0p2
    run_hybrid_noisy_dataset "${dataset}" gaussian 30 gaussian_30p0_label_shuffle_0p2
  done
}

run_gold_evaluator() {
  local dataset="$1"
  local patience="$2"
  local target_metrics="${EVALUATORS_ROOT}/${dataset}/metrics.json"

  if ! ensure_can_write "${target_metrics}" "gold evaluator"; then
    return
  fi

  echo
  echo "==> Train_Gold_Evaluator dataset=${dataset} seed=${SEED}"
  "${PYTHON_BIN}" scripts/Train_Gold_Evaluator.py \
    --dataset "${dataset}" \
    --epochs 40 \
    --lr 0.0005 \
    --selection-metric val-loss \
    --train-augmentation feature-noise \
    --early-stop-patience "${patience}" \
    --early-stop-min-delta 0.001 \
    --seed "${SEED}" \
    --evaluators-root "${EVALUATORS_ROOT}"
}

run_gold_evaluators_stage() {
  run_gold_evaluator STL 6
  run_gold_evaluator Flower_102 5
}

run_clean_baseline() {
  local dataset="$1"
  local generated_dir="${RAW_CLEAN_BASELINE_ROOT}/Train_Clean_Test_Clean/${dataset}"
  local final_dir="Clean_Baseline/${dataset}"
  local target_metrics="${final_dir}/metrics.json"

  if ! should_run_final_dir "${final_dir}" "${target_metrics}" "clean baseline"; then
    return
  fi
  if [[ -e "${generated_dir}" ]]; then
    echo "Raw clean baseline directory already exists: ${generated_dir}" >&2
    echo "Move or remove it before rerunning this baseline." >&2
    exit 1
  fi

  echo
  echo "==> Baseline_Exp clean dataset=${dataset} shared root baseline"
  "${PYTHON_BIN}" scripts/Baseline_Exp.py \
    --dataset "${dataset}" \
    --early-stop-patience 3 \
    --selection-metric val-loss \
    --seed 42 \
    --results-root "${RAW_CLEAN_BASELINE_ROOT}"

  install_generated_dir "${generated_dir}" "${final_dir}" "clean baseline"
  echo "Moved clean baseline result to: ${final_dir}"
}

run_clean_baselines_stage() {
  run_clean_baseline STL
  run_clean_baseline Flower_102
}

run_noise_baseline() {
  local dataset="$1"
  local train_rel="$2"
  local generated_dir="${RAW_NOISE_BASELINE_ROOT}/Train_Noise_Test_Clean/Baseline_Exp/${dataset}/${train_rel}"
  local final_dir="${NOISE_BASELINE_ROOT}/${dataset}/${train_rel}"
  local target_metrics="${final_dir}/metrics.json"

  if ! should_run_final_dir "${final_dir}" "${target_metrics}" "noise baseline"; then
    return
  fi
  if [[ -e "${generated_dir}" ]]; then
    echo "Raw noise baseline directory already exists: ${generated_dir}" >&2
    echo "Move or remove it before rerunning this baseline." >&2
    exit 1
  fi

  echo
  echo "==> Baseline_Exp noisy dataset=${dataset} seed=${SEED} ${train_rel}"
  "${PYTHON_BIN}" scripts/Baseline_Exp.py \
    --dataset "${dataset}" \
    --train-dir "${NOISE_ROOT}/${dataset}/${train_rel}" \
    --test-dir "Image_Data/Test_Clean_Data/${dataset}" \
    --early-stop-patience 3 \
    --selection-metric val-loss \
    --seed "${SEED}" \
    --results-root "${RAW_NOISE_BASELINE_ROOT}" \
    --run-name "${train_rel}"

  install_generated_dir "${generated_dir}" "${final_dir}" "noise baseline"
  echo "Moved noise baseline result to: ${final_dir}"
}

run_noise_baselines_stage() {
  local dataset
  local train_rel
  for dataset in STL Flower_102; do
    for train_rel in \
      feature_noise/blur_3p0 \
      feature_noise/brightness_0p75 \
      feature_noise/gaussian_30p0 \
      label_noise/label_shuffle_0p2 \
      hybrid_noise/blur_3p0_label_shuffle_0p2 \
      hybrid_noise/brightness_0p75_label_shuffle_0p2 \
      hybrid_noise/gaussian_30p0_label_shuffle_0p2
    do
      run_noise_baseline "${dataset}" "${train_rel}"
    done
  done
}

run_ggcl() {
  local dataset="$1"
  local train_rel="$2"
  local beta="$3"
  local lambda_gold="$4"
  local run_name="$5"
  local generated_dir="${RAW_GGCL_RESULTS_ROOT}/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/${dataset}/${run_name}"
  local final_dir="${RESULTS_ROOT}/${dataset}/${run_name}"
  local target_metrics="${final_dir}/metrics.json"

  if ! should_run_final_dir "${final_dir}" "${target_metrics}" "GGCL run"; then
    return
  fi
  if [[ -e "${generated_dir}" ]]; then
    echo "Raw GGCL directory already exists: ${generated_dir}" >&2
    echo "Move or remove it before rerunning this GGCL run." >&2
    exit 1
  fi

  echo
  echo "==> GGCL dataset=${dataset} seed=${SEED} beta=${beta} lambda_gold=${lambda_gold} ${run_name}"
  "${PYTHON_BIN}" scripts/Gold_Guided_Critical_Learning.py \
    --dataset "${dataset}" \
    --mixed-train-dir "${NOISE_ROOT}/${dataset}/${train_rel}" \
    --gold-evaluator-checkpoint "${EVALUATORS_ROOT}/${dataset}/model.pt" \
    --results-root "${RAW_GGCL_RESULTS_ROOT}" \
    --early-stop-patience 3 \
    --selection-metric val-loss \
    --seed "${SEED}" \
    --beta "${beta}" \
    --lambda-gold "${lambda_gold}" \
    --run-name "${run_name}"

  install_generated_dir "${generated_dir}" "${final_dir}" "GGCL run"
  "${PYTHON_BIN}" - "${final_dir}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metrics_path = run_dir / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    metrics["peak_test_checkpoint"] = str(run_dir / "model_peak_test.pt")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
PY
  echo "Moved GGCL result to: ${final_dir}"
}

run_manifest_ggcl_stage() {
  local dataset
  local train_rel
  local beta
  local lambda_gold
  local run_name

  while IFS='|' read -r dataset train_rel beta lambda_gold run_name; do
    [[ -z "${dataset}" || "${dataset}" == \#* ]] && continue
    run_ggcl "${dataset}" "${train_rel}" "${beta}" "${lambda_gold}" "${run_name}"
  done < "${GGCL_MANIFEST}"

  find "${RAW_GGCL_RESULTS_ROOT}" -type d -empty -delete 2>/dev/null || true
}

run_supplemental_ablation() {
  local dataset="$1"
  local train_rel="$2"
  local beta="$3"
  local lambda_gold="$4"
  local run_name="$5"
  local group="${run_name%%/*}"
  local run_leaf="${run_name#*/}"
  local generated_dir="${RAW_ABLATION_ROOT}/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/${dataset}/${run_name}"
  local final_dir="${ABLATION_ROOT}/${dataset}/${group}/${run_leaf}"
  local target_metrics="${final_dir}/metrics.json"

  if ! should_run_final_dir "${final_dir}" "${target_metrics}" "supplemental ablation"; then
    return
  fi
  if [[ -e "${generated_dir}" ]]; then
    echo "Raw supplemental directory already exists: ${generated_dir}" >&2
    echo "Move or remove it before rerunning this ablation." >&2
    exit 1
  fi

  echo
  echo "==> Supplemental ablation dataset=${dataset} seed=${SEED} beta=${beta} lambda_gold=${lambda_gold} ${run_name}"
  "${PYTHON_BIN}" scripts/Gold_Guided_Critical_Learning.py \
    --dataset "${dataset}" \
    --mixed-train-dir "${NOISE_ROOT}/${dataset}/${train_rel}" \
    --gold-evaluator-checkpoint "${EVALUATORS_ROOT}/${dataset}/model.pt" \
    --results-root "${RAW_ABLATION_ROOT}" \
    --early-stop-patience 3 \
    --selection-metric val-loss \
    --seed "${SEED}" \
    --beta "${beta}" \
    --lambda-gold "${lambda_gold}" \
    --run-name "${run_name}"

  install_generated_dir "${generated_dir}" "${final_dir}" "supplemental ablation"
  "${PYTHON_BIN}" - "${final_dir}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metrics_path = run_dir / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    metrics["peak_test_checkpoint"] = str(run_dir / "model_peak_test.pt")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
PY
  echo "Moved supplemental ablation result to: ${final_dir}"
}

run_supplemental_ablation_stage() {
  local dataset
  local train_rel
  local beta
  local lambda_gold
  local run_name

  while IFS='|' read -r dataset train_rel beta lambda_gold run_name; do
    [[ -z "${dataset}" || "${dataset}" == \#* ]] && continue
    run_supplemental_ablation "${dataset}" "${train_rel}" "${beta}" "${lambda_gold}" "${run_name}"
  done < "${SUPPLEMENTAL_MANIFEST}"

  find "${RAW_ABLATION_ROOT}" -type d -empty -delete 2>/dev/null || true
}

ensure_seed_launcher() {
  local launcher="${SEED_ROOT}/run_all_experiments.sh"

  if [[ -e "${launcher}" ]]; then
    return
  fi

  cat > "${launcher}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="\$(cd "\${SCRIPT_DIR}/.." && pwd)"
cd "\${REPO_ROOT}"

STAGES="\${1:-all}"
bash scripts/seeded_runs/run_seed_experiments.sh ${SEED} "\${STAGES}"
EOF
  chmod +x "${launcher}"
}

validate_stages
mkdir -p "${SEED_ROOT}" "${NOISE_ROOT}" "${EVALUATORS_ROOT}" "${RESULTS_ROOT}" "${NOISE_BASELINE_ROOT}" "${ABLATION_ROOT}"
ensure_seed_launcher

echo "Seed: ${SEED}"
echo "Seed root: ${SEED_ROOT}"
echo "Stages: ${REQUESTED_STAGES}"
echo "Python: ${PYTHON_BIN}"
echo "Replace existing final outputs: ${REPLACE_EXISTING}"

if stage_enabled noise-data; then
  run_noise_data_stage
fi
if stage_enabled gold-evaluators; then
  run_gold_evaluators_stage
fi
if stage_enabled clean-baselines; then
  run_clean_baselines_stage
fi
if stage_enabled noise-baselines; then
  run_noise_baselines_stage
fi
if stage_enabled ggcl; then
  run_manifest_ggcl_stage
fi
if stage_enabled supplemental-ablation; then
  run_supplemental_ablation_stage
fi

echo
echo "Done. Seed artifacts are under: ${SEED_ROOT}"
