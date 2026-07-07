#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-Supplemental_Ablation_Results}"
RAW_RESULTS_ROOT="${RAW_RESULTS_ROOT:-.supplemental_ablation_raw}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

run_ggcl() {
  local dataset="$1"
  local train_dir="$2"
  local beta="$3"
  local lambda_gold="$4"
  local run_name="$5"
  local group="${run_name%%/*}"
  local run_leaf="${run_name#*/}"
  local generated_dir="${RAW_RESULTS_ROOT}/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/${dataset}/${run_name}"
  local final_dir="${RESULTS_ROOT}/${dataset}/${group}/${run_leaf}"

  if [[ -e "${final_dir}" && "${SKIP_EXISTING}" == "1" ]]; then
    echo
    echo "==> Skipping existing result: ${final_dir}"
    return
  fi
  if [[ -e "${final_dir}" ]]; then
    echo "Refusing to overwrite existing result: ${final_dir}" >&2
    echo "Remove it first, or run with SKIP_EXISTING=1 to skip completed runs." >&2
    exit 1
  fi

  echo
  echo "==> ${dataset} beta=${beta} lambda_gold=${lambda_gold} ${run_name}"
  "${PYTHON_BIN}" scripts/Gold_Guided_Critical_Learning.py --dataset "${dataset}" \
    --mixed-train-dir "${train_dir}" \
    --gold-evaluator-checkpoint "Gold_Evaluators/${dataset}/model.pt" \
    --results-root "${RAW_RESULTS_ROOT}" \
    --early-stop-patience 3 \
    --selection-metric val-loss \
    --beta "${beta}" \
    --lambda-gold "${lambda_gold}" \
    --run-name "${run_name}"

  if [[ ! -d "${generated_dir}" ]]; then
    echo "Expected generated result directory was not found: ${generated_dir}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${final_dir}")"
  mv "${generated_dir}" "${final_dir}"
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
  echo "Moved result to: ${final_dir}"
}

echo "Writing supplemental ablation results under: ${RESULTS_ROOT}"

"${PYTHON_BIN}" - "${RESULTS_ROOT}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit

for group in ("lambda0", "beta00", "beta10"):
    group_dir = root / group
    if not group_dir.exists():
        continue
    for metrics_path in sorted(group_dir.rglob("metrics.json")):
        run_dir = metrics_path.parent
        relative = run_dir.relative_to(group_dir)
        metrics = json.loads(metrics_path.read_text())
        dataset = metrics.get("dataset")
        if dataset not in {"STL", "Flower_102"}:
            continue
        if relative.parts and relative.parts[0] in {"STL", "Flower_102"}:
            relative = Path(*relative.parts[1:])
        target_dir = root / dataset / group / relative
        if target_dir.exists():
            continue
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(target_dir))
        target_metrics_path = target_dir / "metrics.json"
        target_metrics = json.loads(target_metrics_path.read_text())
        target_metrics["peak_test_checkpoint"] = str(target_dir / "model_peak_test.pt")
        target_metrics_path.write_text(
            json.dumps(target_metrics, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"Migrated legacy result to: {target_dir}")

for group in ("lambda0", "beta00", "beta10"):
    group_dir = root / group
    if group_dir.exists():
        for path in sorted(group_dir.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            group_dir.rmdir()
        except OSError:
            pass
PY

# lambda_gold=0 ablations: one per dataset/noise condition, using each condition's best beta.
run_ggcl STL Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 0.90 0.0 lambda0/feature_noise/blur/blur_3p0_beta09_lg00
run_ggcl STL Image_Data/Train_Noise_Data/STL/feature_noise/brightness_0p75 0.90 0.0 lambda0/feature_noise/brightness/brightness_0p75_beta09_lg00
run_ggcl STL Image_Data/Train_Noise_Data/STL/feature_noise/gaussian_30p0 0.50 0.0 lambda0/feature_noise/gaussian/gaussian_30p0_beta05_lg00
run_ggcl STL Image_Data/Train_Noise_Data/STL/label_noise/label_shuffle_0p2 0.50 0.0 lambda0/label_noise/label_shuffle/label_shuffle_0p2_beta05_lg00
run_ggcl STL Image_Data/Train_Noise_Data/STL/hybrid_noise/blur_3p0_label_shuffle_0p2 0.90 0.0 lambda0/hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_beta09_lg00
run_ggcl STL Image_Data/Train_Noise_Data/STL/hybrid_noise/brightness_0p75_label_shuffle_0p2 0.85 0.0 lambda0/hybrid_noise/brightness_label_shuffle/brightness_0p75_label_shuffle_0p2_beta085_lg00
run_ggcl STL Image_Data/Train_Noise_Data/STL/hybrid_noise/gaussian_30p0_label_shuffle_0p2 0.60 0.0 lambda0/hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_beta06_lg00

run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/feature_noise/blur_3p0 0.80 0.0 lambda0/feature_noise/blur/blur_3p0_beta08_lg00
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 0.80 0.0 lambda0/feature_noise/brightness/brightness_0p75_beta08_lg00
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 0.50 0.0 lambda0/feature_noise/gaussian/gaussian_30p0_beta05_lg00
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 0.95 0.0 lambda0/label_noise/label_shuffle/label_shuffle_0p2_beta095_lg00
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/hybrid_noise/blur_3p0_label_shuffle_0p2 0.70 0.0 lambda0/hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_beta07_lg00
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/hybrid_noise/brightness_0p75_label_shuffle_0p2 0.80 0.0 lambda0/hybrid_noise/brightness_label_shuffle/brightness_0p75_label_shuffle_0p2_beta08_lg00
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/hybrid_noise/gaussian_30p0_label_shuffle_0p2 0.50 0.0 lambda0/hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_beta05_lg00

# beta=0 ablations: prediction-confidence signal off, prototype-similarity only.
run_ggcl STL Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 0.0 0.15 beta00/feature_noise/blur/blur_3p0_beta00_lg015
run_ggcl STL Image_Data/Train_Noise_Data/STL/feature_noise/brightness_0p75 0.0 0.10 beta00/feature_noise/brightness/brightness_0p75_beta00_lg01
run_ggcl STL Image_Data/Train_Noise_Data/STL/feature_noise/gaussian_30p0 0.0 0.15 beta00/feature_noise/gaussian/gaussian_30p0_beta00_lg015
run_ggcl STL Image_Data/Train_Noise_Data/STL/label_noise/label_shuffle_0p2 0.0 0.25 beta00/label_noise/label_shuffle/label_shuffle_0p2_beta00_lg025
run_ggcl STL Image_Data/Train_Noise_Data/STL/hybrid_noise/blur_3p0_label_shuffle_0p2 0.0 0.25 beta00/hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_beta00_lg025
run_ggcl STL Image_Data/Train_Noise_Data/STL/hybrid_noise/brightness_0p75_label_shuffle_0p2 0.0 0.10 beta00/hybrid_noise/brightness_label_shuffle/brightness_0p75_label_shuffle_0p2_beta00_lg01
run_ggcl STL Image_Data/Train_Noise_Data/STL/hybrid_noise/gaussian_30p0_label_shuffle_0p2 0.0 0.15 beta00/hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_beta00_lg015

run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/feature_noise/blur_3p0 0.0 0.10 beta00/feature_noise/blur/blur_3p0_beta00_lg01
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 0.0 0.10 beta00/feature_noise/brightness/brightness_0p75_beta00_lg01
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 0.0 0.15 beta00/feature_noise/gaussian/gaussian_30p0_beta00_lg015
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 0.0 0.10 beta00/label_noise/label_shuffle/label_shuffle_0p2_beta00_lg01
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/hybrid_noise/blur_3p0_label_shuffle_0p2 0.0 0.10 beta00/hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_beta00_lg01
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/hybrid_noise/brightness_0p75_label_shuffle_0p2 0.0 0.10 beta00/hybrid_noise/brightness_label_shuffle/brightness_0p75_label_shuffle_0p2_beta00_lg01
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/hybrid_noise/gaussian_30p0_label_shuffle_0p2 0.0 0.10 beta00/hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_beta00_lg01

# beta=1 ablations: prototype-similarity signal off, prediction-confidence only.
run_ggcl STL Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 1.0 0.15 beta10/feature_noise/blur/blur_3p0_beta10_lg015
run_ggcl STL Image_Data/Train_Noise_Data/STL/feature_noise/brightness_0p75 1.0 0.10 beta10/feature_noise/brightness/brightness_0p75_beta10_lg01
run_ggcl STL Image_Data/Train_Noise_Data/STL/feature_noise/gaussian_30p0 1.0 0.15 beta10/feature_noise/gaussian/gaussian_30p0_beta10_lg015
run_ggcl STL Image_Data/Train_Noise_Data/STL/label_noise/label_shuffle_0p2 1.0 0.25 beta10/label_noise/label_shuffle/label_shuffle_0p2_beta10_lg025
run_ggcl STL Image_Data/Train_Noise_Data/STL/hybrid_noise/blur_3p0_label_shuffle_0p2 1.0 0.25 beta10/hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_beta10_lg025
run_ggcl STL Image_Data/Train_Noise_Data/STL/hybrid_noise/brightness_0p75_label_shuffle_0p2 1.0 0.10 beta10/hybrid_noise/brightness_label_shuffle/brightness_0p75_label_shuffle_0p2_beta10_lg01
run_ggcl STL Image_Data/Train_Noise_Data/STL/hybrid_noise/gaussian_30p0_label_shuffle_0p2 1.0 0.15 beta10/hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_beta10_lg015

run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/feature_noise/blur_3p0 1.0 0.10 beta10/feature_noise/blur/blur_3p0_beta10_lg01
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 1.0 0.10 beta10/feature_noise/brightness/brightness_0p75_beta10_lg01
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 1.0 0.15 beta10/feature_noise/gaussian/gaussian_30p0_beta10_lg015
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 1.0 0.10 beta10/label_noise/label_shuffle/label_shuffle_0p2_beta10_lg01
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/hybrid_noise/blur_3p0_label_shuffle_0p2 1.0 0.10 beta10/hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_beta10_lg01
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/hybrid_noise/brightness_0p75_label_shuffle_0p2 1.0 0.10 beta10/hybrid_noise/brightness_label_shuffle/brightness_0p75_label_shuffle_0p2_beta10_lg01
run_ggcl Flower_102 Image_Data/Train_Noise_Data/Flower_102/hybrid_noise/gaussian_30p0_label_shuffle_0p2 1.0 0.10 beta10/hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_beta10_lg01

find "${RAW_RESULTS_ROOT}" -type d -empty -delete 2>/dev/null || true

echo
echo "Done. Supplemental ablation results are under: ${RESULTS_ROOT}"
