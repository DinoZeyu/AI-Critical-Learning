# Flower_102 blur_3p0 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `blur_3p0`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `feature_noise/blur_3p0`
- Seed: `22`
- Validation source: stratified split from root `Image_Data/Gold_Data/<dataset>`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Gold evaluator path: `seed_22/Gold_Evaluators/Flower_102/model.pt`
- Headline metric: `test_accuracy_at_selected_epoch`; peak test accuracy is diagnostic.
- Recovery ratio: `(method_sel - noise_baseline_sel) / (clean_baseline_sel - noise_baseline_sel)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean upper reference | Baseline | Clean train | 20 | 0.5596 | 0.5693 | 0.5596 | 100.0% |
| Noise lower reference | Baseline | `blur_3p0` train | 10 | 0.4301 | 0.4301 | 0.3842 | 0.0% |
| Best current method | Gold-guided CL, beta=0.80, lambda_gold=0.200 | `blur_3p0` train | 9 | 0.5602 | 0.5779 | 0.5779 | 100.5% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blur_3p0_beta08_lg02` | 0.80 | 0.200 | 9 | 1.6705 | 0.5672 | 0.5602 | 0.5779 | 12 | 0.5779 | 100.5% |
| `blur_3p0_beta05_lg01` | 0.50 | 0.100 | 4 | 1.7738 | 0.5597 | 0.4966 | 0.4966 | 4 | 0.4838 | 51.4% |
| `blur_3p0_method_beta08_lg01` | 0.80 | 0.100 | 4 | 1.7571 | 0.5746 | 0.4960 | 0.5180 | 7 | 0.5180 | 50.9% |
| `blur_3p0_beta08_lg015` | 0.80 | 0.150 | 4 | 1.7786 | 0.5373 | 0.4960 | 0.5363 | 7 | 0.5363 | 50.9% |
| `blur_3p0_beta02_lg01` | 0.20 | 0.100 | 4 | 1.7687 | 0.5373 | 0.4954 | 0.5046 | 6 | 0.4649 | 50.5% |
| `blur_3p0_beta09_lg01` | 0.90 | 0.100 | 4 | 1.7468 | 0.5821 | 0.4942 | 0.5333 | 7 | 0.5333 | 49.5% |
| `blur_3p0_beta085_lg01` | 0.85 | 0.100 | 4 | 1.7887 | 0.5522 | 0.4918 | 0.5247 | 7 | 0.5247 | 47.6% |
| `blur_3p0_beta08_lg005` | 0.80 | 0.050 | 4 | 1.7963 | 0.5597 | 0.4844 | 0.4869 | 5 | 0.4801 | 42.0% |

## Notes

- Best selected test accuracy for this seed/setting is `0.5602` from `blur_3p0_beta08_lg02`.
- Noise baseline selected test accuracy is `0.4301`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_22/Noise_Baseline/Flower_102/feature_noise/blur_3p0/` |
| Best GGCL run | `seed_22/Experiments_Results/Flower_102/feature_noise/blur/blur_ablations/blur_3p0_beta08_lg02/` |
