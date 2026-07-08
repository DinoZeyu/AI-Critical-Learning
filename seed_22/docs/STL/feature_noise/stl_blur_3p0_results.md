# STL blur_3p0 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `blur_3p0`.

Protocol:

- Dataset: `STL`
- Noise setting: `feature_noise/blur_3p0`
- Seed: `22`
- Validation source: stratified split from root `Image_Data/Gold_Data/<dataset>`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Gold evaluator path: `seed_22/Gold_Evaluators/STL/model.pt`
- Headline metric: `test_accuracy_at_selected_epoch`; peak test accuracy is diagnostic.
- Recovery ratio: `(method_sel - noise_baseline_sel) / (clean_baseline_sel - noise_baseline_sel)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `blur_3p0` train | 1 | 0.2546 | 0.2669 | 0.2496 | 0.0% |
| Best current method | Gold-guided CL, beta=0.90, lambda_gold=0.075 | `blur_3p0` train | 5 | 0.6531 | 0.6531 | 0.6531 | 90.9% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blur_3p0_beta09_lg0075` | 0.90 | 0.075 | 5 | 0.6216 | 0.7839 | 0.6531 | 0.6531 | 5 | 0.6531 | 90.9% |
| `blur_3p0_beta09_lg01` | 0.90 | 0.100 | 5 | 0.7424 | 0.7548 | 0.6427 | 0.6458 | 8 | 0.6458 | 88.5% |
| `blur_3p0_beta05_lg01` | 0.50 | 0.100 | 5 | 0.7303 | 0.7645 | 0.6354 | 0.6446 | 8 | 0.6446 | 86.8% |
| `blur_3p0_method_beta09_lg015` | 0.90 | 0.150 | 1 | 0.7134 | 0.7452 | 0.6288 | 0.6288 | 1 | 0.6062 | 85.4% |
| `blur_3p0_beta085_lg01` | 0.85 | 0.100 | 1 | 0.7641 | 0.7258 | 0.6119 | 0.6169 | 3 | 0.5962 | 81.5% |
| `blur_3p0_beta02_lg01` | 0.20 | 0.100 | 1 | 0.9113 | 0.6548 | 0.5804 | 0.5804 | 1 | 0.4954 | 74.3% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6531` from `blur_3p0_beta09_lg0075`.
- Noise baseline selected test accuracy is `0.2546`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_22/Noise_Baseline/STL/feature_noise/blur_3p0/` |
| Best GGCL run | `seed_22/Experiments_Results/STL/feature_noise/blur/blur_ablations/blur_3p0_beta09_lg0075/` |
