# STL brightness_0p75 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `brightness_0p75`.

Protocol:

- Dataset: `STL`
- Noise setting: `feature_noise/brightness_0p75`
- Seed: `62`
- Validation source: stratified split from root `Image_Data/Gold_Data/<dataset>`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Gold evaluator path: `seed_62/Gold_Evaluators/STL/model.pt`
- Headline metric: `test_accuracy_at_selected_epoch`; peak test accuracy is diagnostic.
- Recovery ratio: `(method_sel - noise_baseline_sel) / (clean_baseline_sel - noise_baseline_sel)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `brightness_0p75` train | 11 | 0.6731 | 0.6731 | 0.6485 | 0.0% |
| Best current method | Gold-guided CL, beta=0.90, lambda_gold=0.075 | `brightness_0p75` train | 16 | 0.7408 | 0.7431 | 0.7431 | 338.5% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brightness_0p75_beta09_lg0075` | 0.90 | 0.075 | 16 | 0.5500 | 0.8194 | 0.7408 | 0.7431 | 19 | 0.7431 | 338.5% |
| `brightness_0p75_beta05_lg01` | 0.50 | 0.100 | 16 | 0.5474 | 0.8161 | 0.7331 | 0.7515 | 19 | 0.7515 | 300.0% |
| `brightness_0p75_method_beta09_lg01` | 0.90 | 0.100 | 13 | 0.5167 | 0.8226 | 0.7296 | 0.7485 | 16 | 0.7485 | 282.7% |
| `brightness_0p75_beta085_lg01` | 0.85 | 0.100 | 6 | 0.5996 | 0.7839 | 0.7123 | 0.7212 | 9 | 0.7212 | 196.2% |
| `brightness_0p75_beta09_lg015` | 0.90 | 0.150 | 7 | 0.6137 | 0.7839 | 0.7073 | 0.7200 | 9 | 0.6827 | 171.2% |
| `brightness_0p75_beta02_lg01` | 0.20 | 0.100 | 7 | 0.6196 | 0.7806 | 0.6981 | 0.7081 | 9 | 0.6777 | 125.0% |

## Notes

- Best selected test accuracy for this seed/setting is `0.7408` from `brightness_0p75_beta09_lg0075`.
- Noise baseline selected test accuracy is `0.6731`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_62/Noise_Baseline/STL/feature_noise/brightness_0p75/` |
| Best GGCL run | `seed_62/Experiments_Results/STL/feature_noise/brightness/brightness_ablation/brightness_0p75_beta09_lg0075/` |
