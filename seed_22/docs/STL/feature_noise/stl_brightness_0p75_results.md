# STL brightness_0p75 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `brightness_0p75`.

Protocol:

- Dataset: `STL`
- Noise setting: `feature_noise/brightness_0p75`
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
| Noise lower reference | Baseline | `brightness_0p75` train | 16 | 0.6904 | 0.6904 | 0.6365 | 0.0% |
| Best current method | Gold-guided CL, beta=0.20, lambda_gold=0.100 | `brightness_0p75` train | 9 | 0.7262 | 0.7338 | 0.7169 | 1328.6% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brightness_0p75_beta02_lg01` | 0.20 | 0.100 | 9 | 0.5158 | 0.8290 | 0.7262 | 0.7338 | 10 | 0.7169 | 1328.6% |
| `brightness_0p75_beta085_lg01` | 0.85 | 0.100 | 9 | 0.4808 | 0.8097 | 0.7258 | 0.7427 | 10 | 0.7100 | 1314.3% |
| `brightness_0p75_beta05_lg01` | 0.50 | 0.100 | 9 | 0.5086 | 0.7968 | 0.7231 | 0.7415 | 10 | 0.6923 | 1214.3% |
| `brightness_0p75_method_beta09_lg01` | 0.90 | 0.100 | 9 | 0.4738 | 0.8290 | 0.7227 | 0.7335 | 11 | 0.7065 | 1200.0% |
| `brightness_0p75_beta09_lg0075` | 0.90 | 0.075 | 9 | 0.4866 | 0.8194 | 0.7162 | 0.7258 | 11 | 0.7181 | 957.1% |
| `brightness_0p75_beta09_lg015` | 0.90 | 0.150 | 5 | 0.5141 | 0.8161 | 0.6985 | 0.7015 | 8 | 0.7015 | 300.0% |

## Notes

- Best selected test accuracy for this seed/setting is `0.7262` from `brightness_0p75_beta02_lg01`.
- Noise baseline selected test accuracy is `0.6904`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_22/Noise_Baseline/STL/feature_noise/brightness_0p75/` |
| Best GGCL run | `seed_22/Experiments_Results/STL/feature_noise/brightness/brightness_ablation/brightness_0p75_beta02_lg01/` |
