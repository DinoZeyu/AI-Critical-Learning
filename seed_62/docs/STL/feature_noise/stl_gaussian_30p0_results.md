# STL gaussian_30p0 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `gaussian_30p0`.

Protocol:

- Dataset: `STL`
- Noise setting: `feature_noise/gaussian_30p0`
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
| Noise lower reference | Baseline | `gaussian_30p0` train | 2 | 0.3823 | 0.3823 | 0.3569 | 0.0% |
| Best current method | Gold-guided CL, beta=0.50, lambda_gold=0.150 | `gaussian_30p0` train | 6 | 0.6581 | 0.6777 | 0.6777 | 88.7% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gaussian_30p0_method_beta05_lg015` | 0.50 | 0.150 | 6 | 0.6808 | 0.7516 | 0.6581 | 0.6777 | 9 | 0.6777 | 88.7% |
| `gaussian_30p0_beta02_lg01` | 0.20 | 0.100 | 12 | 0.6862 | 0.7613 | 0.6542 | 0.6781 | 13 | 0.6169 | 87.5% |
| `gaussian_30p0_beta09_lg01` | 0.90 | 0.100 | 5 | 0.6985 | 0.7645 | 0.6538 | 0.6538 | 5 | 0.6538 | 87.4% |
| `gaussian_30p0_beta05_lg02` | 0.50 | 0.200 | 6 | 0.6781 | 0.7419 | 0.6496 | 0.6727 | 9 | 0.6727 | 86.0% |
| `gaussian_30p0_beta05_lg0075` | 0.50 | 0.075 | 6 | 0.6964 | 0.7355 | 0.6477 | 0.6585 | 9 | 0.6585 | 85.4% |
| `gaussian_30p0_beta05_lg01` | 0.50 | 0.100 | 6 | 0.6936 | 0.7452 | 0.6435 | 0.6846 | 8 | 0.6600 | 84.0% |
| `gaussian_30p0_beta085_lg01` | 0.85 | 0.100 | 6 | 0.7064 | 0.7419 | 0.6412 | 0.6604 | 9 | 0.6604 | 83.3% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6581` from `gaussian_30p0_method_beta05_lg015`.
- Noise baseline selected test accuracy is `0.3823`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_62/Noise_Baseline/STL/feature_noise/gaussian_30p0/` |
| Best GGCL run | `seed_62/Experiments_Results/STL/feature_noise/gaussian/gaussian_30p0_method_beta05_lg015/` |
