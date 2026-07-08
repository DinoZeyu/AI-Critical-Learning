# STL gaussian_30p0 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `gaussian_30p0`.

Protocol:

- Dataset: `STL`
- Noise setting: `feature_noise/gaussian_30p0`
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
| Noise lower reference | Baseline | `gaussian_30p0` train | 6 | 0.4688 | 0.4704 | 0.4565 | 0.0% |
| Best current method | Gold-guided CL, beta=0.50, lambda_gold=0.200 | `gaussian_30p0` train | 15 | 0.7077 | 0.7131 | 0.6635 | 106.5% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gaussian_30p0_beta05_lg02` | 0.50 | 0.200 | 15 | 0.6166 | 0.7839 | 0.7077 | 0.7131 | 16 | 0.6635 | 106.5% |
| `gaussian_30p0_beta05_lg01` | 0.50 | 0.100 | 7 | 0.6908 | 0.7645 | 0.6742 | 0.6781 | 10 | 0.6781 | 91.6% |
| `gaussian_30p0_beta02_lg01` | 0.20 | 0.100 | 8 | 0.6767 | 0.7548 | 0.6731 | 0.6731 | 8 | 0.6450 | 91.1% |
| `gaussian_30p0_beta05_lg0075` | 0.50 | 0.075 | 7 | 0.6725 | 0.7677 | 0.6715 | 0.6842 | 8 | 0.6781 | 90.4% |
| `gaussian_30p0_method_beta05_lg015` | 0.50 | 0.150 | 3 | 0.7517 | 0.7419 | 0.6608 | 0.6608 | 3 | 0.6481 | 85.6% |
| `gaussian_30p0_beta085_lg01` | 0.85 | 0.100 | 5 | 0.6922 | 0.7710 | 0.6473 | 0.6515 | 3 | 0.6454 | 79.6% |
| `gaussian_30p0_beta09_lg01` | 0.90 | 0.100 | 5 | 0.7046 | 0.7774 | 0.6315 | 0.6623 | 6 | 0.6577 | 72.6% |

## Notes

- Best selected test accuracy for this seed/setting is `0.7077` from `gaussian_30p0_beta05_lg02`.
- Noise baseline selected test accuracy is `0.4688`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_22/Noise_Baseline/STL/feature_noise/gaussian_30p0/` |
| Best GGCL run | `seed_22/Experiments_Results/STL/feature_noise/gaussian/gaussian_ablation/gaussian_30p0_beta05_lg02/` |
