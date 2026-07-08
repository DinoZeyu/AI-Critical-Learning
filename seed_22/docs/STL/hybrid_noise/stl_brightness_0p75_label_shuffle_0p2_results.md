# STL brightness_0p75 + label_shuffle_0p2 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `brightness_0p75 + label_shuffle_0p2`.

Protocol:

- Dataset: `STL`
- Noise setting: `hybrid_noise/brightness_0p75_label_shuffle_0p2`
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
| Noise lower reference | Baseline | `brightness_0p75 + label_shuffle_0p2` train | 13 | 0.5869 | 0.5985 | 0.5754 | 0.0% |
| Best current method | Gold-guided CL, beta=0.90, lambda_gold=0.100 | `brightness_0p75 + label_shuffle_0p2` train | 9 | 0.7212 | 0.7212 | 0.7185 | 126.4% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brightness_0p75_label_shuffle_0p2_feature_best_beta09_lg01` | 0.90 | 0.100 | 9 | 0.5569 | 0.8065 | 0.7212 | 0.7212 | 9 | 0.7185 | 126.4% |
| `brightness_0p75_label_shuffle_0p2_beta09_lg005` | 0.90 | 0.050 | 12 | 0.5898 | 0.8065 | 0.7104 | 0.7169 | 13 | 0.6969 | 116.3% |
| `brightness_0p75_label_shuffle_0p2_beta085_lg01` | 0.85 | 0.100 | 9 | 0.5719 | 0.8000 | 0.7081 | 0.7196 | 12 | 0.7196 | 114.1% |
| `brightness_0p75_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 5 | 0.6107 | 0.8129 | 0.7035 | 0.7035 | 5 | 0.6946 | 109.8% |
| `brightness_0p75_label_shuffle_0p2_compromise_beta09_lg015` | 0.90 | 0.150 | 5 | 0.5885 | 0.8000 | 0.6904 | 0.6988 | 8 | 0.6988 | 97.5% |
| `brightness_0p75_label_shuffle_0p2_beta095_lg01` | 0.95 | 0.100 | 5 | 0.5990 | 0.7871 | 0.6892 | 0.7015 | 7 | 0.6950 | 96.4% |
| `brightness_0p75_label_shuffle_0p2_beta09_lg0075` | 0.90 | 0.075 | 1 | 0.7263 | 0.7548 | 0.6331 | 0.6658 | 3 | 0.6535 | 43.5% |

## Notes

- Best selected test accuracy for this seed/setting is `0.7212` from `brightness_0p75_label_shuffle_0p2_feature_best_beta09_lg01`.
- Noise baseline selected test accuracy is `0.5869`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_22/Noise_Baseline/STL/hybrid_noise/brightness_0p75_label_shuffle_0p2/` |
| Best GGCL run | `seed_22/Experiments_Results/STL/hybrid_noise/brightness_label_shuffle/brightness_hybrid_ablation/brightness_0p75_label_shuffle_0p2_feature_best_beta09_lg01/` |
