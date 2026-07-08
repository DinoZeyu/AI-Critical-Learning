# STL brightness_0p75 + label_shuffle_0p2 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `brightness_0p75 + label_shuffle_0p2`.

Protocol:

- Dataset: `STL`
- Noise setting: `hybrid_noise/brightness_0p75_label_shuffle_0p2`
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
| Noise lower reference | Baseline | `brightness_0p75 + label_shuffle_0p2` train | 13 | 0.6254 | 0.6254 | 0.5869 | 0.0% |
| Best current method | Gold-guided CL, beta=0.90, lambda_gold=0.150 | `brightness_0p75 + label_shuffle_0p2` train | 13 | 0.7146 | 0.7146 | 0.6954 | 131.8% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brightness_0p75_label_shuffle_0p2_compromise_beta09_lg015` | 0.90 | 0.150 | 13 | 0.5596 | 0.8323 | 0.7146 | 0.7146 | 13 | 0.6954 | 131.8% |
| `brightness_0p75_label_shuffle_0p2_beta09_lg0075` | 0.90 | 0.075 | 17 | 0.5936 | 0.7968 | 0.7050 | 0.7119 | 19 | 0.6812 | 117.6% |
| `brightness_0p75_label_shuffle_0p2_beta085_lg01` | 0.85 | 0.100 | 6 | 0.6762 | 0.7581 | 0.6862 | 0.6862 | 6 | 0.6827 | 89.8% |
| `brightness_0p75_label_shuffle_0p2_feature_best_beta09_lg01` | 0.90 | 0.100 | 8 | 0.6791 | 0.7452 | 0.6846 | 0.6927 | 9 | 0.6888 | 87.5% |
| `brightness_0p75_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 8 | 0.6681 | 0.7774 | 0.6835 | 0.6942 | 9 | 0.6862 | 85.8% |
| `brightness_0p75_label_shuffle_0p2_beta09_lg005` | 0.90 | 0.050 | 2 | 0.7810 | 0.7387 | 0.6615 | 0.6615 | 2 | 0.6404 | 53.4% |
| `brightness_0p75_label_shuffle_0p2_beta095_lg01` | 0.95 | 0.100 | 2 | 0.7727 | 0.7484 | 0.6515 | 0.6631 | 5 | 0.6631 | 38.6% |

## Notes

- Best selected test accuracy for this seed/setting is `0.7146` from `brightness_0p75_label_shuffle_0p2_compromise_beta09_lg015`.
- Noise baseline selected test accuracy is `0.6254`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_62/Noise_Baseline/STL/hybrid_noise/brightness_0p75_label_shuffle_0p2/` |
| Best GGCL run | `seed_62/Experiments_Results/STL/hybrid_noise/brightness_label_shuffle/brightness_hybrid_ablation/brightness_0p75_label_shuffle_0p2_compromise_beta09_lg015/` |
