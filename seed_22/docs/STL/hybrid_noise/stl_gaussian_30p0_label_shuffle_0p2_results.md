# STL gaussian_30p0 + label_shuffle_0p2 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `gaussian_30p0 + label_shuffle_0p2`.

Protocol:

- Dataset: `STL`
- Noise setting: `hybrid_noise/gaussian_30p0_label_shuffle_0p2`
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
| Noise lower reference | Baseline | `gaussian_30p0 + label_shuffle_0p2` train | 13 | 0.5112 | 0.5112 | 0.4400 | 0.0% |
| Best current method | Gold-guided CL, beta=0.70, lambda_gold=0.150 | `gaussian_30p0 + label_shuffle_0p2` train | 13 | 0.6931 | 0.6931 | 0.6631 | 100.0% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gaussian_30p0_label_shuffle_0p2_beta07_lg015` | 0.70 | 0.150 | 13 | 0.5980 | 0.7806 | 0.6931 | 0.6931 | 13 | 0.6631 | 100.0% |
| `gaussian_30p0_label_shuffle_0p2_beta06_lg015` | 0.60 | 0.150 | 13 | 0.6453 | 0.7613 | 0.6877 | 0.6877 | 13 | 0.6835 | 97.0% |
| `gaussian_30p0_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 11 | 0.6598 | 0.7871 | 0.6804 | 0.6804 | 11 | 0.6304 | 93.0% |
| `gaussian_30p0_label_shuffle_0p2_compromise_beta05_lg02` | 0.50 | 0.200 | 13 | 0.6859 | 0.7677 | 0.6735 | 0.6735 | 13 | 0.6504 | 89.2% |
| `gaussian_30p0_label_shuffle_0p2_feature_best_beta05_lg015` | 0.50 | 0.150 | 2 | 0.8042 | 0.7581 | 0.6465 | 0.6465 | 2 | 0.5488 | 74.4% |
| `gaussian_30p0_label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 9 | 0.7365 | 0.7581 | 0.6442 | 0.6654 | 8 | 0.6646 | 73.2% |
| `gaussian_30p0_label_shuffle_0p2_beta04_lg015` | 0.40 | 0.150 | 9 | 0.7320 | 0.7452 | 0.6392 | 0.6615 | 12 | 0.6615 | 70.4% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6931` from `gaussian_30p0_label_shuffle_0p2_beta07_lg015`.
- Noise baseline selected test accuracy is `0.5112`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_22/Noise_Baseline/STL/hybrid_noise/gaussian_30p0_label_shuffle_0p2/` |
| Best GGCL run | `seed_22/Experiments_Results/STL/hybrid_noise/gaussian_label_shuffle/gaussian_hybrid_ablation/gaussian_30p0_label_shuffle_0p2_beta07_lg015/` |
