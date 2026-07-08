# STL gaussian_30p0 + label_shuffle_0p2 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `gaussian_30p0 + label_shuffle_0p2`.

Protocol:

- Dataset: `STL`
- Noise setting: `hybrid_noise/gaussian_30p0_label_shuffle_0p2`
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
| Noise lower reference | Baseline | `gaussian_30p0 + label_shuffle_0p2` train | 6 | 0.4150 | 0.4550 | 0.4550 | 0.0% |
| Best current method | Gold-guided CL, beta=0.70, lambda_gold=0.150 | `gaussian_30p0 + label_shuffle_0p2` train | 10 | 0.6658 | 0.6692 | 0.6692 | 90.2% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gaussian_30p0_label_shuffle_0p2_beta07_lg015` | 0.70 | 0.150 | 10 | 0.6656 | 0.7645 | 0.6658 | 0.6692 | 13 | 0.6692 | 90.2% |
| `gaussian_30p0_label_shuffle_0p2_compromise_beta05_lg02` | 0.50 | 0.200 | 10 | 0.7038 | 0.7516 | 0.6565 | 0.6573 | 13 | 0.6573 | 86.9% |
| `gaussian_30p0_label_shuffle_0p2_beta06_lg015` | 0.60 | 0.150 | 12 | 0.6960 | 0.7516 | 0.6531 | 0.6638 | 13 | 0.6458 | 85.6% |
| `gaussian_30p0_label_shuffle_0p2_feature_best_beta05_lg015` | 0.50 | 0.150 | 17 | 0.6589 | 0.7710 | 0.6523 | 0.6581 | 19 | 0.6035 | 85.3% |
| `gaussian_30p0_label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 12 | 0.7716 | 0.7355 | 0.6415 | 0.6531 | 10 | 0.6358 | 81.5% |
| `gaussian_30p0_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 6 | 0.7598 | 0.7484 | 0.6385 | 0.6396 | 7 | 0.6208 | 80.4% |
| `gaussian_30p0_label_shuffle_0p2_beta04_lg015` | 0.40 | 0.150 | 12 | 0.6999 | 0.7484 | 0.6362 | 0.6608 | 13 | 0.6165 | 79.5% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6658` from `gaussian_30p0_label_shuffle_0p2_beta07_lg015`.
- Noise baseline selected test accuracy is `0.4150`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_62/Noise_Baseline/STL/hybrid_noise/gaussian_30p0_label_shuffle_0p2/` |
| Best GGCL run | `seed_62/Experiments_Results/STL/hybrid_noise/gaussian_label_shuffle/gaussian_hybrid_ablation/gaussian_30p0_label_shuffle_0p2_beta07_lg015/` |
