# Flower_102 brightness_0p75 + label_shuffle_0p2 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `brightness_0p75 + label_shuffle_0p2`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `hybrid_noise/brightness_0p75_label_shuffle_0p2`
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
| Noise lower reference | Baseline | `brightness_0p75 + label_shuffle_0p2` train | 11 | 0.3659 | 0.3659 | 0.3641 | 0.0% |
| Best current method | Gold-guided CL, beta=0.80, lambda_gold=0.150 | `brightness_0p75 + label_shuffle_0p2` train | 12 | 0.5602 | 0.5602 | 0.5382 | 100.3% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brightness_0p75_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 12 | 1.7826 | 0.5373 | 0.5602 | 0.5602 | 12 | 0.5382 | 100.3% |
| `brightness_0p75_label_shuffle_0p2_beta078_lg01` | 0.78 | 0.100 | 14 | 1.7338 | 0.5597 | 0.5418 | 0.5516 | 12 | 0.5443 | 90.9% |
| `brightness_0p75_label_shuffle_0p2_compromise_beta09_lg01` | 0.90 | 0.100 | 9 | 1.7547 | 0.5746 | 0.5351 | 0.5455 | 12 | 0.5455 | 87.4% |
| `brightness_0p75_label_shuffle_0p2_feature_best_beta08_lg01` | 0.80 | 0.100 | 9 | 1.7945 | 0.5448 | 0.5296 | 0.5388 | 12 | 0.5388 | 84.5% |
| `brightness_0p75_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 9 | 1.7374 | 0.5373 | 0.5296 | 0.5461 | 12 | 0.5461 | 84.5% |
| `brightness_0p75_label_shuffle_0p2_beta08_lg005` | 0.80 | 0.050 | 11 | 1.8500 | 0.5448 | 0.5192 | 0.5376 | 14 | 0.5376 | 79.2% |
| `brightness_0p75_label_shuffle_0p2_beta085_lg01` | 0.85 | 0.100 | 8 | 1.7748 | 0.5224 | 0.5180 | 0.5247 | 9 | 0.5235 | 78.5% |
| `brightness_0p75_label_shuffle_0p2_beta075_lg01` | 0.75 | 0.100 | 8 | 1.8383 | 0.5224 | 0.5168 | 0.5223 | 9 | 0.5052 | 77.9% |
| `brightness_0p75_label_shuffle_0p2_beta07_lg01` | 0.70 | 0.100 | 7 | 1.8468 | 0.5373 | 0.5027 | 0.5186 | 9 | 0.5119 | 70.7% |

## Notes

- Best selected test accuracy for this seed/setting is `0.5602` from `brightness_0p75_label_shuffle_0p2_beta08_lg015`.
- Noise baseline selected test accuracy is `0.3659`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_22/Noise_Baseline/Flower_102/hybrid_noise/brightness_0p75_label_shuffle_0p2/` |
| Best GGCL run | `seed_22/Experiments_Results/Flower_102/hybrid_noise/brightness_label_shuffle/brightness_hybrid_ablation/brightness_0p75_label_shuffle_0p2_beta08_lg015/` |
