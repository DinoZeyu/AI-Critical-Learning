# Flower_102 brightness_0p75 + label_shuffle_0p2 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `brightness_0p75 + label_shuffle_0p2`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `hybrid_noise/brightness_0p75_label_shuffle_0p2`
- Seed: `62`
- Validation source: stratified split from root `Image_Data/Gold_Data/<dataset>`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Gold evaluator path: `seed_62/Gold_Evaluators/Flower_102/model.pt`
- Headline metric: `test_accuracy_at_selected_epoch`; peak test accuracy is diagnostic.
- Recovery ratio: `(method_sel - noise_baseline_sel) / (clean_baseline_sel - noise_baseline_sel)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean upper reference | Baseline | Clean train | 20 | 0.5596 | 0.5693 | 0.5596 | 100.0% |
| Noise lower reference | Baseline | `brightness_0p75 + label_shuffle_0p2` train | 6 | 0.3299 | 0.3329 | 0.3238 | 0.0% |
| Best current method | Gold-guided CL, beta=0.75, lambda_gold=0.100 | `brightness_0p75 + label_shuffle_0p2` train | 14 | 0.5828 | 0.5828 | 0.5522 | 110.1% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brightness_0p75_label_shuffle_0p2_beta075_lg01` | 0.75 | 0.100 | 14 | 1.5552 | 0.5672 | 0.5828 | 0.5828 | 14 | 0.5522 | 110.1% |
| `brightness_0p75_label_shuffle_0p2_compromise_beta09_lg01` | 0.90 | 0.100 | 14 | 1.4858 | 0.5896 | 0.5828 | 0.5828 | 14 | 0.5669 | 110.1% |
| `brightness_0p75_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 13 | 1.4967 | 0.6194 | 0.5816 | 0.5877 | 14 | 0.5620 | 109.6% |
| `brightness_0p75_label_shuffle_0p2_beta085_lg01` | 0.85 | 0.100 | 14 | 1.5222 | 0.5970 | 0.5809 | 0.5834 | 15 | 0.5724 | 109.3% |
| `brightness_0p75_label_shuffle_0p2_beta08_lg005` | 0.80 | 0.050 | 14 | 1.5687 | 0.5821 | 0.5803 | 0.5803 | 14 | 0.5644 | 109.0% |
| `brightness_0p75_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 10 | 1.5268 | 0.5597 | 0.5767 | 0.5779 | 13 | 0.5779 | 107.4% |
| `brightness_0p75_label_shuffle_0p2_beta078_lg01` | 0.78 | 0.100 | 14 | 1.5050 | 0.5896 | 0.5730 | 0.5730 | 14 | 0.5504 | 105.9% |
| `brightness_0p75_label_shuffle_0p2_feature_best_beta08_lg01` | 0.80 | 0.100 | 14 | 1.5154 | 0.6194 | 0.5724 | 0.5724 | 14 | 0.5516 | 105.6% |
| `brightness_0p75_label_shuffle_0p2_beta07_lg01` | 0.70 | 0.100 | 12 | 1.5700 | 0.5448 | 0.5712 | 0.5925 | 14 | 0.5596 | 105.1% |

## Notes

- Best selected test accuracy for this seed/setting is `0.5828` from `brightness_0p75_label_shuffle_0p2_beta075_lg01`.
- Noise baseline selected test accuracy is `0.3299`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_62/Noise_Baseline/Flower_102/hybrid_noise/brightness_0p75_label_shuffle_0p2/` |
| Best GGCL run | `seed_62/Experiments_Results/Flower_102/hybrid_noise/brightness_label_shuffle/brightness_hybrid_ablation/brightness_0p75_label_shuffle_0p2_beta075_lg01/` |
