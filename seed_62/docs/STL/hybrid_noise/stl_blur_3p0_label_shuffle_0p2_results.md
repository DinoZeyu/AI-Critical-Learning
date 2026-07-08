# STL blur_3p0 + label_shuffle_0p2 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `blur_3p0 + label_shuffle_0p2`.

Protocol:

- Dataset: `STL`
- Noise setting: `hybrid_noise/blur_3p0_label_shuffle_0p2`
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
| Noise lower reference | Baseline | `blur_3p0 + label_shuffle_0p2` train | 1 | 0.2496 | 0.2496 | 0.2223 | 0.0% |
| Best current method | Gold-guided CL, beta=0.90, lambda_gold=0.200 | `blur_3p0 + label_shuffle_0p2` train | 4 | 0.6412 | 0.6412 | 0.6035 | 88.3% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blur_3p0_label_shuffle_0p2_compromise_beta09_lg02` | 0.90 | 0.200 | 4 | 0.8193 | 0.7194 | 0.6412 | 0.6412 | 4 | 0.6035 | 88.3% |
| `blur_3p0_label_shuffle_0p2_beta085_lg015` | 0.85 | 0.150 | 4 | 0.7939 | 0.7355 | 0.6350 | 0.6350 | 4 | 0.5927 | 86.9% |
| `blur_3p0_label_shuffle_0p2_beta09_lg025` | 0.90 | 0.250 | 2 | 0.7855 | 0.7194 | 0.6327 | 0.6327 | 2 | 0.6223 | 86.4% |
| `blur_3p0_label_shuffle_0p2_beta095_lg015` | 0.95 | 0.150 | 4 | 0.8468 | 0.7065 | 0.6319 | 0.6319 | 4 | 0.6092 | 86.2% |
| `blur_3p0_label_shuffle_0p2_beta09_lg01` | 0.90 | 0.100 | 4 | 0.8148 | 0.7194 | 0.6296 | 0.6296 | 4 | 0.5769 | 85.7% |
| `blur_3p0_label_shuffle_0p2_beta09_lg03` | 0.90 | 0.300 | 2 | 0.8098 | 0.7129 | 0.6285 | 0.6346 | 4 | 0.6058 | 85.4% |
| `blur_3p0_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 2 | 0.8266 | 0.6968 | 0.6119 | 0.6246 | 4 | 0.5862 | 81.7% |
| `blur_3p0_label_shuffle_0p2_feature_best_beta09_lg015` | 0.90 | 0.150 | 2 | 0.8370 | 0.7065 | 0.6065 | 0.6181 | 4 | 0.5931 | 80.5% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6412` from `blur_3p0_label_shuffle_0p2_compromise_beta09_lg02`.
- Noise baseline selected test accuracy is `0.2496`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_62/Noise_Baseline/STL/hybrid_noise/blur_3p0_label_shuffle_0p2/` |
| Best GGCL run | `seed_62/Experiments_Results/STL/hybrid_noise/blur_label_shuffle/blur_hybrid_ablation/blur_3p0_label_shuffle_0p2_compromise_beta09_lg02/` |
