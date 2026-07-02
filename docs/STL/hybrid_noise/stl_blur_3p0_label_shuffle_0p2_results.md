# STL blur_3p0 + label_shuffle_0p2 Hybrid Experiment Results

This file summarizes the current STL `blur_3p0 + label_shuffle_0p2` train-noise / clean-test results.

Protocol:

- Dataset: `STL`
- Noise setting: train hybrid noise `blur_3p0 + label_shuffle_0p2`, clean test
- Validation source: stratified 10% split from `Image_Data/Gold_Data/STL`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.6931`
- Hybrid baseline selected test accuracy: `0.2396`
- Recovery ratio: `(method_acc - hybrid_baseline_acc) / (clean_baseline_acc - hybrid_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `blur_3p0 + label_shuffle_0p2` train | 2 | 0.2396 | 0.2804 | 0.2804 | 0.0% |
| Best current method | Gold-guided CL, beta=0.90, lambda_gold=0.25 | `blur_3p0 + label_shuffle_0p2` train | 4 | 0.6500 | 0.6500 | 0.6331 | 90.5% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `blur_3p0_label_shuffle_0p2_beta09_lg025` | 0.90 | 0.250 | 4 | 0.7712 | 0.7419 | 0.6500 | 0.6500 | 4 | 0.6331 | 90.5% |
| `blur_3p0_label_shuffle_0p2_beta09_lg03` | 0.90 | 0.300 | 4 | 0.8160 | 0.7419 | 0.6454 | 0.6454 | 4 | 0.6438 | 89.5% |
| `blur_3p0_label_shuffle_0p2_beta095_lg015` | 0.95 | 0.150 | 4 | 0.8943 | 0.7161 | 0.6396 | 0.6396 | 4 | 0.6215 | 88.2% |
| `blur_3p0_label_shuffle_0p2_feature_best_beta09_lg015` | 0.90 | 0.150 | 4 | 0.8347 | 0.7419 | 0.6392 | 0.6392 | 4 | 0.6077 | 88.1% |
| `blur_3p0_label_shuffle_0p2_beta085_lg015` | 0.85 | 0.150 | 5 | 0.8364 | 0.7161 | 0.6365 | 0.6365 | 5 | 0.5719 | 87.5% |
| `blur_3p0_label_shuffle_0p2_compromise_beta09_lg02` | 0.90 | 0.200 | 4 | 0.8446 | 0.7258 | 0.6319 | 0.6319 | 4 | 0.6312 | 86.5% |
| `blur_3p0_label_shuffle_0p2_beta09_lg01` | 0.90 | 0.100 | 4 | 0.8705 | 0.7161 | 0.6123 | 0.6123 | 4 | 0.5754 | 82.2% |
| `blur_3p0_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 4 | 0.8920 | 0.7097 | 0.6054 | 0.6054 | 4 | 0.5988 | 80.7% |

## Notes

- The hybrid baseline is much weaker than either single label noise or single blur noise, dropping to `0.2396` selected test accuracy.
- Gold-guided CL recovers most of the gap: the best selected accuracy is `0.6500`, recovering `90.5%` of the clean-vs-hybrid gap.
- The best setting keeps high `beta=0.90`, matching the single blur result where feature similarity is unreliable under strong blur.
- Compared with single blur, the hybrid setting benefits from a stronger gold stability anchor: `lambda_gold=0.25` beats `0.15`, `0.20`, and `0.30`.
- The label-noise best setting `beta=0.50, lambda_gold=0.25` is substantially weaker here, suggesting that blur corruption still dominates the controller design.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/STL/` |
| Hybrid baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/STL/hybrid_noise/blur_3p0_label_shuffle_0p2/` |
| Gold-guided beta=0.90, lambda_gold=0.25 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_beta09_lg025/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/hybrid_noise/blur_label_shuffle/blur_hybrid_ablation/` |
