# STL gaussian_30p0 + label_shuffle_0p2 Hybrid Experiment Results

This file summarizes the current STL `gaussian_30p0 + label_shuffle_0p2` train-noise / clean-test results.

Protocol:

- Dataset: `STL`
- Noise setting: train hybrid noise `gaussian_30p0 + label_shuffle_0p2`, clean test
- Validation source: stratified 10% split from `Image_Data/Gold_Data/STL`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.6931`
- Hybrid baseline selected test accuracy: `0.4069`
- Recovery ratio: `(method_acc - hybrid_baseline_acc) / (clean_baseline_acc - hybrid_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `gaussian_30p0 + label_shuffle_0p2` train | 4 | 0.4069 | 0.4069 | 0.3400 | 0.0% |
| Best current method | Gold-guided CL, beta=0.60, lambda_gold=0.15 | `gaussian_30p0 + label_shuffle_0p2` train | 13 | 0.6604 | 0.6723 | 0.6319 | 88.6% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gaussian_30p0_label_shuffle_0p2_beta06_lg015` | 0.60 | 0.150 | 13 | 0.7095 | 0.7806 | 0.6604 | 0.6723 | 11 | 0.6319 | 88.6% |
| `gaussian_30p0_label_shuffle_0p2_beta07_lg015` | 0.70 | 0.150 | 5 | 0.7680 | 0.7613 | 0.6427 | 0.6427 | 5 | 0.6162 | 82.4% |
| `gaussian_30p0_label_shuffle_0p2_feature_best_beta05_lg015` | 0.50 | 0.150 | 9 | 0.7452 | 0.7645 | 0.6408 | 0.6731 | 11 | 0.6442 | 81.7% |
| `gaussian_30p0_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 9 | 0.7434 | 0.7613 | 0.6350 | 0.6696 | 11 | 0.6338 | 79.7% |
| `gaussian_30p0_label_shuffle_0p2_compromise_beta05_lg02` | 0.50 | 0.200 | 5 | 0.7832 | 0.7323 | 0.6338 | 0.6465 | 6 | 0.6038 | 79.3% |
| `gaussian_30p0_label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 9 | 0.7628 | 0.7387 | 0.6285 | 0.6723 | 11 | 0.6558 | 77.4% |
| `gaussian_30p0_label_shuffle_0p2_beta04_lg015` | 0.40 | 0.150 | 5 | 0.8036 | 0.7581 | 0.6262 | 0.6535 | 6 | 0.5846 | 76.6% |

## Notes

- Gaussian hybrid is harder than single Gaussian and single label noise under the baseline: selected test accuracy is `0.4069`.
- Gold-guided CL improves the selected test accuracy to `0.6604`, recovering `88.6%` of the clean-vs-hybrid gap.
- The best selected checkpoint uses `beta=0.60`, slightly higher than the single Gaussian best `beta=0.50`. This suggests that adding label noise makes the gold evaluator prediction signal more important, while still preserving some feature-prototype similarity.
- The best gold stability weight remains `lambda_gold=0.15`; stronger anchors at `0.20` and `0.25` do not improve selected accuracy.
- The diagnostic peak for `beta=0.50, lambda_gold=0.15` is slightly higher (`0.6731`) than the best selected run's peak (`0.6723`), but headline comparison follows validation-selected accuracy.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/STL/` |
| Hybrid baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/STL/hybrid_noise/gaussian_30p0_label_shuffle_0p2/` |
| Gold-guided beta=0.60, lambda_gold=0.15 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_beta06_lg015/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/hybrid_noise/gaussian_label_shuffle/gaussian_hybrid_ablation/` |
