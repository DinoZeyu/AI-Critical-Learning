# STL label_shuffle_0p2 Experiment Results

This file summarizes the current STL `label_shuffle_0p2` train-noise / clean-test results.

Protocol:

- Dataset: `STL`
- Noise setting: train label noise `label_shuffle_0p2`, clean test
- Label-noise strength: `20%` of train labels are randomly shuffled to another class
- Validation source: stratified 10% split from `Image_Data/Gold_Data/STL`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.6931`
- Label-noise baseline selected test accuracy: `0.6123`
- Recovery ratio: `(method_acc - label_noise_baseline_acc) / (clean_baseline_acc - label_noise_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `label_shuffle_0p2` train | 10 | 0.6123 | 0.6123 | 0.5946 | 0.0% |
| Previous GGCL best | Gold-guided CL, beta=0.50, lambda_gold=0.25 | `label_shuffle_0p2` train | 10 | 0.7173 | 0.7173 | 0.7038 | 130.0% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `label_shuffle_0p2_beta00_lg01` | 0.00 | 0.100 | 14 | 0.7454 | 0.7581 | 0.6673 | 0.6869 | 13 | 0.6527 | 68.1% |
| `label_shuffle_0p2_beta01_lg01` | 0.10 | 0.100 | 10 | 0.7564 | 0.7645 | 0.6877 | 0.6877 | 10 | 0.6738 | 93.3% |
| `label_shuffle_0p2_beta02_lg01` | 0.20 | 0.100 | 10 | 0.7540 | 0.7806 | 0.6838 | 0.6896 | 13 | 0.6896 | 88.6% |
| `label_shuffle_0p2_beta05_lg005` | 0.50 | 0.050 | 9 | 0.7347 | 0.7581 | 0.6642 | 0.6885 | 12 | 0.6885 | 64.3% |
| `label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 14 | 0.6617 | 0.8032 | 0.7069 | 0.7069 | 14 | 0.6877 | 117.1% |
| `label_shuffle_0p2_beta05_lg015` | 0.50 | 0.150 | 15 | 0.6352 | 0.7839 | 0.6919 | 0.7108 | 12 | 0.6781 | 98.6% |
| `label_shuffle_0p2_beta05_lg02` | 0.50 | 0.200 | 10 | 0.6421 | 0.7871 | 0.7108 | 0.7108 | 10 | 0.7031 | 121.9% |
| `label_shuffle_0p2_method_beta05_lg025` | 0.50 | 0.250 | 10 | 0.6038 | 0.8097 | 0.7173 | 0.7173 | 10 | 0.7038 | 130.0% |
| `label_shuffle_0p2_beta05_lg03` | 0.50 | 0.300 | 4 | 0.6852 | 0.7710 | 0.6608 | 0.6915 | 7 | 0.6915 | 60.0% |

## Notes

- Label noise is harmful under the plain baseline: selected test accuracy drops from clean `0.6931` to label-noise `0.6123`.
- Gold-guided CL more than closes this gap: the previous GGCL best run reaches `0.7173`, which is above the clean-train baseline in this STL setting.
- Within the original sweep, the best observed setting uses `beta=0.50`, which balances gold evaluator prediction confidence and feature-prototype similarity. Pure feature similarity (`beta=0.00`) is much weaker.
- A stronger gold stability anchor is beneficial for label noise. Increasing `lambda_gold` from `0.10` to `0.20` and `0.25` improves selected test accuracy, while `0.30` is too strong and hurts performance.
- The previous GGCL best setting is therefore `beta=0.50, lambda_gold=0.25`.
- A supplemental rerun-confirmed `beta=1.00, lambda_gold=0.25` ablation reaches `0.7177`, slightly exceeding this previous GGCL best; see `docs/ablation/supplemental_ablation_results.md`.
- The historical `Evaluators/STL/model.pt` path in some metrics refers to the same evaluator later stored under `Gold_Evaluators/STL/model.pt`.
- Recovery ratios above `100%` mean the method exceeds the clean-train baseline, not just recovers the label-noise damage.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/STL/` |
| Label-noise baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/STL/label_noise/label_shuffle_0p2/` |
| Gold-guided beta=0.50, lambda=0.25 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/label_noise/label_shuffle/label_shuffle_0p2_method_beta05_lg025/` |
| Supplemental beta=1.00, lambda_gold=0.25 ablation | `Supplemental_Ablation_Results/STL/overrun_ablation_rerun_round2/seed42/label_noise/label_shuffle/label_shuffle_0p2_beta10_lg025/` |
| Unified-path rerun | `Supplemental_Ablation_Results/STL/original_best_evaluator_rerun/seed42/label_noise/label_shuffle/label_shuffle_0p2_original_best_beta05_lg025/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/label_noise/label_shuffle/label_ablation/` |
