# STL gaussian_30p0 Experiment Results

This file summarizes the current STL `gaussian_30p0` train-noise / clean-test results.

Protocol:

- Dataset: `STL`
- Noise setting: train feature noise `gaussian_30p0`, clean test
- Validation source: stratified 10% split from `Image_Data/Gold_Data/STL`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.6931`
- Gaussian baseline selected test accuracy: `0.4669`
- Recovery ratio: `(method_acc - gaussian_baseline_acc) / (clean_baseline_acc - gaussian_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `gaussian_30p0` train | 8 | 0.4669 | 0.4892 | 0.4892 | 0.0% |
| Best current method | Gold-guided CL, beta=0.50, lambda_gold=0.15 | `gaussian_30p0` train | 11 | 0.6869 | 0.6869 | 0.6692 | 97.3% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gaussian_30p0_beta02_lg01` | 0.20 | 0.100 | 9 | 0.6965 | 0.7323 | 0.6523 | 0.6669 | 12 | 0.6669 | 82.0% |
| `gaussian_30p0_beta05_lg0075` | 0.50 | 0.075 | 9 | 0.6907 | 0.7484 | 0.6662 | 0.6662 | 9 | 0.6458 | 88.1% |
| `gaussian_30p0_beta05_lg01` | 0.50 | 0.100 | 9 | 0.6878 | 0.7452 | 0.6562 | 0.6758 | 11 | 0.6669 | 83.7% |
| `gaussian_30p0_method_beta05_lg015` | 0.50 | 0.150 | 11 | 0.6782 | 0.7677 | 0.6869 | 0.6869 | 11 | 0.6692 | 97.3% |
| `gaussian_30p0_beta05_lg02` | 0.50 | 0.200 | 4 | 0.7429 | 0.7774 | 0.6635 | 0.6635 | 4 | 0.6396 | 86.9% |
| `gaussian_30p0_beta085_lg01` | 0.85 | 0.100 | 4 | 0.7251 | 0.7806 | 0.6512 | 0.6512 | 4 | 0.6354 | 81.5% |
| `gaussian_30p0_beta09_lg01` | 0.90 | 0.100 | 4 | 0.7293 | 0.7742 | 0.6369 | 0.6504 | 5 | 0.6338 | 75.2% |

## Notes

- Gaussian noise is substantially harmful under the plain baseline: selected test accuracy drops from clean `0.6931` to gaussian `0.4669`.
- Gold-guided CL nearly closes this gap: the best run reaches `0.6869`, recovering about `97.3%` of the clean-vs-gaussian gap.
- Unlike blur and brightness, gaussian noise does not prefer very high `beta`. The best setting uses `beta=0.50`, preserving a balanced mixture of gold evaluator prediction confidence and feature-prototype similarity.
- Gaussian noise benefits from a stronger gold stability anchor. Increasing `lambda_gold` from `0.10` to `0.15` improves selected test accuracy, while increasing further to `0.20` hurts performance.
- The best current setting is therefore `beta=0.50, lambda_gold=0.15`.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/STL/` |
| Gaussian baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/STL/feature_noise/gaussian_30p0/` |
| Gold-guided beta=0.50, lambda=0.15 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/feature_noise/gaussian/gaussian_30p0_method_beta05_lg015/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/feature_noise/gaussian/gaussian_ablation/` |
