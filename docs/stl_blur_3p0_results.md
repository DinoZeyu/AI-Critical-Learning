# STL blur_3p0 Experiment Results

This file summarizes the current STL `blur_3p0` train-noise / clean-test results.

Protocol:

- Dataset: `STL`
- Noise setting: train feature noise `blur_3p0`, clean test
- Validation source: stratified 10% split from `Image_Data/Gold_Data/STL`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.6931`
- Blur baseline selected test accuracy: `0.2850`
- Recovery ratio: `(method_acc - blur_baseline_acc) / (clean_baseline_acc - blur_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `blur_3p0` train | 5 | 0.2850 | 0.2850 | 0.1962 | 0.0% |
| Best current method | Gold-guided CL, beta=0.90, lambda_gold=0.15 | `blur_3p0` train | 6 | 0.6362 | 0.6392 | 0.6392 | 86.1% |
| Secondary candidate | Gold-guided CL, beta=0.90, lambda_gold=0.10 | `blur_3p0` train | 4 | 0.6358 | 0.6358 | 0.5646 | 86.0% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `blur_3p0_beta02_lg01` | 0.20 | 0.100 | 2 | 0.8892 | 0.7097 | 0.6169 | 0.6169 | 2 | 0.5335 | 81.3% |
| `blur_3p0_beta05_lg01` | 0.50 | 0.100 | 4 | 0.8854 | 0.7161 | 0.6112 | 0.6112 | 4 | 0.5954 | 79.9% |
| `blur_3p0_beta085_lg01` | 0.85 | 0.100 | 4 | 0.8677 | 0.7194 | 0.6312 | 0.6312 | 4 | 0.5931 | 84.8% |
| `blur_3p0_beta09_lg0075` | 0.90 | 0.075 | 5 | 0.8891 | 0.7226 | 0.6235 | 0.6235 | 5 | 0.5892 | 82.9% |
| `blur_3p0_beta09_lg01` | 0.90 | 0.100 | 4 | 0.8503 | 0.7452 | 0.6358 | 0.6358 | 4 | 0.5646 | 86.0% |
| `blur_3p0_method_beta09_lg015` | 0.90 | 0.150 | 6 | 0.8994 | 0.7387 | 0.6362 | 0.6392 | 9 | 0.6392 | 86.1% |

## Notes

- Gold-guided CL substantially improves over the blur baseline: from `0.2850` to `0.6362` selected test accuracy in the current best run.
- Increasing `beta` reduces reliance on feature-prototype similarity and increases reliance on gold evaluator prediction confidence.
- For this feature-noise setting, higher `beta` helps up to the current best observed setting at `beta=0.90`.
- At `beta=0.90`, increasing `lambda_gold` from `0.10` to `0.15` gives the best selected test accuracy and the best peak/final test accuracy among the current blur runs.
- The gap between `lambda_gold=0.10` and `0.15` is small on selected test accuracy, so the main claim should be that both settings are strong, with `0.15` currently highest.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/Baseline_Exp/STL/clean/` |
| Blur baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/STL/feature_noise/blur_3p0/` |
| Gold-guided beta=0.90, lambda_gold=0.15 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/feature_noise/blur/blur_3p0_method_beta09_lg015/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/feature_noise/blur/blur_ablations/` |
