# STL brightness_0p75 Experiment Results

This file summarizes the current STL `brightness_0p75` train-noise / clean-test results.

Protocol:

- Dataset: `STL`
- Noise setting: train feature noise `brightness_0p75`, clean test
- Validation source: stratified 10% split from `Image_Data/Gold_Data/STL`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.6931`
- Brightness baseline selected test accuracy: `0.6527`
- Recovery ratio: `(method_acc - brightness_baseline_acc) / (clean_baseline_acc - brightness_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise reference | Baseline | `brightness_0p75` train | 9 | 0.6527 | 0.6546 | 0.5162 | 0.0% |
| Best current method | Gold-guided CL, beta=0.90, lambda_gold=0.10 | `brightness_0p75` train | 12 | 0.7435 | 0.7500 | 0.7281 | 224.8% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `brightness_0p75_beta02_lg01` | 0.20 | 0.100 | 12 | 0.6266 | 0.7839 | 0.7315 | 0.7362 | 15 | 0.7362 | 195.2% |
| `brightness_0p75_beta05_lg01` | 0.50 | 0.100 | 10 | 0.5616 | 0.8161 | 0.7296 | 0.7373 | 12 | 0.7365 | 190.5% |
| `brightness_0p75_beta085_lg01` | 0.85 | 0.100 | 11 | 0.5614 | 0.8226 | 0.7362 | 0.7473 | 14 | 0.7473 | 206.7% |
| `brightness_0p75_method_beta09_lg01` | 0.90 | 0.100 | 12 | 0.5216 | 0.8355 | 0.7435 | 0.7500 | 14 | 0.7281 | 224.8% |
| `brightness_0p75_beta09_lg0075` | 0.90 | 0.075 | 12 | 0.5735 | 0.8065 | 0.7304 | 0.7396 | 14 | 0.7269 | 192.4% |
| `brightness_0p75_beta09_lg015` | 0.90 | 0.150 | 7 | 0.5690 | 0.8161 | 0.7100 | 0.7292 | 10 | 0.7292 | 141.9% |

## Notes

- Brightness noise is much milder than blur under the baseline: `0.6527` selected test accuracy, only about 4 points below the clean baseline.
- Gold-guided CL still improves strongly over the brightness baseline and also exceeds the clean baseline in this run: `0.7435` selected test accuracy and `0.7500` peak test accuracy.
- Higher `beta` is again helpful. The best current setting is `beta=0.90`, which puts more weight on gold evaluator prediction confidence and less on feature-prototype similarity.
- Gold stability should stay at `lambda_gold=0.10` for this setting. Reducing it to `0.075` and increasing it to `0.15` both hurt selected test accuracy.
- Recovery ratio is above 100% because the method exceeds the clean-train baseline on the selected test metric.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/STL/` |
| Brightness baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/STL/feature_noise/brightness_0p75/` |
| Gold-guided beta=0.90 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/feature_noise/brightness/brightness_0p75_method_beta09_lg01/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/feature_noise/brightness/brightness_ablation/` |
