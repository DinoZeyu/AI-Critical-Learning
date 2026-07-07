# Flower_102 brightness_0p75 Experiment Results

This file summarizes the current Flower_102 `brightness_0p75` train-noise / clean-test results.

Protocol:

- Dataset: `Flower_102`
- Noise setting: train feature noise `brightness_0p75`, clean test
- Image size: `224`
- Validation source: stratified 10% split from `Image_Data/Gold_Data/Flower_102`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.5596`
- Brightness baseline selected test accuracy: `0.4496`
- Recovery ratio: `(method_acc - brightness_baseline_acc) / (clean_baseline_acc - brightness_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 20 | 0.5596 | 0.5693 | 0.5596 | 100.0% |
| Noise lower reference | Baseline | `brightness_0p75` train | 10 | 0.4496 | 0.4575 | 0.4557 | 0.0% |
| Best current method | Gold-guided CL, beta=0.80, lambda_gold=0.10 | `brightness_0p75` train | 11 | 0.6329 | 0.6365 | 0.6365 | 166.7% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `brightness_0p75_beta02_lg01` | 0.20 | 0.100 | 11 | 1.2206 | 0.6716 | 0.6084 | 0.6316 | 14 | 0.6316 | 144.4% |
| `brightness_0p75_beta05_lg01` | 0.50 | 0.100 | 11 | 1.1638 | 0.6940 | 0.6225 | 0.6261 | 12 | 0.6200 | 157.2% |
| `brightness_0p75_method_beta08_lg01` | 0.80 | 0.100 | 11 | 1.0930 | 0.6716 | 0.6329 | 0.6365 | 14 | 0.6365 | 166.7% |
| `brightness_0p75_beta09_lg01` | 0.90 | 0.100 | 11 | 1.1308 | 0.6642 | 0.6231 | 0.6341 | 14 | 0.6341 | 157.8% |
| `brightness_0p75_beta08_lg005` | 0.80 | 0.050 | 11 | 1.1726 | 0.6642 | 0.6145 | 0.6292 | 14 | 0.6292 | 150.0% |
| `brightness_0p75_beta08_lg015` | 0.80 | 0.150 | 11 | 1.1284 | 0.6791 | 0.6261 | 0.6261 | 10 | 0.6188 | 160.6% |
| `brightness_0p75_beta08_lg02` | 0.80 | 0.200 | 11 | 1.1383 | 0.6567 | 0.6237 | 0.6316 | 14 | 0.6316 | 158.3% |

## Notes

- Brightness noise is less damaging than blur for the Flower_102 baseline, but it still lowers selected test accuracy from `0.5596` to `0.4496`.
- Gold-guided CL strongly improves over the brightness baseline in every tested setting.
- The current best setting is `beta=0.80, lambda_gold=0.10`, with selected test accuracy `0.6329` and peak/final test accuracy `0.6365`.
- The best method surpasses the clean baseline reference, so the recovery ratio is above `100%`.
- Around `beta=0.80`, `lambda_gold=0.10` is the best observed stability weight. Both lower `0.05` and higher `0.15` or `0.20` weights are weaker on selected test accuracy.
- The `Peak Test Acc` column is diagnostic only. The selected result should still be reported from the validation-loss selected checkpoint.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/Flower_102/` |
| Brightness baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/Flower_102/feature_noise/brightness_0p75/` |
| Gold-guided beta=0.80, lambda_gold=0.10 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/feature_noise/brightness/brightness_0p75_method_beta08_lg01/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/feature_noise/brightness/brightness_ablations/` |
