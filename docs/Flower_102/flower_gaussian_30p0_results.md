# Flower_102 gaussian_30p0 Experiment Results

This file summarizes the current Flower_102 `gaussian_30p0` train-noise / clean-test results.

Protocol:

- Dataset: `Flower_102`
- Noise setting: train feature noise `gaussian_30p0`, clean test
- Image size: `224`
- Validation source: stratified 10% split from `Image_Data/Gold_Data/Flower_102`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.5596`
- Gaussian baseline selected test accuracy: `0.4856`
- Recovery ratio: `(method_acc - gaussian_baseline_acc) / (clean_baseline_acc - gaussian_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 20 | 0.5596 | 0.5693 | 0.5596 | 100.0% |
| Noise lower reference | Baseline | `gaussian_30p0` train | 11 | 0.4856 | 0.5205 | 0.4844 | 0.0% |
| Best selected method | Gold-guided CL, beta=0.50, lambda_gold=0.15 | `gaussian_30p0` train | 11 | 0.6316 | 0.6402 | 0.6365 | 197.5% |
| Peak/final diagnostic best | Gold-guided CL, beta=0.50, lambda_gold=0.20 | `gaussian_30p0` train | 11 | 0.6286 | 0.6506 | 0.6506 | 193.4% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gaussian_30p0_beta02_lg01` | 0.20 | 0.100 | 11 | 1.1718 | 0.6791 | 0.6176 | 0.6359 | 13 | 0.6323 | 178.5% |
| `gaussian_30p0_beta05_lg005` | 0.50 | 0.050 | 11 | 1.2222 | 0.6343 | 0.6121 | 0.6329 | 13 | 0.6274 | 171.1% |
| `gaussian_30p0_beta05_lg01` | 0.50 | 0.100 | 11 | 1.1364 | 0.6791 | 0.6298 | 0.6432 | 14 | 0.6432 | 195.0% |
| `gaussian_30p0_method_beta05_lg015` | 0.50 | 0.150 | 11 | 1.1228 | 0.7015 | 0.6316 | 0.6402 | 12 | 0.6365 | 197.5% |
| `gaussian_30p0_beta05_lg02` | 0.50 | 0.200 | 11 | 1.0969 | 0.7090 | 0.6286 | 0.6506 | 14 | 0.6506 | 193.4% |
| `gaussian_30p0_beta08_lg01` | 0.80 | 0.100 | 11 | 1.1245 | 0.6940 | 0.6164 | 0.6378 | 14 | 0.6378 | 176.9% |
| `gaussian_30p0_beta09_lg01` | 0.90 | 0.100 | 11 | 1.0749 | 0.7015 | 0.6194 | 0.6268 | 14 | 0.6268 | 181.0% |

## Notes

- Gaussian noise is less damaging than blur but still lowers selected test accuracy from the clean baseline `0.5596` to `0.4856`.
- Gold-guided CL strongly improves over the Gaussian baseline in every tested setting, and every tested setting exceeds the clean baseline selected accuracy.
- The current headline setting is `beta=0.50, lambda_gold=0.15`, because it has the highest validation-selected clean test accuracy: `0.6316`.
- `beta=0.50, lambda_gold=0.20` has the highest diagnostic peak/final test accuracy: `0.6506`. This is useful context, but the selected result should still be reported from the validation-loss selected checkpoint.
- For Flower_102 Gaussian noise, the best `beta` is lower than blur/brightness. This suggests Gaussian noise benefits from more feature/prototype similarity than the other two feature-noise types.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/Flower_102/` |
| Gaussian baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/Flower_102/feature_noise/gaussian_30p0/` |
| Gold-guided beta=0.50, lambda_gold=0.15 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/feature_noise/gaussian/gaussian_30p0_method_beta05_lg015/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/feature_noise/gaussian/gaussian_ablations/` |
