# Flower_102 label_shuffle_0p2 Experiment Results

This file summarizes the current Flower_102 `label_shuffle_0p2` train-noise / clean-test results.

Protocol:

- Dataset: `Flower_102`
- Noise setting: train label noise `label_shuffle_0p2`, clean test
- Image size: `224`
- Validation source: stratified 10% split from `Image_Data/Gold_Data/Flower_102`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.5596`
- Label-noise baseline selected test accuracy: `0.4759`
- Recovery ratio: `(method_acc - label_noise_baseline_acc) / (clean_baseline_acc - label_noise_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 20 | 0.5596 | 0.5693 | 0.5596 | 100.0% |
| Noise lower reference | Baseline | `label_shuffle_0p2` train | 15 | 0.4759 | 0.4820 | 0.4820 | 0.0% |
| Best current method | Gold-guided CL, beta=0.95, lambda_gold=0.10 | `label_shuffle_0p2` train | 14 | 0.5767 | 0.6005 | 0.5657 | 120.4% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `label_shuffle_0p2_beta02_lg01` | 0.20 | 0.100 | 5 | 1.7223 | 0.6045 | 0.5235 | 0.5235 | 5 | 0.5192 | 56.9% |
| `label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 5 | 1.6229 | 0.5970 | 0.5315 | 0.5370 | 6 | 0.5308 | 66.4% |
| `label_shuffle_0p2_beta08_lg01` | 0.80 | 0.100 | 5 | 1.4779 | 0.6194 | 0.5223 | 0.5510 | 6 | 0.5370 | 55.5% |
| `label_shuffle_0p2_beta09_lg01` | 0.90 | 0.100 | 11 | 1.4355 | 0.6194 | 0.5712 | 0.5822 | 13 | 0.5736 | 113.9% |
| `label_shuffle_0p2_beta095_lg005` | 0.95 | 0.050 | 5 | 1.5402 | 0.5821 | 0.5223 | 0.5388 | 6 | 0.5284 | 55.5% |
| `label_shuffle_0p2_method_beta095_lg01` | 0.95 | 0.100 | 14 | 1.3949 | 0.6343 | 0.5767 | 0.6005 | 13 | 0.5657 | 120.4% |
| `label_shuffle_0p2_beta095_lg015` | 0.95 | 0.150 | 5 | 1.4641 | 0.5970 | 0.5321 | 0.5443 | 8 | 0.5443 | 67.2% |
| `label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 5 | 1.4844 | 0.5896 | 0.5217 | 0.5443 | 7 | 0.5345 | 54.7% |

## Notes

- Label noise lowers the Flower_102 baseline selected test accuracy from `0.5596` to `0.4759`.
- The current best setting is `beta=0.95, lambda_gold=0.10`, with selected test accuracy `0.5767` and peak test accuracy `0.6005`.
- The best method exceeds the clean baseline selected accuracy, giving a recovery ratio above `100%`.
- Label noise favors a much higher `beta` than the feature-noise settings. This suggests the frozen gold evaluator is more reliable than learner-dependent feature similarity when the training labels are corrupted.
- `beta=1.00` is much weaker than `beta=0.95`, so a small amount of feature/prototype similarity still helps stabilize the controller.
- Around `beta=0.95`, both `lambda_gold=0.05` and `0.15` are much weaker than `0.10`; the planned `0.20` run was skipped.
- The `Peak Test Acc` column is diagnostic only. The selected result should still be reported from the validation-loss selected checkpoint.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/Flower_102/` |
| Label-noise baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/Flower_102/label_noise/label_shuffle_0p2/` |
| Gold-guided beta=0.95, lambda_gold=0.10 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/label_noise/label_shuffle/label_shuffle_0p2_method_beta095_lg01/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/label_noise/label_shuffle/label_ablations/` |
