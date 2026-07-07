# Flower_102 brightness_0p75 + label_shuffle_0p2 Hybrid Experiment Results

This file summarizes the current Flower_102 `brightness_0p75 + label_shuffle_0p2` train-noise / clean-test results.

Protocol:

- Dataset: `Flower_102`
- Noise setting: train hybrid noise `brightness_0p75 + label_shuffle_0p2`, clean test
- Image size: `224`
- Validation source: stratified 10% split from `Image_Data/Gold_Data/Flower_102`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.5596`
- Hybrid baseline selected test accuracy: `0.3488`
- Recovery ratio: `(method_acc - hybrid_baseline_acc) / (clean_baseline_acc - hybrid_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 20 | 0.5596 | 0.5693 | 0.5596 | 100.0% |
| Noise lower reference | Baseline | `brightness_0p75 + label_shuffle_0p2` train | 11 | 0.3488 | 0.3653 | 0.3415 | 0.0% |
| Best current method | Gold-guided CL, beta=0.80, lambda_gold=0.10 | `brightness_0p75 + label_shuffle_0p2` train | 11 | 0.5663 | 0.5663 | 0.5535 | 103.2% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `brightness_0p75_label_shuffle_0p2_feature_best_beta08_lg01` | 0.80 | 0.100 | 11 | 1.4569 | 0.6418 | 0.5663 | 0.5663 | 11 | 0.5535 | 103.2% |
| `brightness_0p75_label_shuffle_0p2_beta078_lg01` | 0.78 | 0.100 | 11 | 1.4318 | 0.6269 | 0.5602 | 0.5632 | 13 | 0.5418 | 100.3% |
| `brightness_0p75_label_shuffle_0p2_beta07_lg01` | 0.70 | 0.100 | 11 | 1.4624 | 0.6418 | 0.5547 | 0.5608 | 9 | 0.5528 | 97.7% |
| `brightness_0p75_label_shuffle_0p2_beta08_lg005` | 0.80 | 0.050 | 7 | 1.6204 | 0.5373 | 0.5302 | 0.5388 | 9 | 0.5266 | 86.1% |
| `brightness_0p75_label_shuffle_0p2_compromise_beta09_lg01` | 0.90 | 0.100 | 5 | 1.5104 | 0.5896 | 0.5150 | 0.5406 | 7 | 0.5235 | 78.8% |
| `brightness_0p75_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 5 | 1.4971 | 0.5821 | 0.5131 | 0.5284 | 7 | 0.5150 | 78.0% |
| `brightness_0p75_label_shuffle_0p2_beta085_lg01` | 0.85 | 0.100 | 5 | 1.5657 | 0.5970 | 0.5125 | 0.5284 | 7 | 0.5247 | 77.7% |
| `brightness_0p75_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 5 | 1.5018 | 0.5821 | 0.5119 | 0.5363 | 8 | 0.5363 | 77.4% |
| `brightness_0p75_label_shuffle_0p2_beta075_lg01` | 0.75 | 0.100 | 5 | 1.5894 | 0.5821 | 0.5113 | 0.5296 | 7 | 0.5192 | 77.1% |

## Notes

- The hybrid baseline is much weaker than the clean baseline, dropping from `0.5596` to `0.3488` selected test accuracy.
- Gold-guided CL closes the gap and slightly exceeds the clean baseline: the best selected accuracy reaches `0.5663`.
- The best setting matches the single brightness result, using `beta=0.80` and `lambda_gold=0.10`.
- Higher beta values are clearly weaker in this hybrid setting. `beta=0.85`, `0.90`, and `0.95` all select much earlier checkpoints and lose accuracy.
- Around `beta=0.80`, `lambda_gold=0.10` is best. Both `0.05` and `0.15` reduce selected test accuracy.
- The `Peak Test Acc` column is diagnostic only. The selected result should still be reported from the validation-loss selected checkpoint.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/Flower_102/` |
| Hybrid baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/Flower_102/hybrid_noise/brightness_0p75_label_shuffle_0p2/` |
| Gold-guided beta=0.80, lambda_gold=0.10 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/hybrid_noise/brightness_label_shuffle/brightness_0p75_label_shuffle_0p2_feature_best_beta08_lg01/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/hybrid_noise/brightness_label_shuffle/brightness_hybrid_ablation/` |
