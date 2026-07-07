# Flower_102 blur_3p0 + label_shuffle_0p2 Hybrid Experiment Results

This file summarizes the current Flower_102 `blur_3p0 + label_shuffle_0p2` train-noise / clean-test results.

Protocol:

- Dataset: `Flower_102`
- Noise setting: train hybrid noise `blur_3p0 + label_shuffle_0p2`, clean test
- Image size: `224`
- Validation source: stratified 10% split from `Image_Data/Gold_Data/Flower_102`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.5596`
- Hybrid baseline selected test accuracy: `0.3885`
- Recovery ratio: `(method_acc - hybrid_baseline_acc) / (clean_baseline_acc - hybrid_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 20 | 0.5596 | 0.5693 | 0.5596 | 100.0% |
| Noise lower reference | Baseline | `blur_3p0 + label_shuffle_0p2` train | 11 | 0.3885 | 0.3928 | 0.3928 | 0.0% |
| Best current method | Gold-guided CL, beta=0.70, lambda_gold=0.10 | `blur_3p0 + label_shuffle_0p2` train | 11 | 0.5547 | 0.5547 | 0.5107 | 97.1% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `blur_3p0_label_shuffle_0p2_beta07_lg01` | 0.70 | 0.100 | 11 | 1.4984 | 0.6045 | 0.5547 | 0.5547 | 11 | 0.5107 | 97.1% |
| `blur_3p0_label_shuffle_0p2_feature_best_beta08_lg01` | 0.80 | 0.100 | 11 | 1.5050 | 0.6119 | 0.5528 | 0.5528 | 11 | 0.5186 | 96.1% |
| `blur_3p0_label_shuffle_0p2_beta075_lg01` | 0.75 | 0.100 | 7 | 1.5864 | 0.5597 | 0.5254 | 0.5345 | 9 | 0.5199 | 80.0% |
| `blur_3p0_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 5 | 1.5779 | 0.5672 | 0.5205 | 0.5254 | 7 | 0.5003 | 77.1% |
| `blur_3p0_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 5 | 1.5497 | 0.5970 | 0.5174 | 0.5315 | 7 | 0.5052 | 75.4% |
| `blur_3p0_label_shuffle_0p2_compromise_beta09_lg01` | 0.90 | 0.100 | 5 | 1.5736 | 0.5522 | 0.5156 | 0.5174 | 7 | 0.5040 | 74.3% |
| `blur_3p0_label_shuffle_0p2_beta08_lg005` | 0.80 | 0.050 | 5 | 1.6722 | 0.5299 | 0.4918 | 0.5144 | 7 | 0.4979 | 60.4% |

## Notes

- The hybrid baseline is substantially weaker than the clean baseline, dropping from `0.5596` to `0.3885` selected test accuracy.
- Gold-guided CL nearly closes the clean-vs-hybrid gap: the best selected accuracy reaches `0.5547`, only `0.0049` below the clean baseline.
- The best hybrid setting uses `beta=0.70`, which is lower than the single blur best setting `beta=0.80`. This suggests that Flower_102 blur+label hybrid noise benefits from keeping more feature/prototype similarity than pure blur.
- Around the best beta region, `lambda_gold=0.10` is still the strongest stability weight. Both `0.05` and `0.15` reduce selected test accuracy.
- The `Peak Test Acc` column is diagnostic only. The selected result should still be reported from the validation-loss selected checkpoint.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/Flower_102/` |
| Hybrid baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/Flower_102/hybrid_noise/blur_3p0_label_shuffle_0p2/` |
| Gold-guided beta=0.70, lambda_gold=0.10 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_beta07_lg01/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/hybrid_noise/blur_label_shuffle/blur_hybrid_ablation/` |
