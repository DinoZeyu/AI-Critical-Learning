# Flower_102 gaussian_30p0 + label_shuffle_0p2 Hybrid Experiment Results

This file summarizes the current Flower_102 `gaussian_30p0 + label_shuffle_0p2` train-noise / clean-test results.

Protocol:

- Dataset: `Flower_102`
- Noise setting: train hybrid noise `gaussian_30p0 + label_shuffle_0p2`, clean test
- Image size: `224`
- Validation source: stratified 10% split from `Image_Data/Gold_Data/Flower_102`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.5596`
- Hybrid baseline selected test accuracy: `0.4472`
- Recovery ratio: `(method_acc - hybrid_baseline_acc) / (clean_baseline_acc - hybrid_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 20 | 0.5596 | 0.5693 | 0.5596 | 100.0% |
| Noise lower reference | Baseline | `gaussian_30p0 + label_shuffle_0p2` train | 16 | 0.4472 | 0.4679 | 0.4331 | 0.0% |
| Best current method | Gold-guided CL, beta=0.50, lambda_gold=0.10 | `gaussian_30p0 + label_shuffle_0p2` train | 14 | 0.5486 | 0.5626 | 0.5559 | 90.2% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gaussian_30p0_label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 14 | 1.4790 | 0.6194 | 0.5486 | 0.5626 | 11 | 0.5559 | 90.2% |
| `gaussian_30p0_label_shuffle_0p2_feature_best_beta05_lg015` | 0.50 | 0.150 | 11 | 1.4617 | 0.6194 | 0.5461 | 0.5626 | 13 | 0.5547 | 88.0% |
| `gaussian_30p0_label_shuffle_0p2_beta055_lg015` | 0.55 | 0.150 | 11 | 1.4445 | 0.5970 | 0.5437 | 0.5626 | 14 | 0.5626 | 85.9% |
| `gaussian_30p0_label_shuffle_0p2_beta045_lg015` | 0.45 | 0.150 | 5 | 1.6026 | 0.6119 | 0.5247 | 0.5290 | 8 | 0.5290 | 69.0% |
| `gaussian_30p0_label_shuffle_0p2_beta05_lg02` | 0.50 | 0.200 | 5 | 1.5344 | 0.6045 | 0.5229 | 0.5382 | 8 | 0.5382 | 67.4% |
| `gaussian_30p0_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 5 | 1.4576 | 0.6045 | 0.5217 | 0.5370 | 6 | 0.5351 | 66.3% |
| `gaussian_30p0_label_shuffle_0p2_beta06_lg015` | 0.60 | 0.150 | 5 | 1.5545 | 0.5970 | 0.5192 | 0.5394 | 7 | 0.5327 | 64.1% |
| `gaussian_30p0_label_shuffle_0p2_compromise_beta07_lg015` | 0.70 | 0.150 | 5 | 1.5036 | 0.5970 | 0.5162 | 0.5412 | 7 | 0.5284 | 61.4% |
| `gaussian_30p0_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 5 | 1.5033 | 0.6045 | 0.5064 | 0.5370 | 8 | 0.5370 | 52.7% |

## Notes

- The gaussian hybrid baseline is stronger than the blur and brightness hybrid baselines, but it still drops below the clean baseline from `0.5596` to `0.4472`.
- Gold-guided CL recovers most of the clean-vs-hybrid gap: the best selected accuracy reaches `0.5486`, recovering `90.2%` of the gap.
- The best setting is `beta=0.50, lambda_gold=0.10`. This is close to the single Gaussian setting and confirms that Gaussian hybrid noise benefits from preserving feature/prototype similarity.
- Increasing `beta` toward the label-noise best setting is clearly weaker here. The high-beta runs select early checkpoints and give lower selected accuracy.
- `lambda_gold=0.10` slightly beats `0.15` on validation-selected test accuracy, while `0.20` is too strong for this hybrid setting.
- The `Peak Test Acc` column is diagnostic only. The selected result should still be reported from the validation-loss selected checkpoint.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/Flower_102/` |
| Hybrid baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/Flower_102/hybrid_noise/gaussian_30p0_label_shuffle_0p2/` |
| Gold-guided beta=0.50, lambda_gold=0.10 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_beta05_lg01/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/hybrid_noise/gaussian_label_shuffle/gaussian_hybrid_ablation/` |
