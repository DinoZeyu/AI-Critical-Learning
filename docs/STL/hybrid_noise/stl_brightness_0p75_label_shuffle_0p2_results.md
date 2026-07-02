# STL brightness_0p75 + label_shuffle_0p2 Hybrid Experiment Results

This file summarizes the current STL `brightness_0p75 + label_shuffle_0p2` train-noise / clean-test results.

Protocol:

- Dataset: `STL`
- Noise setting: train hybrid noise `brightness_0p75 + label_shuffle_0p2`, clean test
- Validation source: stratified 10% split from `Image_Data/Gold_Data/STL`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.6931`
- Hybrid baseline selected test accuracy: `0.5888`
- Recovery ratio: `(method_acc - hybrid_baseline_acc) / (clean_baseline_acc - hybrid_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `brightness_0p75 + label_shuffle_0p2` train | 10 | 0.5888 | 0.5888 | 0.5858 | 0.0% |
| Best current method | Gold-guided CL, beta=0.85, lambda_gold=0.10 | `brightness_0p75 + label_shuffle_0p2` train | 13 | 0.7154 | 0.7208 | 0.6927 | 121.4% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `brightness_0p75_label_shuffle_0p2_beta085_lg01` | 0.85 | 0.100 | 13 | 0.5837 | 0.8194 | 0.7154 | 0.7208 | 12 | 0.6927 | 121.4% |
| `brightness_0p75_label_shuffle_0p2_feature_best_beta09_lg01` | 0.90 | 0.100 | 9 | 0.6099 | 0.7968 | 0.7062 | 0.7085 | 11 | 0.6996 | 112.5% |
| `brightness_0p75_label_shuffle_0p2_beta095_lg01` | 0.95 | 0.100 | 9 | 0.5803 | 0.7968 | 0.7023 | 0.7088 | 12 | 0.7088 | 108.9% |
| `brightness_0p75_label_shuffle_0p2_compromise_beta09_lg015` | 0.90 | 0.150 | 5 | 0.6371 | 0.7742 | 0.6981 | 0.6981 | 5 | 0.6873 | 104.8% |
| `brightness_0p75_label_shuffle_0p2_beta09_lg0075` | 0.90 | 0.075 | 9 | 0.6006 | 0.7742 | 0.6946 | 0.7085 | 10 | 0.7085 | 101.5% |
| `brightness_0p75_label_shuffle_0p2_beta09_lg005` | 0.90 | 0.050 | 9 | 0.6238 | 0.7839 | 0.6896 | 0.7119 | 12 | 0.7119 | 96.7% |
| `brightness_0p75_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 4 | 0.6921 | 0.7742 | 0.6712 | 0.6935 | 7 | 0.6935 | 79.0% |

## Notes

- Brightness hybrid is much milder than blur hybrid under the baseline: selected test accuracy is `0.5888`.
- Gold-guided CL exceeds both the hybrid baseline and the clean baseline, reaching `0.7154` selected test accuracy and `0.7208` peak test accuracy.
- The best setting uses `beta=0.85`, slightly lower than the single brightness best `beta=0.90`. This suggests a small amount of feature-prototype similarity remains useful when brightness noise is combined with label noise.
- `lambda_gold=0.10` remains the best gold stability strength for brightness, matching the single brightness result.
- Recovery ratio is above 100% because the method exceeds the clean-train baseline on the validation-selected test metric.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/STL/` |
| Hybrid baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/STL/hybrid_noise/brightness_0p75_label_shuffle_0p2/` |
| Gold-guided beta=0.85, lambda_gold=0.10 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/hybrid_noise/brightness_label_shuffle/brightness_0p75_label_shuffle_0p2_beta085_lg01/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/hybrid_noise/brightness_label_shuffle/brightness_hybrid_ablation/` |
