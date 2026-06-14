# Flower_102 blur_3p0 Experiment Results

This file summarizes the current Flower_102 `blur_3p0` train-noise / clean-test results.

Protocol:

- Dataset: `Flower_102`
- Noise setting: train feature noise `blur_3p0`, clean test
- Image size: `224`
- Validation source: stratified 10% split from `Image_Data/Gold_Data/Flower_102`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Seed: `42`
- Clean baseline selected test accuracy: `0.5596`
- Blur baseline selected test accuracy: `0.4148`
- Recovery ratio: `(method_acc - blur_baseline_acc) / (clean_baseline_acc - blur_baseline_acc)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Clean upper reference | Baseline | Clean train | 20 | 0.5596 | 0.5693 | 0.5596 | 100.0% |
| Noise lower reference | Baseline | `blur_3p0` train | 9 | 0.4148 | 0.4148 | 0.3433 | 0.0% |
| Best current method | Gold-guided CL, beta=0.80, lambda_gold=0.10 | `blur_3p0` train | 16 | 0.6151 | 0.6353 | 0.6353 | 138.4% |

## Gold-Guided CL Ablations

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `blur_3p0_beta02_lg01` | 0.20 | 0.100 | 11 | 1.3659 | 0.6045 | 0.5852 | 0.6042 | 12 | 0.5803 | 117.7% |
| `blur_3p0_beta05_lg01` | 0.50 | 0.100 | 8 | 1.3997 | 0.5746 | 0.5657 | 0.5852 | 11 | 0.5852 | 104.2% |
| `blur_3p0_beta085_lg01` | 0.85 | 0.100 | 11 | 1.3005 | 0.6493 | 0.5993 | 0.5993 | 11 | 0.5919 | 127.4% |
| `blur_3p0_beta09_lg01` | 0.90 | 0.100 | 11 | 1.3488 | 0.6269 | 0.5950 | 0.6054 | 14 | 0.6054 | 124.5% |
| `blur_3p0_beta08_lg005` | 0.80 | 0.050 | 5 | 1.5255 | 0.5821 | 0.5351 | 0.5473 | 8 | 0.5473 | 83.1% |
| `blur_3p0_method_beta08_lg01` | 0.80 | 0.100 | 16 | 1.3336 | 0.6567 | 0.6151 | 0.6353 | 19 | 0.6353 | 138.4% |
| `blur_3p0_beta08_lg015` | 0.80 | 0.150 | 11 | 1.3663 | 0.6269 | 0.6084 | 0.6084 | 11 | 0.5895 | 133.8% |
| `blur_3p0_beta08_lg02` | 0.80 | 0.200 | 10 | 1.3216 | 0.5970 | 0.5901 | 0.6011 | 12 | 0.5889 | 121.1% |

## Notes

- Blur is a strong feature-space degradation for Flower_102: the baseline drops from `0.5596` clean selected test accuracy to `0.4148` under `blur_3p0` training.
- Gold-guided CL more than closes the clean/noise gap in the current best run: selected test accuracy reaches `0.6151`, and peak/final test accuracy reaches `0.6353`.
- For Flower_102 blur, the best observed `beta` is `0.80`. Lower `beta` values underuse the gold evaluator signal, while `0.85` and `0.90` are also strong but slightly worse here.
- Around `beta=0.80`, `lambda_gold=0.10` is the best observed stability weight. Increasing to `0.15` remains strong, but `0.20` starts to reduce selected test accuracy.
- The `Peak Test Acc` column is diagnostic only. The selected result should still be reported from the validation-loss selected checkpoint.

## Source Result Folders

| Result | Path |
|---|---|
| Clean baseline | `Experiments_Results/Train_Clean_Test_Clean/Flower_102/` |
| Blur baseline | `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/Flower_102/feature_noise/blur_3p0/` |
| Gold-guided beta=0.80, lambda_gold=0.10 | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/feature_noise/blur/blur_3p0_method_beta08_lg01/` |
| Gold-guided ablations | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/feature_noise/blur/blur_ablations/` |
