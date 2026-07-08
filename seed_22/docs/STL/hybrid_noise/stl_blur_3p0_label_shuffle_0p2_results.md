# STL blur_3p0 + label_shuffle_0p2 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `blur_3p0 + label_shuffle_0p2`.

Protocol:

- Dataset: `STL`
- Noise setting: `hybrid_noise/blur_3p0_label_shuffle_0p2`
- Seed: `22`
- Validation source: stratified split from root `Image_Data/Gold_Data/<dataset>`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Gold evaluator path: `seed_22/Gold_Evaluators/STL/model.pt`
- Headline metric: `test_accuracy_at_selected_epoch`; peak test accuracy is diagnostic.
- Recovery ratio: `(method_sel - noise_baseline_sel) / (clean_baseline_sel - noise_baseline_sel)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `blur_3p0 + label_shuffle_0p2` train | 1 | 0.2750 | 0.2750 | 0.2227 | 0.0% |
| Best current method | Gold-guided CL, beta=0.90, lambda_gold=0.250 | `blur_3p0 + label_shuffle_0p2` train | 1 | 0.6254 | 0.6358 | 0.6358 | 83.8% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blur_3p0_label_shuffle_0p2_beta09_lg025` | 0.90 | 0.250 | 1 | 0.7183 | 0.7355 | 0.6254 | 0.6358 | 4 | 0.6358 | 83.8% |
| `blur_3p0_label_shuffle_0p2_compromise_beta09_lg02` | 0.90 | 0.200 | 1 | 0.7332 | 0.7226 | 0.6204 | 0.6338 | 3 | 0.6265 | 82.6% |
| `blur_3p0_label_shuffle_0p2_beta09_lg03` | 0.90 | 0.300 | 1 | 0.7257 | 0.7323 | 0.6146 | 0.6412 | 4 | 0.6412 | 81.2% |
| `blur_3p0_label_shuffle_0p2_feature_best_beta09_lg015` | 0.90 | 0.150 | 1 | 0.8012 | 0.7065 | 0.6058 | 0.6208 | 3 | 0.6142 | 79.1% |
| `blur_3p0_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 1 | 0.7670 | 0.7129 | 0.6050 | 0.6419 | 4 | 0.6419 | 78.9% |
| `blur_3p0_label_shuffle_0p2_beta085_lg015` | 0.85 | 0.150 | 1 | 0.7870 | 0.7226 | 0.6042 | 0.6208 | 3 | 0.6062 | 78.7% |
| `blur_3p0_label_shuffle_0p2_beta095_lg015` | 0.95 | 0.150 | 1 | 0.8037 | 0.7065 | 0.6042 | 0.6277 | 4 | 0.6277 | 78.7% |
| `blur_3p0_label_shuffle_0p2_beta09_lg01` | 0.90 | 0.100 | 1 | 0.8236 | 0.7129 | 0.6012 | 0.6012 | 1 | 0.5804 | 78.0% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6254` from `blur_3p0_label_shuffle_0p2_beta09_lg025`.
- Noise baseline selected test accuracy is `0.2750`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_22/Noise_Baseline/STL/hybrid_noise/blur_3p0_label_shuffle_0p2/` |
| Best GGCL run | `seed_22/Experiments_Results/STL/hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_beta09_lg025/` |
