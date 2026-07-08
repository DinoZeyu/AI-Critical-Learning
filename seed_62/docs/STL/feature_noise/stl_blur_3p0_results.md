# STL blur_3p0 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `blur_3p0`.

Protocol:

- Dataset: `STL`
- Noise setting: `feature_noise/blur_3p0`
- Seed: `62`
- Validation source: stratified split from root `Image_Data/Gold_Data/<dataset>`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Gold evaluator path: `seed_62/Gold_Evaluators/STL/model.pt`
- Headline metric: `test_accuracy_at_selected_epoch`; peak test accuracy is diagnostic.
- Recovery ratio: `(method_sel - noise_baseline_sel) / (clean_baseline_sel - noise_baseline_sel)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `blur_3p0` train | 1 | 0.2262 | 0.2696 | 0.2104 | 0.0% |
| Best current method | Gold-guided CL, beta=0.90, lambda_gold=0.150 | `blur_3p0` train | 2 | 0.6300 | 0.6300 | 0.5812 | 86.5% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blur_3p0_method_beta09_lg015` | 0.90 | 0.150 | 2 | 0.7994 | 0.7097 | 0.6300 | 0.6300 | 2 | 0.5812 | 86.5% |
| `blur_3p0_beta09_lg01` | 0.90 | 0.100 | 2 | 0.8206 | 0.7290 | 0.6300 | 0.6300 | 2 | 0.5812 | 86.5% |
| `blur_3p0_beta085_lg01` | 0.85 | 0.100 | 2 | 0.8070 | 0.7161 | 0.6223 | 0.6223 | 2 | 0.5888 | 84.8% |
| `blur_3p0_beta09_lg0075` | 0.90 | 0.075 | 2 | 0.8515 | 0.7065 | 0.6135 | 0.6135 | 2 | 0.6019 | 82.9% |
| `blur_3p0_beta05_lg01` | 0.50 | 0.100 | 2 | 0.8919 | 0.6742 | 0.6088 | 0.6088 | 2 | 0.5796 | 82.0% |
| `blur_3p0_beta02_lg01` | 0.20 | 0.100 | 2 | 0.9095 | 0.6903 | 0.5992 | 0.5992 | 2 | 0.5735 | 79.9% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6300` from `blur_3p0_method_beta09_lg015`.
- Noise baseline selected test accuracy is `0.2262`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_62/Noise_Baseline/STL/feature_noise/blur_3p0/` |
| Best GGCL run | `seed_62/Experiments_Results/STL/feature_noise/blur/blur_3p0_method_beta09_lg015/` |
