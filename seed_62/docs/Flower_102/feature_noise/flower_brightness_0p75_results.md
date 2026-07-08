# Flower_102 brightness_0p75 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `brightness_0p75`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `feature_noise/brightness_0p75`
- Seed: `62`
- Validation source: stratified split from root `Image_Data/Gold_Data/<dataset>`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Gold evaluator path: `seed_62/Gold_Evaluators/Flower_102/model.pt`
- Headline metric: `test_accuracy_at_selected_epoch`; peak test accuracy is diagnostic.
- Recovery ratio: `(method_sel - noise_baseline_sel) / (clean_baseline_sel - noise_baseline_sel)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean upper reference | Baseline | Clean train | 20 | 0.5596 | 0.5693 | 0.5596 | 100.0% |
| Noise lower reference | Baseline | `brightness_0p75` train | 10 | 0.4380 | 0.4582 | 0.4582 | 0.0% |
| Best current method | Gold-guided CL, beta=0.80, lambda_gold=0.150 | `brightness_0p75` train | 17 | 0.6451 | 0.6518 | 0.6518 | 170.4% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brightness_0p75_beta08_lg015` | 0.80 | 0.150 | 17 | 1.2772 | 0.6791 | 0.6451 | 0.6518 | 20 | 0.6518 | 170.4% |
| `brightness_0p75_beta08_lg005` | 0.80 | 0.050 | 13 | 1.3877 | 0.6194 | 0.6365 | 0.6390 | 14 | 0.6347 | 163.3% |
| `brightness_0p75_beta09_lg01` | 0.90 | 0.100 | 9 | 1.3036 | 0.6791 | 0.6078 | 0.6121 | 12 | 0.6121 | 139.7% |
| `brightness_0p75_method_beta08_lg01` | 0.80 | 0.100 | 9 | 1.3179 | 0.6940 | 0.6054 | 0.6158 | 12 | 0.6158 | 137.7% |
| `brightness_0p75_beta08_lg02` | 0.80 | 0.200 | 8 | 1.2690 | 0.6493 | 0.5999 | 0.6170 | 11 | 0.6170 | 133.2% |
| `brightness_0p75_beta05_lg01` | 0.50 | 0.100 | 9 | 1.3604 | 0.6493 | 0.5974 | 0.6194 | 12 | 0.6194 | 131.2% |
| `brightness_0p75_beta02_lg01` | 0.20 | 0.100 | 9 | 1.4376 | 0.6269 | 0.5883 | 0.6103 | 12 | 0.6103 | 123.6% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6451` from `brightness_0p75_beta08_lg015`.
- Noise baseline selected test accuracy is `0.4380`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_62/Noise_Baseline/Flower_102/feature_noise/brightness_0p75/` |
| Best GGCL run | `seed_62/Experiments_Results/Flower_102/feature_noise/brightness/brightness_ablations/brightness_0p75_beta08_lg015/` |
