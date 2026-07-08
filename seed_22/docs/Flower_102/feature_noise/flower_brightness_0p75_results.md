# Flower_102 brightness_0p75 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `brightness_0p75`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `feature_noise/brightness_0p75`
- Seed: `22`
- Validation source: stratified split from root `Image_Data/Gold_Data/<dataset>`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Gold evaluator path: `seed_22/Gold_Evaluators/Flower_102/model.pt`
- Headline metric: `test_accuracy_at_selected_epoch`; peak test accuracy is diagnostic.
- Recovery ratio: `(method_sel - noise_baseline_sel) / (clean_baseline_sel - noise_baseline_sel)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean upper reference | Baseline | Clean train | 20 | 0.5596 | 0.5693 | 0.5596 | 100.0% |
| Noise lower reference | Baseline | `brightness_0p75` train | 8 | 0.4282 | 0.4368 | 0.4368 | 0.0% |
| Best current method | Gold-guided CL, beta=0.90, lambda_gold=0.100 | `brightness_0p75` train | 15 | 0.6365 | 0.6365 | 0.6103 | 158.6% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brightness_0p75_beta09_lg01` | 0.90 | 0.100 | 15 | 1.5371 | 0.5821 | 0.6365 | 0.6365 | 15 | 0.6103 | 158.6% |
| `brightness_0p75_method_beta08_lg01` | 0.80 | 0.100 | 4 | 1.6976 | 0.5522 | 0.5156 | 0.5632 | 7 | 0.5632 | 66.5% |
| `brightness_0p75_beta05_lg01` | 0.50 | 0.100 | 4 | 1.7015 | 0.5597 | 0.5107 | 0.5522 | 7 | 0.5522 | 62.8% |
| `brightness_0p75_beta08_lg015` | 0.80 | 0.150 | 4 | 1.6705 | 0.5522 | 0.5082 | 0.5730 | 7 | 0.5730 | 60.9% |
| `brightness_0p75_beta08_lg02` | 0.80 | 0.200 | 4 | 1.6588 | 0.5672 | 0.5064 | 0.5663 | 7 | 0.5663 | 59.5% |
| `brightness_0p75_beta02_lg01` | 0.20 | 0.100 | 4 | 1.7419 | 0.5299 | 0.4991 | 0.5412 | 7 | 0.5412 | 54.0% |
| `brightness_0p75_beta08_lg005` | 0.80 | 0.050 | 4 | 1.7624 | 0.5373 | 0.4960 | 0.5327 | 7 | 0.5327 | 51.6% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6365` from `brightness_0p75_beta09_lg01`.
- Noise baseline selected test accuracy is `0.4282`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_22/Noise_Baseline/Flower_102/feature_noise/brightness_0p75/` |
| Best GGCL run | `seed_22/Experiments_Results/Flower_102/feature_noise/brightness/brightness_ablations/brightness_0p75_beta09_lg01/` |
