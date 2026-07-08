# Flower_102 blur_3p0 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `blur_3p0`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `feature_noise/blur_3p0`
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
| Noise lower reference | Baseline | `blur_3p0` train | 7 | 0.3861 | 0.4007 | 0.3873 | 0.0% |
| Best current method | Gold-guided CL, beta=0.50, lambda_gold=0.100 | `blur_3p0` train | 14 | 0.6396 | 0.6396 | 0.6329 | 146.1% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blur_3p0_beta05_lg01` | 0.50 | 0.100 | 14 | 1.4047 | 0.6866 | 0.6396 | 0.6396 | 14 | 0.6329 | 146.1% |
| `blur_3p0_method_beta08_lg01` | 0.80 | 0.100 | 12 | 1.4090 | 0.6493 | 0.6188 | 0.6347 | 14 | 0.5785 | 134.2% |
| `blur_3p0_beta09_lg01` | 0.90 | 0.100 | 12 | 1.3726 | 0.6567 | 0.6188 | 0.6371 | 14 | 0.5901 | 134.2% |
| `blur_3p0_beta08_lg005` | 0.80 | 0.050 | 12 | 1.3980 | 0.6642 | 0.6145 | 0.6274 | 14 | 0.5840 | 131.7% |
| `blur_3p0_beta08_lg02` | 0.80 | 0.200 | 8 | 1.4539 | 0.5970 | 0.5822 | 0.6066 | 11 | 0.6066 | 113.0% |
| `blur_3p0_beta08_lg015` | 0.80 | 0.150 | 8 | 1.4800 | 0.6418 | 0.5742 | 0.5950 | 11 | 0.5950 | 108.5% |
| `blur_3p0_beta085_lg01` | 0.85 | 0.100 | 8 | 1.4424 | 0.6418 | 0.5718 | 0.5870 | 10 | 0.5828 | 107.0% |
| `blur_3p0_beta02_lg01` | 0.20 | 0.100 | 8 | 1.5186 | 0.6045 | 0.5663 | 0.5870 | 10 | 0.5773 | 103.9% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6396` from `blur_3p0_beta05_lg01`.
- Noise baseline selected test accuracy is `0.3861`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_62/Noise_Baseline/Flower_102/feature_noise/blur_3p0/` |
| Best GGCL run | `seed_62/Experiments_Results/Flower_102/feature_noise/blur/blur_ablations/blur_3p0_beta05_lg01/` |
