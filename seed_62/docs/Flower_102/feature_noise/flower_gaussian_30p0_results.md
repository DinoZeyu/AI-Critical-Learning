# Flower_102 gaussian_30p0 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `gaussian_30p0`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `feature_noise/gaussian_30p0`
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
| Noise lower reference | Baseline | `gaussian_30p0` train | 16 | 0.5186 | 0.5186 | 0.4936 | 0.0% |
| Best current method | Gold-guided CL, beta=0.80, lambda_gold=0.100 | `gaussian_30p0` train | 12 | 0.6286 | 0.6426 | 0.6426 | 268.7% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gaussian_30p0_beta08_lg01` | 0.80 | 0.100 | 12 | 1.3463 | 0.6716 | 0.6286 | 0.6426 | 15 | 0.6426 | 268.7% |
| `gaussian_30p0_beta05_lg02` | 0.50 | 0.200 | 9 | 1.3088 | 0.6567 | 0.6237 | 0.6335 | 11 | 0.6219 | 256.7% |
| `gaussian_30p0_beta02_lg01` | 0.20 | 0.100 | 10 | 1.3744 | 0.6716 | 0.6188 | 0.6353 | 13 | 0.6353 | 244.8% |
| `gaussian_30p0_beta09_lg01` | 0.90 | 0.100 | 12 | 1.3061 | 0.6567 | 0.6170 | 0.6359 | 15 | 0.6359 | 240.3% |
| `gaussian_30p0_beta05_lg01` | 0.50 | 0.100 | 8 | 1.3247 | 0.6791 | 0.6072 | 0.6213 | 11 | 0.6213 | 216.4% |
| `gaussian_30p0_method_beta05_lg015` | 0.50 | 0.150 | 8 | 1.3481 | 0.6716 | 0.6005 | 0.6219 | 11 | 0.6219 | 200.0% |
| `gaussian_30p0_beta05_lg005` | 0.50 | 0.050 | 9 | 1.4057 | 0.6269 | 0.5950 | 0.6298 | 12 | 0.6298 | 186.6% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6286` from `gaussian_30p0_beta08_lg01`.
- Noise baseline selected test accuracy is `0.5186`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_62/Noise_Baseline/Flower_102/feature_noise/gaussian_30p0/` |
| Best GGCL run | `seed_62/Experiments_Results/Flower_102/feature_noise/gaussian/gaussian_ablations/gaussian_30p0_beta08_lg01/` |
