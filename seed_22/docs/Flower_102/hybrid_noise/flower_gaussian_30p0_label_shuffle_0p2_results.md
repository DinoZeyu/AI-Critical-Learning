# Flower_102 gaussian_30p0 + label_shuffle_0p2 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `gaussian_30p0 + label_shuffle_0p2`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `hybrid_noise/gaussian_30p0_label_shuffle_0p2`
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
| Noise lower reference | Baseline | `gaussian_30p0 + label_shuffle_0p2` train | 16 | 0.4404 | 0.4404 | 0.4319 | 0.0% |
| Best current method | Gold-guided CL, beta=0.55, lambda_gold=0.150 | `gaussian_30p0 + label_shuffle_0p2` train | 15 | 0.5687 | 0.5687 | 0.5614 | 107.7% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gaussian_30p0_label_shuffle_0p2_beta055_lg015` | 0.55 | 0.150 | 15 | 1.6173 | 0.5597 | 0.5687 | 0.5687 | 15 | 0.5614 | 107.7% |
| `gaussian_30p0_label_shuffle_0p2_beta045_lg015` | 0.45 | 0.150 | 15 | 1.6373 | 0.5597 | 0.5577 | 0.5577 | 15 | 0.5510 | 98.5% |
| `gaussian_30p0_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 14 | 1.6243 | 0.5522 | 0.5553 | 0.5693 | 15 | 0.5455 | 96.4% |
| `gaussian_30p0_label_shuffle_0p2_compromise_beta07_lg015` | 0.70 | 0.150 | 14 | 1.6207 | 0.5597 | 0.5522 | 0.5718 | 15 | 0.5406 | 93.8% |
| `gaussian_30p0_label_shuffle_0p2_feature_best_beta05_lg015` | 0.50 | 0.150 | 15 | 1.6474 | 0.5448 | 0.5522 | 0.5589 | 12 | 0.5571 | 93.8% |
| `gaussian_30p0_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 14 | 1.6592 | 0.5522 | 0.5522 | 0.5663 | 16 | 0.5376 | 93.8% |
| `gaussian_30p0_label_shuffle_0p2_beta05_lg02` | 0.50 | 0.200 | 14 | 1.6641 | 0.5597 | 0.5516 | 0.5767 | 16 | 0.5412 | 93.3% |
| `gaussian_30p0_label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 15 | 1.7556 | 0.5373 | 0.5480 | 0.5492 | 16 | 0.5406 | 90.3% |
| `gaussian_30p0_label_shuffle_0p2_beta06_lg015` | 0.60 | 0.150 | 9 | 1.7240 | 0.5299 | 0.5345 | 0.5553 | 12 | 0.5553 | 79.0% |

## Notes

- Best selected test accuracy for this seed/setting is `0.5687` from `gaussian_30p0_label_shuffle_0p2_beta055_lg015`.
- Noise baseline selected test accuracy is `0.4404`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_22/Noise_Baseline/Flower_102/hybrid_noise/gaussian_30p0_label_shuffle_0p2/` |
| Best GGCL run | `seed_22/Experiments_Results/Flower_102/hybrid_noise/gaussian_label_shuffle/gaussian_hybrid_ablation/gaussian_30p0_label_shuffle_0p2_beta055_lg015/` |
