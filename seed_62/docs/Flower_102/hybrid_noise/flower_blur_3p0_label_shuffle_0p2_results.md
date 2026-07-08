# Flower_102 blur_3p0 + label_shuffle_0p2 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `blur_3p0 + label_shuffle_0p2`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `hybrid_noise/blur_3p0_label_shuffle_0p2`
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
| Noise lower reference | Baseline | `blur_3p0 + label_shuffle_0p2` train | 11 | 0.3659 | 0.3885 | 0.3830 | 0.0% |
| Best current method | Gold-guided CL, beta=0.80, lambda_gold=0.150 | `blur_3p0 + label_shuffle_0p2` train | 14 | 0.5712 | 0.5712 | 0.5547 | 106.0% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blur_3p0_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 14 | 1.4705 | 0.6119 | 0.5712 | 0.5712 | 14 | 0.5547 | 106.0% |
| `blur_3p0_label_shuffle_0p2_beta075_lg01` | 0.75 | 0.100 | 14 | 1.5342 | 0.6045 | 0.5687 | 0.5687 | 14 | 0.5589 | 104.7% |
| `blur_3p0_label_shuffle_0p2_beta07_lg01` | 0.70 | 0.100 | 14 | 1.5555 | 0.5672 | 0.5614 | 0.5614 | 14 | 0.5498 | 100.9% |
| `blur_3p0_label_shuffle_0p2_feature_best_beta08_lg01` | 0.80 | 0.100 | 14 | 1.6046 | 0.5597 | 0.5535 | 0.5535 | 11 | 0.5535 | 96.8% |
| `blur_3p0_label_shuffle_0p2_compromise_beta09_lg01` | 0.90 | 0.100 | 11 | 1.5719 | 0.5299 | 0.5522 | 0.5687 | 14 | 0.5687 | 96.2% |
| `blur_3p0_label_shuffle_0p2_beta08_lg005` | 0.80 | 0.050 | 17 | 1.6096 | 0.5299 | 0.5449 | 0.5651 | 19 | 0.5296 | 92.4% |
| `blur_3p0_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 7 | 1.6130 | 0.5746 | 0.5376 | 0.5431 | 10 | 0.5431 | 88.6% |

## Notes

- Best selected test accuracy for this seed/setting is `0.5712` from `blur_3p0_label_shuffle_0p2_beta08_lg015`.
- Noise baseline selected test accuracy is `0.3659`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_62/Noise_Baseline/Flower_102/hybrid_noise/blur_3p0_label_shuffle_0p2/` |
| Best GGCL run | `seed_62/Experiments_Results/Flower_102/hybrid_noise/blur_label_shuffle/blur_hybrid_ablation/blur_3p0_label_shuffle_0p2_beta08_lg015/` |
