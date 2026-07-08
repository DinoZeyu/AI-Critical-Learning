# Flower_102 gaussian_30p0 + label_shuffle_0p2 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `gaussian_30p0 + label_shuffle_0p2`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `hybrid_noise/gaussian_30p0_label_shuffle_0p2`
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
| Noise lower reference | Baseline | `gaussian_30p0 + label_shuffle_0p2` train | 20 | 0.4380 | 0.4502 | 0.4380 | 0.0% |
| Best current method | Gold-guided CL, beta=0.50, lambda_gold=0.200 | `gaussian_30p0 + label_shuffle_0p2` train | 11 | 0.5858 | 0.5858 | 0.5748 | 121.6% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gaussian_30p0_label_shuffle_0p2_beta05_lg02` | 0.50 | 0.200 | 11 | 1.4361 | 0.6269 | 0.5858 | 0.5858 | 11 | 0.5748 | 121.6% |
| `gaussian_30p0_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 11 | 1.4107 | 0.6269 | 0.5858 | 0.5889 | 12 | 0.5748 | 121.6% |
| `gaussian_30p0_label_shuffle_0p2_compromise_beta07_lg015` | 0.70 | 0.150 | 11 | 1.4483 | 0.5970 | 0.5846 | 0.5938 | 12 | 0.5773 | 120.6% |
| `gaussian_30p0_label_shuffle_0p2_beta045_lg015` | 0.45 | 0.150 | 14 | 1.4710 | 0.6119 | 0.5834 | 0.5834 | 14 | 0.5608 | 119.6% |
| `gaussian_30p0_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 11 | 1.4221 | 0.5821 | 0.5828 | 0.5919 | 12 | 0.5907 | 119.1% |
| `gaussian_30p0_label_shuffle_0p2_beta055_lg015` | 0.55 | 0.150 | 11 | 1.4860 | 0.6194 | 0.5754 | 0.5828 | 13 | 0.5816 | 113.1% |
| `gaussian_30p0_label_shuffle_0p2_beta06_lg015` | 0.60 | 0.150 | 11 | 1.4655 | 0.5896 | 0.5754 | 0.5852 | 9 | 0.5791 | 113.1% |
| `gaussian_30p0_label_shuffle_0p2_feature_best_beta05_lg015` | 0.50 | 0.150 | 12 | 1.5113 | 0.5821 | 0.5699 | 0.5779 | 14 | 0.5712 | 108.5% |
| `gaussian_30p0_label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 16 | 1.5028 | 0.6045 | 0.5638 | 0.5803 | 14 | 0.5638 | 103.5% |

## Notes

- Best selected test accuracy for this seed/setting is `0.5858` from `gaussian_30p0_label_shuffle_0p2_beta05_lg02`.
- Noise baseline selected test accuracy is `0.4380`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_62/Noise_Baseline/Flower_102/hybrid_noise/gaussian_30p0_label_shuffle_0p2/` |
| Best GGCL run | `seed_62/Experiments_Results/Flower_102/hybrid_noise/gaussian_label_shuffle/gaussian_hybrid_ablation/gaussian_30p0_label_shuffle_0p2_beta05_lg02/` |
