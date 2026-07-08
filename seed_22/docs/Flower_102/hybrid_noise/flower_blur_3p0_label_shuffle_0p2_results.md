# Flower_102 blur_3p0 + label_shuffle_0p2 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `blur_3p0 + label_shuffle_0p2`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `hybrid_noise/blur_3p0_label_shuffle_0p2`
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
| Noise lower reference | Baseline | `blur_3p0 + label_shuffle_0p2` train | 16 | 0.3879 | 0.3879 | 0.3873 | 0.0% |
| Best current method | Gold-guided CL, beta=0.80, lambda_gold=0.050 | `blur_3p0 + label_shuffle_0p2` train | 15 | 0.5174 | 0.5382 | 0.5235 | 75.4% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blur_3p0_label_shuffle_0p2_beta08_lg005` | 0.80 | 0.050 | 15 | 1.7819 | 0.5373 | 0.5174 | 0.5382 | 17 | 0.5235 | 75.4% |
| `blur_3p0_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 4 | 1.8819 | 0.5448 | 0.4728 | 0.4893 | 7 | 0.4893 | 49.5% |
| `blur_3p0_label_shuffle_0p2_beta075_lg01` | 0.75 | 0.100 | 4 | 1.9386 | 0.5149 | 0.4673 | 0.4856 | 7 | 0.4856 | 46.3% |
| `blur_3p0_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 4 | 1.9104 | 0.5299 | 0.4673 | 0.4783 | 6 | 0.4716 | 46.3% |
| `blur_3p0_label_shuffle_0p2_compromise_beta09_lg01` | 0.90 | 0.100 | 4 | 1.9284 | 0.5448 | 0.4655 | 0.4832 | 7 | 0.4832 | 45.2% |
| `blur_3p0_label_shuffle_0p2_feature_best_beta08_lg01` | 0.80 | 0.100 | 4 | 1.9550 | 0.5000 | 0.4637 | 0.4850 | 7 | 0.4850 | 44.1% |
| `blur_3p0_label_shuffle_0p2_beta07_lg01` | 0.70 | 0.100 | 4 | 1.9966 | 0.5000 | 0.4618 | 0.4740 | 6 | 0.4710 | 43.1% |

## Notes

- Best selected test accuracy for this seed/setting is `0.5174` from `blur_3p0_label_shuffle_0p2_beta08_lg005`.
- Noise baseline selected test accuracy is `0.3879`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_22/Noise_Baseline/Flower_102/hybrid_noise/blur_3p0_label_shuffle_0p2/` |
| Best GGCL run | `seed_22/Experiments_Results/Flower_102/hybrid_noise/blur_label_shuffle/blur_hybrid_ablation/blur_3p0_label_shuffle_0p2_beta08_lg005/` |
