# Flower_102 label_shuffle_0p2 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `label_shuffle_0p2`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `label_noise/label_shuffle_0p2`
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
| Noise lower reference | Baseline | `label_shuffle_0p2` train | 16 | 0.4575 | 0.4801 | 0.4472 | 0.0% |
| Best current method | Gold-guided CL, beta=1.00, lambda_gold=0.100 | `label_shuffle_0p2` train | 15 | 0.5840 | 0.5840 | 0.5767 | 124.0% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 15 | 1.6416 | 0.6045 | 0.5840 | 0.5840 | 15 | 0.5767 | 124.0% |
| `label_shuffle_0p2_beta09_lg01` | 0.90 | 0.100 | 15 | 1.6639 | 0.5373 | 0.5754 | 0.5754 | 15 | 0.5736 | 115.6% |
| `label_shuffle_0p2_beta095_lg015` | 0.95 | 0.150 | 15 | 1.6479 | 0.5522 | 0.5730 | 0.5742 | 18 | 0.5742 | 113.2% |
| `label_shuffle_0p2_beta08_lg01` | 0.80 | 0.100 | 15 | 1.6785 | 0.5672 | 0.5638 | 0.5822 | 17 | 0.5620 | 104.2% |
| `label_shuffle_0p2_beta02_lg01` | 0.20 | 0.100 | 15 | 1.7216 | 0.5672 | 0.5492 | 0.5547 | 12 | 0.5431 | 89.8% |
| `label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 9 | 1.7310 | 0.5597 | 0.5461 | 0.5559 | 11 | 0.5486 | 86.8% |
| `label_shuffle_0p2_beta095_lg005` | 0.95 | 0.050 | 9 | 1.7840 | 0.5299 | 0.5339 | 0.5547 | 11 | 0.5492 | 74.9% |
| `label_shuffle_0p2_method_beta095_lg01` | 0.95 | 0.100 | 5 | 1.7909 | 0.5075 | 0.4942 | 0.5333 | 8 | 0.5333 | 35.9% |

## Notes

- Best selected test accuracy for this seed/setting is `0.5840` from `label_shuffle_0p2_beta10_lg01`.
- Noise baseline selected test accuracy is `0.4575`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_22/Noise_Baseline/Flower_102/label_noise/label_shuffle_0p2/` |
| Best GGCL run | `seed_22/Experiments_Results/Flower_102/label_noise/label_shuffle/label_ablations/label_shuffle_0p2_beta10_lg01/` |
