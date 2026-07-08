# Flower_102 label_shuffle_0p2 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `label_shuffle_0p2`.

Protocol:

- Dataset: `Flower_102`
- Noise setting: `label_noise/label_shuffle_0p2`
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
| Noise lower reference | Baseline | `label_shuffle_0p2` train | 17 | 0.4850 | 0.4899 | 0.4899 | 0.0% |
| Best current method | Gold-guided CL, beta=0.80, lambda_gold=0.100 | `label_shuffle_0p2` train | 14 | 0.6035 | 0.6035 | 0.5864 | 159.0% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `label_shuffle_0p2_beta08_lg01` | 0.80 | 0.100 | 14 | 1.4643 | 0.5896 | 0.6035 | 0.6035 | 14 | 0.5864 | 159.0% |
| `label_shuffle_0p2_beta095_lg015` | 0.95 | 0.150 | 14 | 1.3918 | 0.6418 | 0.5999 | 0.6017 | 15 | 0.5901 | 154.1% |
| `label_shuffle_0p2_beta095_lg005` | 0.95 | 0.050 | 14 | 1.4795 | 0.6269 | 0.5968 | 0.5968 | 14 | 0.5938 | 150.0% |
| `label_shuffle_0p2_beta09_lg01` | 0.90 | 0.100 | 13 | 1.4603 | 0.5746 | 0.5877 | 0.6011 | 15 | 0.5822 | 137.7% |
| `label_shuffle_0p2_method_beta095_lg01` | 0.95 | 0.100 | 11 | 1.4425 | 0.5746 | 0.5809 | 0.5913 | 12 | 0.5895 | 128.7% |
| `label_shuffle_0p2_beta02_lg01` | 0.20 | 0.100 | 13 | 1.5920 | 0.6194 | 0.5797 | 0.5852 | 15 | 0.5614 | 127.0% |
| `label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 12 | 1.5106 | 0.6343 | 0.5797 | 0.5877 | 14 | 0.5785 | 127.0% |
| `label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 10 | 1.4498 | 0.6045 | 0.5773 | 0.5901 | 12 | 0.5816 | 123.8% |

## Notes

- Best selected test accuracy for this seed/setting is `0.6035` from `label_shuffle_0p2_beta08_lg01`.
- Noise baseline selected test accuracy is `0.4850`; clean baseline selected test accuracy is `0.5596`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/Flower_102/` |
| Noise baseline | `seed_62/Noise_Baseline/Flower_102/label_noise/label_shuffle_0p2/` |
| Best GGCL run | `seed_62/Experiments_Results/Flower_102/label_noise/label_shuffle/label_ablations/label_shuffle_0p2_beta08_lg01/` |
