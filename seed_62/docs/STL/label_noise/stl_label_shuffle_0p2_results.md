# STL label_shuffle_0p2 Experiment Results

This file summarizes seed 62 train-noise / clean-test results for `label_shuffle_0p2`.

Protocol:

- Dataset: `STL`
- Noise setting: `label_noise/label_shuffle_0p2`
- Seed: `62`
- Validation source: stratified split from root `Image_Data/Gold_Data/<dataset>`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Gold evaluator path: `seed_62/Gold_Evaluators/STL/model.pt`
- Headline metric: `test_accuracy_at_selected_epoch`; peak test accuracy is diagnostic.
- Recovery ratio: `(method_sel - noise_baseline_sel) / (clean_baseline_sel - noise_baseline_sel)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `label_shuffle_0p2` train | 12 | 0.6273 | 0.6331 | 0.6242 | 0.0% |
| Best current method | Gold-guided CL, beta=0.50, lambda_gold=0.250 | `label_shuffle_0p2` train | 13 | 0.7188 | 0.7188 | 0.6838 | 139.2% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `label_shuffle_0p2_method_beta05_lg025` | 0.50 | 0.250 | 13 | 0.5981 | 0.8097 | 0.7188 | 0.7188 | 13 | 0.6838 | 139.2% |
| `label_shuffle_0p2_beta05_lg015` | 0.50 | 0.150 | 13 | 0.5910 | 0.8065 | 0.7123 | 0.7123 | 13 | 0.6508 | 129.2% |
| `label_shuffle_0p2_beta05_lg02` | 0.50 | 0.200 | 6 | 0.7092 | 0.7645 | 0.6965 | 0.6996 | 9 | 0.6996 | 105.3% |
| `label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 13 | 0.6573 | 0.8000 | 0.6938 | 0.6954 | 11 | 0.6612 | 101.2% |
| `label_shuffle_0p2_beta02_lg01` | 0.20 | 0.100 | 13 | 0.7123 | 0.7581 | 0.6838 | 0.6869 | 11 | 0.6469 | 86.0% |
| `label_shuffle_0p2_beta05_lg005` | 0.50 | 0.050 | 13 | 0.7243 | 0.7548 | 0.6812 | 0.6827 | 11 | 0.6642 | 81.9% |
| `label_shuffle_0p2_beta00_lg01` | 0.00 | 0.100 | 13 | 0.7494 | 0.7710 | 0.6704 | 0.6719 | 10 | 0.6638 | 65.5% |
| `label_shuffle_0p2_beta05_lg03` | 0.50 | 0.300 | 2 | 0.7677 | 0.7581 | 0.6669 | 0.6692 | 5 | 0.6692 | 60.2% |
| `label_shuffle_0p2_beta01_lg01` | 0.10 | 0.100 | 4 | 0.8906 | 0.6968 | 0.6235 | 0.6650 | 6 | 0.6465 | -5.8% |

## Notes

- Best selected test accuracy for this seed/setting is `0.7188` from `label_shuffle_0p2_method_beta05_lg025`.
- Noise baseline selected test accuracy is `0.6273`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_62/Noise_Baseline/STL/label_noise/label_shuffle_0p2/` |
| Best GGCL run | `seed_62/Experiments_Results/STL/label_noise/label_shuffle/label_shuffle_0p2_method_beta05_lg025/` |
