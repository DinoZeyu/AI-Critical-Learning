# STL label_shuffle_0p2 Experiment Results

This file summarizes seed 22 train-noise / clean-test results for `label_shuffle_0p2`.

Protocol:

- Dataset: `STL`
- Noise setting: `label_noise/label_shuffle_0p2`
- Seed: `22`
- Validation source: stratified split from root `Image_Data/Gold_Data/<dataset>`
- Selection metric: `val-loss`
- Early stopping patience: `3`
- Gold evaluator path: `seed_22/Gold_Evaluators/STL/model.pt`
- Headline metric: `test_accuracy_at_selected_epoch`; peak test accuracy is diagnostic.
- Recovery ratio: `(method_sel - noise_baseline_sel) / (clean_baseline_sel - noise_baseline_sel)`

## Main Comparison

| Setting | Method | Train Data | Selected Epoch | Selected Test Acc | Peak Test Acc | Final Test Acc | Recovery Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean upper reference | Baseline | Clean train | 13 | 0.6931 | 0.6931 | 0.6662 | 100.0% |
| Noise lower reference | Baseline | `label_shuffle_0p2` train | 13 | 0.6223 | 0.6358 | 0.6358 | 0.0% |
| Best current method | Gold-guided CL, beta=0.50, lambda_gold=0.300 | `label_shuffle_0p2` train | 8 | 0.7208 | 0.7250 | 0.7065 | 139.1% |

## Gold-Guided CL Runs

| Run | beta | lambda_gold | Selected Epoch | Val Loss | Val Acc | Selected Test Acc | Peak Test Acc | Peak Epoch | Final Test Acc | Recovery Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `label_shuffle_0p2_beta05_lg03` | 0.50 | 0.300 | 8 | 0.5892 | 0.8129 | 0.7208 | 0.7250 | 10 | 0.7065 | 139.1% |
| `label_shuffle_0p2_method_beta05_lg025` | 0.50 | 0.250 | 14 | 0.5923 | 0.8161 | 0.7196 | 0.7300 | 12 | 0.6985 | 137.5% |
| `label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 13 | 0.6402 | 0.7903 | 0.7173 | 0.7173 | 13 | 0.6877 | 134.2% |
| `label_shuffle_0p2_beta05_lg02` | 0.50 | 0.200 | 8 | 0.6053 | 0.8129 | 0.7077 | 0.7177 | 10 | 0.6950 | 120.7% |
| `label_shuffle_0p2_beta05_lg015` | 0.50 | 0.150 | 9 | 0.6273 | 0.7968 | 0.7054 | 0.7177 | 12 | 0.7177 | 117.4% |
| `label_shuffle_0p2_beta05_lg005` | 0.50 | 0.050 | 13 | 0.6855 | 0.7742 | 0.6935 | 0.7096 | 14 | 0.7058 | 100.5% |
| `label_shuffle_0p2_beta00_lg01` | 0.00 | 0.100 | 8 | 0.7728 | 0.7645 | 0.6904 | 0.6904 | 8 | 0.6750 | 96.2% |
| `label_shuffle_0p2_beta02_lg01` | 0.20 | 0.100 | 11 | 0.7333 | 0.7806 | 0.6873 | 0.6938 | 10 | 0.6808 | 91.8% |
| `label_shuffle_0p2_beta01_lg01` | 0.10 | 0.100 | 12 | 0.7570 | 0.7548 | 0.6858 | 0.7027 | 14 | 0.6769 | 89.7% |

## Notes

- Best selected test accuracy for this seed/setting is `0.7208` from `label_shuffle_0p2_beta05_lg03`.
- Noise baseline selected test accuracy is `0.6223`; clean baseline selected test accuracy is `0.6931`.
- `Peak Test Acc` is diagnostic/oracle-style; use `Selected Test Acc` for headline comparisons.

## Source Result Folders

| Result | Path |
| --- | --- |
| Clean baseline | `Clean_Baseline/STL/` |
| Noise baseline | `seed_22/Noise_Baseline/STL/label_noise/label_shuffle_0p2/` |
| Best GGCL run | `seed_22/Experiments_Results/STL/label_noise/label_shuffle/label_ablation/label_shuffle_0p2_beta05_lg03/` |
