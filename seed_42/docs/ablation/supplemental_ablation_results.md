# Supplemental Ablation Results

This file records the canonical seed 42 supplemental ablation comparison
against the original GGCL selected-best run for each matching dataset and noise
setting.

Protocol:

- Seed: `42`
- Headline metric: `test_accuracy_at_selected_epoch`
- Checkpoint selection: clean gold validation loss
- Gold evaluator path: `seed_42/Gold_Evaluators/<dataset>/model.pt`
- Supplemental result root: `seed_42/Supplemental_Ablation_Results/`
- Original GGCL result root: `seed_42/Experiments_Results/`
- Recovery ratio: `(Sel. - noise_baseline_selected_acc) / (clean_baseline_selected_acc - noise_baseline_selected_acc)`

## Over Original Selected Best

These supplemental ablation runs exceed the original GGCL selected-best run
under the same dataset and train-noise setting.

| Dataset | Noise setting | Supplemental run | beta | lambda_G | Sel. | Peak | Final | Rec. | Original selected best | Original Sel. | Delta |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta10_lg025` | 1.00 | 0.250 | 0.6596 | 0.6596 | 0.6485 | 92.6% | `blur_3p0_label_shuffle_0p2_beta09_lg025` | 0.6500 | +0.0096 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 0.5577 | 0.5791 | 0.5687 | 98.4% | `gaussian_30p0_label_shuffle_0p2_beta05_lg01` | 0.5486 | +0.0092 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta00_lg015` | 0.00 | 0.150 | 0.6341 | 0.6432 | 0.6432 | 200.8% | `gaussian_30p0_method_beta05_lg015` | 0.6316 | +0.0024 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta10_lg025` | 1.00 | 0.250 | 0.7177 | 0.7200 | 0.7112 | 130.5% | `label_shuffle_0p2_method_beta05_lg025` | 0.7173 | +0.0004 |

## Full Setting Summary

For each dataset/noise setting, this table compares the best supplemental
ablation result against the original GGCL selected-best run.

| Dataset | Noise setting | Best supplemental run | Supp. Sel. | Original selected best | Original Sel. | Delta |
|---|---|---|---:|---|---:|---:|
| Flower_102 | `blur_3p0` | `blur_3p0_beta00_lg01` | 0.5834 | `blur_3p0_method_beta08_lg01` | 0.6151 | -0.0318 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta10_lg01` | 0.6274 | `brightness_0p75_method_beta08_lg01` | 0.6329 | -0.0055 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta00_lg015` | 0.6341 | `gaussian_30p0_method_beta05_lg015` | 0.6316 | +0.0024 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta10_lg01` | 0.5278 | `blur_3p0_label_shuffle_0p2_beta07_lg01` | 0.5547 | -0.0269 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta10_lg01` | 0.5186 | `brightness_0p75_label_shuffle_0p2_feature_best_beta08_lg01` | 0.5663 | -0.0476 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta10_lg01` | 0.5577 | `gaussian_30p0_label_shuffle_0p2_beta05_lg01` | 0.5486 | +0.0092 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta10_lg01` | 0.5217 | `label_shuffle_0p2_method_beta095_lg01` | 0.5767 | -0.0550 |
| STL | `blur_3p0` | `blur_3p0_beta10_lg015` | 0.6135 | `blur_3p0_method_beta09_lg015` | 0.6362 | -0.0227 |
| STL | `brightness_0p75` | `brightness_0p75_beta00_lg01` | 0.7138 | `brightness_0p75_method_beta09_lg01` | 0.7435 | -0.0296 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta00_lg015` | 0.6569 | `gaussian_30p0_method_beta05_lg015` | 0.6869 | -0.0300 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta10_lg025` | 0.6596 | `blur_3p0_label_shuffle_0p2_beta09_lg025` | 0.6500 | +0.0096 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta10_lg01` | 0.7065 | `brightness_0p75_label_shuffle_0p2_beta085_lg01` | 0.7154 | -0.0088 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta00_lg015` | 0.6442 | `gaussian_30p0_label_shuffle_0p2_beta06_lg015` | 0.6604 | -0.0162 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta10_lg025` | 0.7177 | `label_shuffle_0p2_method_beta05_lg025` | 0.7173 | +0.0004 |

## Notes

- The canonical supplemental set contains 42 runs: `lambda0`, `beta00`, and
  `beta10` groups for both `STL` and `Flower_102`.
- Four supplemental settings exceed the original selected best in the canonical
  supplemental run.
- The STL `label_shuffle_0p2` margin is very small (`+0.0004`) and should be
  described as effectively tied/slightly higher rather than a strong gain.
- `Peak` is diagnostic only. Headline comparisons should use `Sel.`, which is
  selected by clean gold validation loss.
