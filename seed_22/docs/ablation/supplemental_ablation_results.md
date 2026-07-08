# Supplemental Ablation Results

This file records the canonical seed 22 supplemental ablation comparison against the GGCL selected-best run for each matching dataset and noise setting.

Protocol:

- Seed: `22`
- Headline metric: `test_accuracy_at_selected_epoch`
- Checkpoint selection: clean gold validation loss
- Gold evaluator path: `seed_22/Gold_Evaluators/<dataset>/model.pt`
- Supplemental result root: `seed_22/Supplemental_Ablation_Results/`
- Original GGCL result root: `seed_22/Experiments_Results/`
- Recovery ratio: `(Sel. - noise_baseline_selected_acc) / (clean_baseline_selected_acc - noise_baseline_selected_acc)`

## Over Original Selected Best

These supplemental ablation runs exceed the original GGCL selected-best run under the same dataset and train-noise setting.

| Dataset | Noise setting | Supplemental run | beta | lambda_G | Sel. | Peak | Final | Rec. | Original selected best | Original Sel. | Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta00_lg015` | 0.00 | 0.150 | 0.6304 | 0.6371 | 0.6371 | 241.5% | `gaussian_30p0_beta02_lg01` | 0.6231 | +0.0073 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta10_lg025` | 1.00 | 0.250 | 0.6315 | 0.6419 | 0.6419 | 85.3% | `blur_3p0_label_shuffle_0p2_beta09_lg025` | 0.6254 | +0.0062 |

## Full Setting Summary

| Dataset | Noise setting | Best supplemental run | Supp. Sel. | Original selected best | Original Sel. | Delta |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Flower_102 | `blur_3p0` | `blur_3p0_beta10_lg01` | 0.4905 | `blur_3p0_beta08_lg02` | 0.5602 | -0.0696 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta10_lg01` | 0.5932 | `brightness_0p75_beta09_lg01` | 0.6365 | -0.0434 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta00_lg015` | 0.6304 | `gaussian_30p0_beta02_lg01` | 0.6231 | +0.0073 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta10_lg01` | 0.5840 | `label_shuffle_0p2_beta10_lg01` | 0.5840 | +0.0000 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta10_lg01` | 0.5046 | `blur_3p0_label_shuffle_0p2_beta08_lg005` | 0.5174 | -0.0128 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta10_lg01` | 0.5516 | `brightness_0p75_label_shuffle_0p2_beta08_lg015` | 0.5602 | -0.0086 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta10_lg01` | 0.5443 | `gaussian_30p0_label_shuffle_0p2_beta055_lg015` | 0.5687 | -0.0244 |
| STL | `blur_3p0` | `blur_3p0_beta10_lg015` | 0.6465 | `blur_3p0_beta09_lg0075` | 0.6531 | -0.0065 |
| STL | `brightness_0p75` | `brightness_0p75_beta10_lg01` | 0.7000 | `brightness_0p75_beta02_lg01` | 0.7262 | -0.0262 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta00_lg015` | 0.6950 | `gaussian_30p0_beta05_lg02` | 0.7077 | -0.0127 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta10_lg025` | 0.7146 | `label_shuffle_0p2_beta05_lg03` | 0.7208 | -0.0062 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta10_lg025` | 0.6315 | `blur_3p0_label_shuffle_0p2_beta09_lg025` | 0.6254 | +0.0062 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta10_lg01` | 0.6946 | `brightness_0p75_label_shuffle_0p2_feature_best_beta09_lg01` | 0.7212 | -0.0265 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta00_lg015` | 0.6581 | `gaussian_30p0_label_shuffle_0p2_beta07_lg015` | 0.6931 | -0.0350 |

## Notes

- The canonical supplemental set contains 42 runs: `lambda0`, `beta00`, and `beta10` groups for both datasets and all seven noise settings.
- `2` supplemental runs exceed the original selected best for seed 22.
- `Peak` is diagnostic only. Headline comparisons should use `Sel.`, selected by clean gold validation loss.
