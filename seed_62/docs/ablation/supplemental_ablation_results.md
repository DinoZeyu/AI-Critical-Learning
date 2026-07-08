# Supplemental Ablation Results

This file records the canonical seed 62 supplemental ablation comparison against the GGCL selected-best run for each matching dataset and noise setting.

Protocol:

- Seed: `62`
- Headline metric: `test_accuracy_at_selected_epoch`
- Checkpoint selection: clean gold validation loss
- Gold evaluator path: `seed_62/Gold_Evaluators/<dataset>/model.pt`
- Supplemental result root: `seed_62/Supplemental_Ablation_Results/`
- Original GGCL result root: `seed_62/Experiments_Results/`
- Recovery ratio: `(Sel. - noise_baseline_selected_acc) / (clean_baseline_selected_acc - noise_baseline_selected_acc)`

## Over Original Selected Best

These supplemental ablation runs exceed the original GGCL selected-best run under the same dataset and train-noise setting.

| Dataset | Noise setting | Supplemental run | beta | lambda_G | Sel. | Peak | Final | Rec. | Original selected best | Original Sel. | Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| STL | `gaussian_30p0` | `gaussian_30p0_beta00_lg015` | 0.00 | 0.150 | 0.6800 | 0.6800 | 0.6604 | 95.8% | `gaussian_30p0_method_beta05_lg015` | 0.6581 | +0.0219 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta10_lg025` | 1.00 | 0.250 | 0.7242 | 0.7242 | 0.7135 | 147.4% | `label_shuffle_0p2_method_beta05_lg025` | 0.7188 | +0.0054 |
| STL | `blur_3p0` | `blur_3p0_beta10_lg015` | 1.00 | 0.150 | 0.6346 | 0.6346 | 0.6027 | 87.5% | `blur_3p0_method_beta09_lg015` | 0.6300 | +0.0046 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 0.5748 | 0.5748 | 0.5693 | 107.9% | `blur_3p0_label_shuffle_0p2_beta08_lg015` | 0.5712 | +0.0037 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 0.5877 | 0.5962 | 0.5822 | 123.1% | `gaussian_30p0_label_shuffle_0p2_beta08_lg015` | 0.5858 | +0.0018 |

## Full Setting Summary

| Dataset | Noise setting | Best supplemental run | Supp. Sel. | Original selected best | Original Sel. | Delta |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Flower_102 | `blur_3p0` | `blur_3p0_beta10_lg01` | 0.6145 | `blur_3p0_beta05_lg01` | 0.6396 | -0.0250 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta10_lg01` | 0.6127 | `brightness_0p75_beta08_lg015` | 0.6451 | -0.0324 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta00_lg015` | 0.6225 | `gaussian_30p0_beta08_lg01` | 0.6286 | -0.0061 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta00_lg01` | 0.5809 | `label_shuffle_0p2_beta08_lg01` | 0.6035 | -0.0226 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta10_lg01` | 0.5748 | `blur_3p0_label_shuffle_0p2_beta08_lg015` | 0.5712 | +0.0037 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta10_lg01` | 0.5785 | `brightness_0p75_label_shuffle_0p2_compromise_beta09_lg01` | 0.5828 | -0.0043 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta10_lg01` | 0.5877 | `gaussian_30p0_label_shuffle_0p2_beta08_lg015` | 0.5858 | +0.0018 |
| STL | `blur_3p0` | `blur_3p0_beta10_lg015` | 0.6346 | `blur_3p0_method_beta09_lg015` | 0.6300 | +0.0046 |
| STL | `brightness_0p75` | `brightness_0p75_beta00_lg01` | 0.7185 | `brightness_0p75_beta09_lg0075` | 0.7408 | -0.0223 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta00_lg015` | 0.6800 | `gaussian_30p0_method_beta05_lg015` | 0.6581 | +0.0219 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta10_lg025` | 0.7242 | `label_shuffle_0p2_method_beta05_lg025` | 0.7188 | +0.0054 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta10_lg025` | 0.6250 | `blur_3p0_label_shuffle_0p2_compromise_beta09_lg02` | 0.6412 | -0.0162 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta10_lg01` | 0.6865 | `brightness_0p75_label_shuffle_0p2_compromise_beta09_lg015` | 0.7146 | -0.0281 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta00_lg015` | 0.6531 | `gaussian_30p0_label_shuffle_0p2_beta07_lg015` | 0.6658 | -0.0127 |

## Notes

- The canonical supplemental set contains 42 runs: `lambda0`, `beta00`, and `beta10` groups for both datasets and all seven noise settings.
- `5` supplemental runs exceed the original selected best for seed 62.
- `Peak` is diagnostic only. Headline comparisons should use `Sel.`, selected by clean gold validation loss.
