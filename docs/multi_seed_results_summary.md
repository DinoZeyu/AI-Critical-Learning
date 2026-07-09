# Multi-Seed Results Summary

This file summarizes seed 22, 42, and 62 results after the project reorganization. All statistics below are computed from the current `metrics.json` and `history.csv` files.

## Completeness

| Seed | Gold evaluators | Noise baselines | GGCL/sweep | Supplemental ablation |
| --- | ---: | ---: | ---: | ---: |
| `22` | 2 | 14 | 105 | 42 |
| `42` | 2 | 14 | 105 | 42 |
| `62` | 2 | 14 | 105 | 42 |

## Best GGCL By Dataset And Noise Setting

Each row chooses the best validation-selected GGCL run within each seed for the same dataset/noise setting. Accuracy columns report mean +/- sample standard deviation across the three seeds; recovery is computed once from the mean selected accuracies.

| Dataset | Noise setting | Noise baseline Sel. | Best GGCL Sel. | Best GGCL Peak | Best GGCL Final | Recovery | Best runs by seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| STL | `blur_3p0` | 0.2553 +/- 0.0294 | 0.6397 +/- 0.0119 | 0.6408 +/- 0.0116 | 0.6245 +/- 0.0382 | 87.82% | `22:blur_3p0_beta09_lg0075`; `42:blur_3p0_method_beta09_lg015`; `62:blur_3p0_method_beta09_lg015` |
| STL | `brightness_0p75` | 0.6721 +/- 0.0189 | 0.7368 +/- 0.0093 | 0.7423 +/- 0.0081 | 0.7294 +/- 0.0131 | 307.93% | `22:brightness_0p75_beta02_lg01`; `42:brightness_0p75_method_beta09_lg01`; `62:brightness_0p75_beta09_lg0075` |
| STL | `gaussian_30p0` | 0.4394 +/- 0.0494 | 0.6842 +/- 0.0249 | 0.6926 +/- 0.0184 | 0.6701 +/- 0.0072 | 96.51% | `22:gaussian_30p0_beta05_lg02`; `42:gaussian_30p0_method_beta05_lg015`; `62:gaussian_30p0_method_beta05_lg015` |
| STL | `label_shuffle_0p2` | 0.6206 +/- 0.0076 | 0.7190 +/- 0.0017 | 0.7204 +/- 0.0041 | 0.6981 +/- 0.0124 | 135.75% | `22:label_shuffle_0p2_beta05_lg03`; `42:label_shuffle_0p2_method_beta05_lg025`; `62:label_shuffle_0p2_method_beta05_lg025` |
| STL | `blur_3p0 + label_shuffle_0p2` | 0.2547 +/- 0.0182 | 0.6388 +/- 0.0125 | 0.6423 +/- 0.0072 | 0.6241 +/- 0.0179 | 87.63% | `22:blur_3p0_label_shuffle_0p2_beta09_lg025`; `42:blur_3p0_label_shuffle_0p2_beta09_lg025`; `62:blur_3p0_label_shuffle_0p2_compromise_beta09_lg02` |
| STL | `brightness_0p75 + label_shuffle_0p2` | 0.6004 +/- 0.0217 | 0.7171 +/- 0.0036 | 0.7188 +/- 0.0037 | 0.7022 +/- 0.0142 | 125.86% | `22:brightness_0p75_label_shuffle_0p2_feature_best_beta09_lg01`; `42:brightness_0p75_label_shuffle_0p2_beta085_lg01`; `62:brightness_0p75_label_shuffle_0p2_compromise_beta09_lg015` |
| STL | `gaussian_30p0 + label_shuffle_0p2` | 0.4444 +/- 0.0580 | 0.6731 +/- 0.0175 | 0.6782 +/- 0.0130 | 0.6547 +/- 0.0200 | 91.96% | `22:gaussian_30p0_label_shuffle_0p2_beta07_lg015`; `42:gaussian_30p0_label_shuffle_0p2_beta06_lg015`; `62:gaussian_30p0_label_shuffle_0p2_beta07_lg015` |
| Flower_102 | `blur_3p0` | 0.4103 +/- 0.0223 | 0.6050 +/- 0.0407 | 0.6176 +/- 0.0345 | 0.6154 +/- 0.0325 | 130.42% | `22:blur_3p0_beta08_lg02`; `42:blur_3p0_method_beta08_lg01`; `62:blur_3p0_beta05_lg01` |
| Flower_102 | `brightness_0p75` | 0.4386 +/- 0.0107 | 0.6382 +/- 0.0063 | 0.6416 +/- 0.0088 | 0.6329 +/- 0.0210 | 164.98% | `22:brightness_0p75_beta09_lg01`; `42:brightness_0p75_method_beta08_lg01`; `62:brightness_0p75_beta08_lg015` |
| Flower_102 | `gaussian_30p0` | 0.5046 +/- 0.0170 | 0.6278 +/- 0.0043 | 0.6355 +/- 0.0103 | 0.6343 +/- 0.0097 | 224.07% | `22:gaussian_30p0_beta02_lg01`; `42:gaussian_30p0_method_beta05_lg015`; `62:gaussian_30p0_beta08_lg01` |
| Flower_102 | `label_shuffle_0p2` | 0.4728 +/- 0.0140 | 0.5881 +/- 0.0139 | 0.5960 +/- 0.0105 | 0.5763 +/- 0.0104 | 132.86% | `22:label_shuffle_0p2_beta10_lg01`; `42:label_shuffle_0p2_method_beta095_lg01`; `62:label_shuffle_0p2_beta08_lg01` |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | 0.3808 +/- 0.0129 | 0.5477 +/- 0.0275 | 0.5547 +/- 0.0165 | 0.5296 +/- 0.0226 | 93.39% | `22:blur_3p0_label_shuffle_0p2_beta08_lg005`; `42:blur_3p0_label_shuffle_0p2_beta07_lg01`; `62:blur_3p0_label_shuffle_0p2_beta08_lg015` |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | 0.3482 +/- 0.0180 | 0.5697 +/- 0.0117 | 0.5697 +/- 0.0117 | 0.5528 +/- 0.0144 | 104.82% | `22:brightness_0p75_label_shuffle_0p2_beta08_lg015`; `42:brightness_0p75_label_shuffle_0p2_feature_best_beta08_lg01`; `62:brightness_0p75_label_shuffle_0p2_compromise_beta09_lg01` |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | 0.4419 +/- 0.0047 | 0.5677 +/- 0.0187 | 0.5734 +/- 0.0137 | 0.5640 +/- 0.0097 | 106.92% | `22:gaussian_30p0_label_shuffle_0p2_beta055_lg015`; `42:gaussian_30p0_label_shuffle_0p2_beta05_lg01`; `62:gaussian_30p0_label_shuffle_0p2_beta08_lg015` |

## Exact Parameter Statistics

Full exact-parameter tables are written separately because they contain every matched run across the three seeds:

- [GGCL exact-run stats](multi_seed_ggcl_exact_run_stats.md) and [CSV](multi_seed_ggcl_exact_run_stats.csv)
- [Supplemental ablation exact-run stats](multi_seed_supplemental_exact_run_stats.md) and [CSV](multi_seed_supplemental_exact_run_stats.csv)
- [Best-by-setting CSV](multi_seed_main_setting_stats.csv)

## Interpretation Notes

- `Selected Acc` is the headline metric selected by clean gold validation loss.
- `Peak Acc` is diagnostic/oracle-style and should not be used as the deployment selection metric.
- `Final Acc` is useful for stability checks and overfitting diagnosis.
- Recovery is normalized by the seed-specific noisy baseline and the shared clean baseline for the same dataset.
