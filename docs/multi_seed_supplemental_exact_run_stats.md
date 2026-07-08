# Multi-Seed Supplemental Ablation Exact-Run Stats

Each row is one supplemental ablation setting matched by dataset, noise setting, beta, lambda_gold, and run name.

Values are mean +/- sample standard deviation over seeds 22, 42, and 62.

| Dataset | Noise setting | Run | beta | lambda_G | Sel. Acc | Peak Acc | Final Acc | Selected Epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STL | `blur_3p0` | `blur_3p0_beta09_lg00` | 0.90 | 0.000 | 0.5233 +/- 0.0171 | 0.5313 +/- 0.0041 | 0.5078 +/- 0.0189 | 1.67 +/- 0.58 |
| STL | `brightness_0p75` | `brightness_0p75_beta09_lg00` | 0.90 | 0.000 | 0.6836 +/- 0.0142 | 0.6900 +/- 0.0247 | 0.6840 +/- 0.0300 | 5.00 +/- 3.61 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta05_lg00` | 0.50 | 0.000 | 0.6191 +/- 0.0139 | 0.6363 +/- 0.0112 | 0.6195 +/- 0.0193 | 7.67 +/- 1.53 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta05_lg00` | 0.50 | 0.000 | 0.6518 +/- 0.0353 | 0.6582 +/- 0.0260 | 0.6537 +/- 0.0201 | 5.00 +/- 2.65 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta09_lg00` | 0.90 | 0.000 | 0.5285 +/- 0.0077 | 0.5285 +/- 0.0077 | 0.5009 +/- 0.0152 | 3.67 +/- 1.53 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta085_lg00` | 0.85 | 0.000 | 0.6431 +/- 0.0052 | 0.6601 +/- 0.0110 | 0.6486 +/- 0.0168 | 3.67 +/- 2.52 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta06_lg00` | 0.60 | 0.000 | 0.6155 +/- 0.0241 | 0.6155 +/- 0.0241 | 0.5881 +/- 0.0062 | 9.00 +/- 1.73 |
| Flower_102 | `blur_3p0` | `blur_3p0_beta08_lg00` | 0.80 | 0.000 | 0.5087 +/- 0.0474 | 0.5339 +/- 0.0599 | 0.5211 +/- 0.0775 | 7.67 +/- 3.21 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta08_lg00` | 0.80 | 0.000 | 0.5877 +/- 0.0192 | 0.5952 +/- 0.0116 | 0.5940 +/- 0.0135 | 11.00 +/- 2.00 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta05_lg00` | 0.50 | 0.000 | 0.5801 +/- 0.0034 | 0.5978 +/- 0.0121 | 0.5946 +/- 0.0165 | 9.67 +/- 1.15 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta095_lg00` | 0.95 | 0.000 | 0.5032 +/- 0.0336 | 0.5321 +/- 0.0228 | 0.5254 +/- 0.0124 | 8.33 +/- 2.89 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta07_lg00` | 0.70 | 0.000 | 0.4850 +/- 0.0169 | 0.4911 +/- 0.0142 | 0.4671 +/- 0.0199 | 10.33 +/- 0.58 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta08_lg00` | 0.80 | 0.000 | 0.4814 +/- 0.0101 | 0.5074 +/- 0.0190 | 0.5019 +/- 0.0201 | 6.67 +/- 1.53 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta05_lg00` | 0.50 | 0.000 | 0.5101 +/- 0.0254 | 0.5168 +/- 0.0204 | 0.5146 +/- 0.0167 | 8.67 +/- 3.21 |
| STL | `blur_3p0` | `blur_3p0_beta00_lg015` | 0.00 | 0.150 | 0.6019 +/- 0.0098 | 0.6026 +/- 0.0088 | 0.5774 +/- 0.0205 | 1.67 +/- 0.58 |
| STL | `brightness_0p75` | `brightness_0p75_beta00_lg01` | 0.00 | 0.100 | 0.7104 +/- 0.0103 | 0.7245 +/- 0.0101 | 0.7156 +/- 0.0052 | 10.33 +/- 1.53 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta00_lg015` | 0.00 | 0.150 | 0.6773 +/- 0.0192 | 0.6854 +/- 0.0187 | 0.6701 +/- 0.0323 | 10.00 +/- 1.73 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta00_lg025` | 0.00 | 0.250 | 0.6803 +/- 0.0246 | 0.6927 +/- 0.0146 | 0.6827 +/- 0.0195 | 10.00 +/- 5.20 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta00_lg025` | 0.00 | 0.250 | 0.5944 +/- 0.0219 | 0.6032 +/- 0.0146 | 0.5642 +/- 0.0152 | 2.67 +/- 2.08 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta00_lg01` | 0.00 | 0.100 | 0.6709 +/- 0.0118 | 0.6721 +/- 0.0120 | 0.6556 +/- 0.0246 | 10.67 +/- 2.52 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta00_lg015` | 0.00 | 0.150 | 0.6518 +/- 0.0070 | 0.6528 +/- 0.0075 | 0.6259 +/- 0.0306 | 12.00 +/- 1.73 |
| Flower_102 | `blur_3p0` | `blur_3p0_beta00_lg01` | 0.00 | 0.100 | 0.5392 +/- 0.0532 | 0.5695 +/- 0.0415 | 0.5410 +/- 0.0718 | 7.67 +/- 3.51 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta00_lg01` | 0.00 | 0.100 | 0.5647 +/- 0.0545 | 0.5889 +/- 0.0494 | 0.5856 +/- 0.0513 | 8.00 +/- 3.61 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta00_lg015` | 0.00 | 0.150 | 0.6290 +/- 0.0059 | 0.6365 +/- 0.0070 | 0.6365 +/- 0.0070 | 10.67 +/- 1.53 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta00_lg01` | 0.00 | 0.100 | 0.5388 +/- 0.0386 | 0.5484 +/- 0.0292 | 0.5347 +/- 0.0115 | 11.00 +/- 5.29 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta00_lg01` | 0.00 | 0.100 | 0.5066 +/- 0.0134 | 0.5205 +/- 0.0227 | 0.5054 +/- 0.0055 | 11.00 +/- 5.29 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta00_lg01` | 0.00 | 0.100 | 0.5135 +/- 0.0481 | 0.5258 +/- 0.0346 | 0.4950 +/- 0.0209 | 9.00 +/- 4.58 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta00_lg01` | 0.00 | 0.100 | 0.5300 +/- 0.0381 | 0.5418 +/- 0.0302 | 0.5410 +/- 0.0290 | 10.67 +/- 5.13 |
| STL | `blur_3p0` | `blur_3p0_beta10_lg015` | 1.00 | 0.150 | 0.6315 +/- 0.0168 | 0.6337 +/- 0.0133 | 0.6121 +/- 0.0087 | 1.67 +/- 0.58 |
| STL | `brightness_0p75` | `brightness_0p75_beta10_lg01` | 1.00 | 0.100 | 0.7017 +/- 0.0069 | 0.7179 +/- 0.0046 | 0.7017 +/- 0.0322 | 6.33 +/- 1.15 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta10_lg015` | 1.00 | 0.150 | 0.6540 +/- 0.0150 | 0.6579 +/- 0.0132 | 0.6521 +/- 0.0150 | 4.67 +/- 0.58 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta10_lg025` | 1.00 | 0.250 | 0.7188 +/- 0.0049 | 0.7197 +/- 0.0046 | 0.7117 +/- 0.0016 | 10.67 +/- 2.08 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta10_lg025` | 1.00 | 0.250 | 0.6387 +/- 0.0184 | 0.6438 +/- 0.0149 | 0.6351 +/- 0.0177 | 2.33 +/- 1.53 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 0.6959 +/- 0.0101 | 0.6995 +/- 0.0160 | 0.6960 +/- 0.0139 | 7.67 +/- 3.79 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta10_lg015` | 1.00 | 0.150 | 0.6333 +/- 0.0193 | 0.6432 +/- 0.0140 | 0.6385 +/- 0.0098 | 4.33 +/- 1.53 |
| Flower_102 | `blur_3p0` | `blur_3p0_beta10_lg01` | 1.00 | 0.100 | 0.5492 +/- 0.0623 | 0.5742 +/- 0.0568 | 0.5604 +/- 0.0356 | 7.00 +/- 4.36 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta10_lg01` | 1.00 | 0.100 | 0.6111 +/- 0.0172 | 0.6272 +/- 0.0140 | 0.6241 +/- 0.0158 | 11.33 +/- 0.58 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta10_lg015` | 1.00 | 0.150 | 0.5993 +/- 0.0124 | 0.6184 +/- 0.0104 | 0.6160 +/- 0.0083 | 9.67 +/- 2.31 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 0.5610 +/- 0.0342 | 0.5728 +/- 0.0249 | 0.5642 +/- 0.0259 | 10.00 +/- 5.00 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 0.5357 +/- 0.0358 | 0.5431 +/- 0.0315 | 0.5345 +/- 0.0331 | 9.67 +/- 3.79 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 0.5496 +/- 0.0300 | 0.5557 +/- 0.0211 | 0.5441 +/- 0.0193 | 10.33 +/- 4.73 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 0.5632 +/- 0.0222 | 0.5777 +/- 0.0193 | 0.5653 +/- 0.0189 | 14.00 +/- 3.00 |
