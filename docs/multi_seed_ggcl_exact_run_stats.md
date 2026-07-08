# Multi-Seed GGCL Exact-Run Stats

Each row is one GGCL/sweep parameter setting matched by dataset, noise setting, beta, lambda_gold, and run name.

Values are mean +/- sample standard deviation over seeds 22, 42, and 62.

| Dataset | Noise setting | Run | beta | lambda_G | Sel. Acc | Peak Acc | Final Acc | Selected Epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flower_102 | `blur_3p0` | `blur_3p0_method_beta08_lg01` | 0.80 | 0.100 | 0.5767 +/- 0.0699 | 0.5960 +/- 0.0675 | 0.5773 +/- 0.0587 | 10.67 +/- 6.11 |
| Flower_102 | `blur_3p0` | `blur_3p0_beta02_lg01` | 0.20 | 0.100 | 0.5490 +/- 0.0473 | 0.5653 +/- 0.0532 | 0.5408 +/- 0.0658 | 7.67 +/- 3.51 |
| Flower_102 | `blur_3p0` | `blur_3p0_beta05_lg01` | 0.50 | 0.100 | 0.5673 +/- 0.0715 | 0.5738 +/- 0.0722 | 0.5673 +/- 0.0761 | 8.67 +/- 5.03 |
| Flower_102 | `blur_3p0` | `blur_3p0_beta085_lg01` | 0.85 | 0.100 | 0.5543 +/- 0.0559 | 0.5704 +/- 0.0400 | 0.5665 +/- 0.0364 | 7.67 +/- 3.51 |
| Flower_102 | `blur_3p0` | `blur_3p0_beta08_lg005` | 0.80 | 0.050 | 0.5447 +/- 0.0656 | 0.5539 +/- 0.0705 | 0.5372 +/- 0.0527 | 7.00 +/- 4.36 |
| Flower_102 | `blur_3p0` | `blur_3p0_beta08_lg015` | 0.80 | 0.150 | 0.5596 +/- 0.0576 | 0.5799 +/- 0.0383 | 0.5736 +/- 0.0324 | 7.67 +/- 3.51 |
| Flower_102 | `blur_3p0` | `blur_3p0_beta08_lg02` | 0.80 | 0.200 | 0.5775 +/- 0.0155 | 0.5952 +/- 0.0152 | 0.5911 +/- 0.0145 | 9.00 +/- 1.00 |
| Flower_102 | `blur_3p0` | `blur_3p0_beta09_lg01` | 0.90 | 0.100 | 0.5693 +/- 0.0662 | 0.5919 +/- 0.0532 | 0.5763 +/- 0.0380 | 9.00 +/- 4.36 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_method_beta08_lg01` | 0.80 | 0.100 | 0.5846 +/- 0.0613 | 0.6052 +/- 0.0378 | 0.6052 +/- 0.0378 | 8.00 +/- 3.61 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta02_lg01` | 0.20 | 0.100 | 0.5653 +/- 0.0582 | 0.5944 +/- 0.0473 | 0.5944 +/- 0.0473 | 8.00 +/- 3.61 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta05_lg01` | 0.50 | 0.100 | 0.5769 +/- 0.0587 | 0.5993 +/- 0.0409 | 0.5972 +/- 0.0390 | 8.00 +/- 3.61 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta08_lg005` | 0.80 | 0.050 | 0.5824 +/- 0.0756 | 0.6003 +/- 0.0587 | 0.5989 +/- 0.0574 | 9.33 +/- 4.73 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta08_lg015` | 0.80 | 0.150 | 0.5932 +/- 0.0741 | 0.6170 +/- 0.0402 | 0.6145 +/- 0.0396 | 10.67 +/- 6.51 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta08_lg02` | 0.80 | 0.200 | 0.5767 +/- 0.0620 | 0.6050 +/- 0.0343 | 0.6050 +/- 0.0343 | 7.67 +/- 3.51 |
| Flower_102 | `brightness_0p75` | `brightness_0p75_beta09_lg01` | 0.90 | 0.100 | 0.6225 +/- 0.0144 | 0.6276 +/- 0.0135 | 0.6188 +/- 0.0133 | 11.67 +/- 3.06 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_method_beta05_lg015` | 0.50 | 0.150 | 0.6164 +/- 0.0156 | 0.6345 +/- 0.0110 | 0.6333 +/- 0.0102 | 10.33 +/- 2.08 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta02_lg01` | 0.20 | 0.100 | 0.6198 +/- 0.0029 | 0.6316 +/- 0.0069 | 0.6304 +/- 0.0060 | 11.00 +/- 1.00 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta05_lg005` | 0.50 | 0.050 | 0.6060 +/- 0.0095 | 0.6292 +/- 0.0040 | 0.6274 +/- 0.0024 | 10.67 +/- 1.53 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta05_lg01` | 0.50 | 0.100 | 0.6078 +/- 0.0217 | 0.6276 +/- 0.0137 | 0.6182 +/- 0.0267 | 9.67 +/- 1.53 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta05_lg02` | 0.50 | 0.200 | 0.6219 +/- 0.0078 | 0.6406 +/- 0.0089 | 0.6259 +/- 0.0229 | 11.33 +/- 2.52 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta08_lg01` | 0.80 | 0.100 | 0.6141 +/- 0.0157 | 0.6339 +/- 0.0112 | 0.6280 +/- 0.0213 | 11.00 +/- 1.00 |
| Flower_102 | `gaussian_30p0` | `gaussian_30p0_beta09_lg01` | 0.90 | 0.100 | 0.6092 +/- 0.0156 | 0.6245 +/- 0.0127 | 0.6245 +/- 0.0127 | 11.00 +/- 1.00 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta07_lg01` | 0.70 | 0.100 | 0.5260 +/- 0.0556 | 0.5300 +/- 0.0486 | 0.5105 +/- 0.0394 | 9.67 +/- 5.13 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta075_lg01` | 0.75 | 0.100 | 0.5205 +/- 0.0509 | 0.5296 +/- 0.0418 | 0.5215 +/- 0.0367 | 8.33 +/- 5.13 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta08_lg005` | 0.80 | 0.050 | 0.5180 +/- 0.0266 | 0.5392 +/- 0.0254 | 0.5170 +/- 0.0169 | 12.33 +/- 6.43 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 0.5205 +/- 0.0492 | 0.5306 +/- 0.0409 | 0.5164 +/- 0.0341 | 7.67 +/- 5.51 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_compromise_beta09_lg01` | 0.90 | 0.100 | 0.5111 +/- 0.0435 | 0.5231 +/- 0.0430 | 0.5186 +/- 0.0446 | 6.67 +/- 3.79 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_feature_best_beta08_lg01` | 0.80 | 0.100 | 0.5233 +/- 0.0517 | 0.5304 +/- 0.0393 | 0.5190 +/- 0.0342 | 9.67 +/- 5.13 |
| Flower_102 | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 0.5085 +/- 0.0366 | 0.5156 +/- 0.0335 | 0.5050 +/- 0.0360 | 5.33 +/- 1.53 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_feature_best_beta08_lg01` | 0.80 | 0.100 | 0.5561 +/- 0.0231 | 0.5592 +/- 0.0179 | 0.5480 +/- 0.0080 | 11.33 +/- 2.52 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta075_lg01` | 0.75 | 0.100 | 0.5370 +/- 0.0398 | 0.5449 +/- 0.0330 | 0.5256 +/- 0.0241 | 9.00 +/- 4.58 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta078_lg01` | 0.78 | 0.100 | 0.5583 +/- 0.0157 | 0.5626 +/- 0.0107 | 0.5455 +/- 0.0044 | 13.00 +/- 1.73 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta07_lg01` | 0.70 | 0.100 | 0.5429 +/- 0.0357 | 0.5573 +/- 0.0371 | 0.5414 +/- 0.0258 | 10.00 +/- 2.65 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta085_lg01` | 0.85 | 0.100 | 0.5372 +/- 0.0380 | 0.5455 +/- 0.0329 | 0.5402 +/- 0.0279 | 9.00 +/- 4.58 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta08_lg005` | 0.80 | 0.050 | 0.5433 +/- 0.0326 | 0.5522 +/- 0.0243 | 0.5429 +/- 0.0195 | 10.67 +/- 3.51 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 0.5496 +/- 0.0336 | 0.5581 +/- 0.0208 | 0.5508 +/- 0.0235 | 9.00 +/- 3.61 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_compromise_beta09_lg01` | 0.90 | 0.100 | 0.5443 +/- 0.0348 | 0.5563 +/- 0.0231 | 0.5453 +/- 0.0217 | 9.33 +/- 4.51 |
| Flower_102 | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 0.5414 +/- 0.0357 | 0.5541 +/- 0.0304 | 0.5410 +/- 0.0239 | 9.00 +/- 4.00 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 0.5535 +/- 0.0090 | 0.5640 +/- 0.0156 | 0.5535 +/- 0.0118 | 15.00 +/- 1.00 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta045_lg015` | 0.45 | 0.150 | 0.5553 +/- 0.0294 | 0.5567 +/- 0.0272 | 0.5469 +/- 0.0163 | 11.33 +/- 5.51 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta055_lg015` | 0.55 | 0.150 | 0.5626 +/- 0.0167 | 0.5714 +/- 0.0103 | 0.5685 +/- 0.0113 | 12.33 +/- 2.31 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta05_lg02` | 0.50 | 0.200 | 0.5535 +/- 0.0315 | 0.5669 +/- 0.0253 | 0.5514 +/- 0.0203 | 10.00 +/- 4.58 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta06_lg015` | 0.60 | 0.150 | 0.5431 +/- 0.0291 | 0.5600 +/- 0.0233 | 0.5557 +/- 0.0232 | 8.33 +/- 3.06 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta08_lg015` | 0.80 | 0.150 | 0.5543 +/- 0.0321 | 0.5651 +/- 0.0262 | 0.5518 +/- 0.0206 | 10.00 +/- 4.58 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_compromise_beta07_lg015` | 0.70 | 0.150 | 0.5510 +/- 0.0342 | 0.5689 +/- 0.0264 | 0.5488 +/- 0.0254 | 10.00 +/- 4.58 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_feature_best_beta05_lg015` | 0.50 | 0.150 | 0.5561 +/- 0.0124 | 0.5665 +/- 0.0100 | 0.5610 +/- 0.0089 | 12.67 +/- 2.08 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_label_best_beta095_lg01` | 0.95 | 0.100 | 0.5471 +/- 0.0384 | 0.5651 +/- 0.0275 | 0.5551 +/- 0.0309 | 10.00 +/- 4.58 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta02_lg01` | 0.20 | 0.100 | 0.5508 +/- 0.0281 | 0.5545 +/- 0.0308 | 0.5412 +/- 0.0211 | 11.00 +/- 5.29 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 0.5524 +/- 0.0247 | 0.5602 +/- 0.0256 | 0.5526 +/- 0.0241 | 8.67 +/- 3.51 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta08_lg01` | 0.80 | 0.100 | 0.5632 +/- 0.0406 | 0.5789 +/- 0.0264 | 0.5618 +/- 0.0247 | 11.33 +/- 5.51 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta095_lg005` | 0.95 | 0.050 | 0.5510 +/- 0.0401 | 0.5634 +/- 0.0300 | 0.5571 +/- 0.0334 | 9.33 +/- 4.51 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta095_lg015` | 0.95 | 0.150 | 0.5683 +/- 0.0341 | 0.5734 +/- 0.0287 | 0.5695 +/- 0.0233 | 11.33 +/- 5.51 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta09_lg01` | 0.90 | 0.100 | 0.5781 +/- 0.0086 | 0.5862 +/- 0.0133 | 0.5765 +/- 0.0049 | 13.00 +/- 2.00 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_beta10_lg01` | 1.00 | 0.100 | 0.5610 +/- 0.0342 | 0.5728 +/- 0.0249 | 0.5642 +/- 0.0259 | 10.00 +/- 5.00 |
| Flower_102 | `label_shuffle_0p2` | `label_shuffle_0p2_method_beta095_lg01` | 0.95 | 0.100 | 0.5506 +/- 0.0489 | 0.5750 +/- 0.0364 | 0.5628 +/- 0.0282 | 10.00 +/- 4.58 |
| STL | `blur_3p0` | `blur_3p0_method_beta09_lg015` | 0.90 | 0.150 | 0.6317 +/- 0.0039 | 0.6327 +/- 0.0057 | 0.6088 +/- 0.0291 | 3.00 +/- 2.65 |
| STL | `blur_3p0` | `blur_3p0_beta02_lg01` | 0.20 | 0.100 | 0.5988 +/- 0.0183 | 0.5988 +/- 0.0183 | 0.5341 +/- 0.0390 | 1.67 +/- 0.58 |
| STL | `blur_3p0` | `blur_3p0_beta05_lg01` | 0.50 | 0.100 | 0.6185 +/- 0.0147 | 0.6215 +/- 0.0200 | 0.6065 +/- 0.0339 | 3.67 +/- 1.53 |
| STL | `blur_3p0` | `blur_3p0_beta085_lg01` | 0.85 | 0.100 | 0.6218 +/- 0.0096 | 0.6235 +/- 0.0072 | 0.5927 +/- 0.0037 | 2.33 +/- 1.53 |
| STL | `blur_3p0` | `blur_3p0_beta09_lg0075` | 0.90 | 0.075 | 0.6300 +/- 0.0206 | 0.6300 +/- 0.0206 | 0.6147 +/- 0.0338 | 4.00 +/- 1.73 |
| STL | `blur_3p0` | `blur_3p0_beta09_lg01` | 0.90 | 0.100 | 0.6362 +/- 0.0064 | 0.6372 +/- 0.0080 | 0.5972 +/- 0.0429 | 3.67 +/- 1.53 |
| STL | `brightness_0p75` | `brightness_0p75_method_beta09_lg01` | 0.90 | 0.100 | 0.7319 +/- 0.0106 | 0.7440 +/- 0.0091 | 0.7277 +/- 0.0210 | 11.33 +/- 2.08 |
| STL | `brightness_0p75` | `brightness_0p75_beta02_lg01` | 0.20 | 0.100 | 0.7186 +/- 0.0180 | 0.7260 +/- 0.0156 | 0.7103 +/- 0.0298 | 9.33 +/- 2.52 |
| STL | `brightness_0p75` | `brightness_0p75_beta05_lg01` | 0.50 | 0.100 | 0.7286 +/- 0.0051 | 0.7435 +/- 0.0073 | 0.7268 +/- 0.0308 | 11.67 +/- 3.79 |
| STL | `brightness_0p75` | `brightness_0p75_beta085_lg01` | 0.85 | 0.100 | 0.7247 +/- 0.0120 | 0.7371 +/- 0.0140 | 0.7262 +/- 0.0191 | 8.67 +/- 2.52 |
| STL | `brightness_0p75` | `brightness_0p75_beta09_lg0075` | 0.90 | 0.075 | 0.7291 +/- 0.0124 | 0.7362 +/- 0.0092 | 0.7294 +/- 0.0127 | 12.33 +/- 3.51 |
| STL | `brightness_0p75` | `brightness_0p75_beta09_lg015` | 0.90 | 0.150 | 0.7053 +/- 0.0060 | 0.7169 +/- 0.0141 | 0.7045 +/- 0.0234 | 6.33 +/- 1.15 |
| STL | `gaussian_30p0` | `gaussian_30p0_method_beta05_lg015` | 0.50 | 0.150 | 0.6686 +/- 0.0159 | 0.6751 +/- 0.0133 | 0.6650 +/- 0.0153 | 6.67 +/- 4.04 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta02_lg01` | 0.20 | 0.100 | 0.6599 +/- 0.0115 | 0.6727 +/- 0.0056 | 0.6429 +/- 0.0251 | 9.67 +/- 2.08 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta05_lg0075` | 0.50 | 0.075 | 0.6618 +/- 0.0125 | 0.6696 +/- 0.0132 | 0.6608 +/- 0.0163 | 7.33 +/- 1.53 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta05_lg01` | 0.50 | 0.100 | 0.6579 +/- 0.0155 | 0.6795 +/- 0.0046 | 0.6683 +/- 0.0091 | 7.33 +/- 1.53 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta05_lg02` | 0.50 | 0.200 | 0.6736 +/- 0.0303 | 0.6831 +/- 0.0264 | 0.6586 +/- 0.0171 | 8.33 +/- 5.86 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta085_lg01` | 0.85 | 0.100 | 0.6465 +/- 0.0050 | 0.6544 +/- 0.0052 | 0.6471 +/- 0.0126 | 5.00 +/- 1.00 |
| STL | `gaussian_30p0` | `gaussian_30p0_beta09_lg01` | 0.90 | 0.100 | 0.6408 +/- 0.0116 | 0.6555 +/- 0.0061 | 0.6485 +/- 0.0128 | 4.67 +/- 0.58 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta09_lg025` | 0.90 | 0.250 | 0.6360 +/- 0.0126 | 0.6395 +/- 0.0092 | 0.6304 +/- 0.0071 | 2.33 +/- 1.53 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta085_lg015` | 0.85 | 0.150 | 0.6253 +/- 0.0182 | 0.6308 +/- 0.0087 | 0.5903 +/- 0.0172 | 3.33 +/- 2.08 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta095_lg015` | 0.95 | 0.150 | 0.6253 +/- 0.0186 | 0.6331 +/- 0.0060 | 0.6195 +/- 0.0094 | 3.00 +/- 1.73 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta09_lg01` | 0.90 | 0.100 | 0.6144 +/- 0.0143 | 0.6144 +/- 0.0143 | 0.5776 +/- 0.0026 | 3.00 +/- 1.73 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_beta09_lg03` | 0.90 | 0.300 | 0.6295 +/- 0.0154 | 0.6404 +/- 0.0054 | 0.6303 +/- 0.0212 | 2.33 +/- 1.53 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_compromise_beta09_lg02` | 0.90 | 0.200 | 0.6312 +/- 0.0104 | 0.6356 +/- 0.0049 | 0.6204 +/- 0.0148 | 3.00 +/- 1.73 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_feature_best_beta09_lg015` | 0.90 | 0.150 | 0.6172 +/- 0.0191 | 0.6260 +/- 0.0115 | 0.6050 +/- 0.0108 | 2.33 +/- 1.53 |
| STL | `blur_3p0 + label_shuffle_0p2` | `blur_3p0_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 0.6074 +/- 0.0039 | 0.6240 +/- 0.0183 | 0.6090 +/- 0.0292 | 2.33 +/- 1.53 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta085_lg01` | 0.85 | 0.100 | 0.7032 +/- 0.0152 | 0.7088 +/- 0.0197 | 0.6983 +/- 0.0191 | 9.33 +/- 3.51 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta095_lg01` | 0.95 | 0.100 | 0.6810 +/- 0.0264 | 0.6912 +/- 0.0246 | 0.6890 +/- 0.0235 | 5.33 +/- 3.51 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta09_lg005` | 0.90 | 0.050 | 0.6872 +/- 0.0245 | 0.6968 +/- 0.0306 | 0.6831 +/- 0.0377 | 7.67 +/- 5.13 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_beta09_lg0075` | 0.90 | 0.075 | 0.6776 +/- 0.0389 | 0.6954 +/- 0.0257 | 0.6810 +/- 0.0275 | 9.00 +/- 8.00 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_compromise_beta09_lg015` | 0.90 | 0.150 | 0.7010 +/- 0.0124 | 0.7038 +/- 0.0093 | 0.6938 +/- 0.0059 | 7.67 +/- 4.62 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_feature_best_beta09_lg01` | 0.90 | 0.100 | 0.7040 +/- 0.0184 | 0.7074 +/- 0.0143 | 0.7023 +/- 0.0150 | 8.67 +/- 0.58 |
| STL | `brightness_0p75 + label_shuffle_0p2` | `brightness_0p75_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 0.6860 +/- 0.0163 | 0.6971 +/- 0.0056 | 0.6914 +/- 0.0046 | 5.67 +/- 2.08 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta06_lg015` | 0.60 | 0.150 | 0.6671 +/- 0.0182 | 0.6746 +/- 0.0121 | 0.6537 +/- 0.0267 | 12.67 +/- 0.58 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta04_lg015` | 0.40 | 0.150 | 0.6338 +/- 0.0068 | 0.6586 +/- 0.0045 | 0.6209 +/- 0.0386 | 8.67 +/- 3.51 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 0.6381 +/- 0.0084 | 0.6636 +/- 0.0097 | 0.6521 +/- 0.0148 | 10.00 +/- 1.73 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_beta07_lg015` | 0.70 | 0.150 | 0.6672 +/- 0.0252 | 0.6683 +/- 0.0252 | 0.6495 +/- 0.0290 | 9.33 +/- 4.04 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_compromise_beta05_lg02` | 0.50 | 0.200 | 0.6546 +/- 0.0199 | 0.6591 +/- 0.0136 | 0.6372 +/- 0.0291 | 9.33 +/- 4.04 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_feature_best_beta05_lg015` | 0.50 | 0.150 | 0.6465 +/- 0.0058 | 0.6592 +/- 0.0133 | 0.5988 +/- 0.0479 | 9.33 +/- 7.51 |
| STL | `gaussian_30p0 + label_shuffle_0p2` | `gaussian_30p0_label_shuffle_0p2_label_best_beta05_lg025` | 0.50 | 0.250 | 0.6513 +/- 0.0253 | 0.6632 +/- 0.0211 | 0.6283 +/- 0.0068 | 8.67 +/- 2.52 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta00_lg01` | 0.00 | 0.100 | 0.6760 +/- 0.0125 | 0.6831 +/- 0.0098 | 0.6638 +/- 0.0112 | 11.67 +/- 3.21 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta01_lg01` | 0.10 | 0.100 | 0.6656 +/- 0.0365 | 0.6851 +/- 0.0190 | 0.6658 +/- 0.0167 | 8.67 +/- 4.16 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta02_lg01` | 0.20 | 0.100 | 0.6850 +/- 0.0020 | 0.6901 +/- 0.0035 | 0.6724 +/- 0.0225 | 11.33 +/- 1.53 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta05_lg005` | 0.50 | 0.050 | 0.6796 +/- 0.0147 | 0.6936 +/- 0.0142 | 0.6862 +/- 0.0209 | 11.67 +/- 2.31 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta05_lg01` | 0.50 | 0.100 | 0.7060 +/- 0.0118 | 0.7065 +/- 0.0110 | 0.6788 +/- 0.0153 | 13.33 +/- 0.58 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta05_lg015` | 0.50 | 0.150 | 0.7032 +/- 0.0104 | 0.7136 +/- 0.0036 | 0.6822 +/- 0.0336 | 12.33 +/- 3.06 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta05_lg02` | 0.50 | 0.200 | 0.7050 +/- 0.0075 | 0.7094 +/- 0.0091 | 0.6992 +/- 0.0041 | 8.00 +/- 2.00 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_beta05_lg03` | 0.50 | 0.300 | 0.6828 +/- 0.0330 | 0.6953 +/- 0.0281 | 0.6891 +/- 0.0188 | 4.67 +/- 3.06 |
| STL | `label_shuffle_0p2` | `label_shuffle_0p2_method_beta05_lg025` | 0.50 | 0.250 | 0.7186 +/- 0.0012 | 0.7221 +/- 0.0069 | 0.6954 +/- 0.0103 | 12.33 +/- 2.08 |
