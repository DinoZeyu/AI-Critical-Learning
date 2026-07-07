# Supplemental Ablation Results

This file records the supplemental ablation comparison against the previously
reported Gold-Guided Critical Learning best run for each matching dataset and
noise setting.

Protocol:

- Seed: `42`
- Headline metric: `test_accuracy_at_selected_epoch`
- Checkpoint selection: clean gold validation loss
- Gold evaluator path: `Gold_Evaluators/<dataset>/model.pt`
- Final over-best decision: use the rerun-confirmed ablation results

## Rerun-Confirmed Over-Best Ablations

These three ablation settings are higher than the previous Gold-Guided Critical
Learning best under the same dataset, noise setting, seed, and `lambda_gold`.

| Dataset | Noise setting | Ablation setting | Previous GGCL best | Ablation selected acc | Previous best selected acc | Delta |
|---|---|---|---|---:|---:|---:|
| STL | `blur_3p0 + label_shuffle_0p2` | `beta=1.00`, `lambda_gold=0.25` | `beta=0.90`, `lambda_gold=0.25` | 0.6596 | 0.6500 | +0.0096 |
| Flower_102 | `gaussian_30p0 + label_shuffle_0p2` | `beta=1.00`, `lambda_gold=0.10` | `beta=0.50`, `lambda_gold=0.10` | 0.5577 | 0.5486 | +0.0092 |
| STL | `label_shuffle_0p2` | `beta=1.00`, `lambda_gold=0.25` | `beta=0.50`, `lambda_gold=0.25` | 0.7177 | 0.7173 | +0.0004 |

## Non-Confirmed Initial Overrun

The initial 42-run supplemental ablation batch also showed Flower_102
`blur_3p0` with `beta=0.00`, `lambda_gold=0.10` above the previous best.
The rerun did not confirm that overrun, so it is not counted in the final
over-best list.

| Dataset | Noise setting | Ablation setting | Rerun selected acc | Previous best selected acc | Delta |
|---|---|---|---:|---:|---:|
| Flower_102 | `blur_3p0` | `beta=0.00`, `lambda_gold=0.10` | 0.5370 | 0.6151 | -0.0782 |

## Source Result Folders

| Result | Path |
|---|---|
| STL hybrid blur+label previous best | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_beta09_lg025/` |
| STL hybrid blur+label confirmed ablation | `Supplemental_Ablation_Results/STL/overrun_ablation_rerun_round2/seed42/hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_beta10_lg025/` |
| Flower_102 hybrid gaussian+label previous best | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/Flower_102/hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_beta05_lg01/` |
| Flower_102 hybrid gaussian+label confirmed ablation | `Supplemental_Ablation_Results/Flower_102/overrun_ablation_rerun_round2/seed42/hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_beta10_lg01/` |
| STL label shuffle previous best | `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/STL/label_noise/label_shuffle/label_shuffle_0p2_method_beta05_lg025/` |
| STL label shuffle confirmed ablation | `Supplemental_Ablation_Results/STL/overrun_ablation_rerun_round2/seed42/label_noise/label_shuffle/label_shuffle_0p2_beta10_lg025/` |
| Flower_102 blur non-confirmed ablation rerun | `Supplemental_Ablation_Results/Flower_102/overrun_ablation_rerun/seed42/feature_noise/blur/blur_3p0_beta00_lg01/` |

## Notes

- All three confirmed over-best cases are `beta=1.00`, meaning the
  prototype-similarity component is disabled and the gold evaluator prediction
  confidence signal is used alone.
- The `lambda_gold` value is unchanged from the previous GGCL best in each
  comparison, so the overrun is isolated to the `beta` ablation.
- The STL `label_shuffle_0p2` margin is very small (`+0.0004`) and should be
  described as effectively tied/slightly higher rather than a strong gain.
