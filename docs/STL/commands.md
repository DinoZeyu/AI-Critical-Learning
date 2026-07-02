# STL Commands

Commands for the STL experiments. These assume the repo root is the working directory.

## Gold Data

Initial selection:

```bash
python scripts/Data_Prepration/LLM_Select_Gold.py --dataset STL
```

Current stronger STL gold split used for the completed experiments:

```bash
python scripts/Data_Prepration/LLM_Select_Gold.py --dataset STL \
  --gold-ratio 0.30 \
  --quality-threshold 0.90 \
  --fill-target-shortfall \
  --fallback-quality-threshold 0.85
```

After changing the gold split, regenerate noisy train data and rerun the evaluator/baselines.

## Noisy Train Data

```bash
python scripts/Data_Prepration/make_noisy_data.py --dataset STL --split train --noise-type blur --value 3 --overwrite
python scripts/Data_Prepration/make_noisy_data.py --dataset STL --split train --noise-type brightness --value 0.75 --overwrite
python scripts/Data_Prepration/make_noisy_data.py --dataset STL --split train --noise-type gaussian --value 30 --overwrite
python scripts/Data_Prepration/make_noisy_data.py --dataset STL --split train --noise-type label_shuffle --value 0.2 --overwrite
```

Hybrid feature+label noisy train data:

```bash
python scripts/Data_Prepration/make_hybrid_noisy_data.py --dataset STL --split train --feature-noise-type blur --feature-value 3 --label-shuffle-fraction 0.2 --overwrite
python scripts/Data_Prepration/make_hybrid_noisy_data.py --dataset STL --split train --feature-noise-type brightness --feature-value 0.75 --label-shuffle-fraction 0.2 --overwrite
python scripts/Data_Prepration/make_hybrid_noisy_data.py --dataset STL --split train --feature-noise-type gaussian --feature-value 30 --label-shuffle-fraction 0.2 --overwrite
```

## Gold Evaluator

Current STL evaluator command:

```bash
python scripts/Train_Gold_Evaluator.py --dataset STL \
  --epochs 40 \
  --lr 0.0005 \
  --selection-metric val-loss \
  --train-augmentation feature-noise \
  --early-stop-patience 6 \
  --early-stop-min-delta 0.001
```

Current saved evaluator summary:

- Path: `Gold_Evaluators/STL/model.pt`
- Best epoch: `12`
- Gold validation loss: `0.8012`
- Gold validation accuracy: `0.7323`

## Clean Baseline

Clean train -> clean test:

```bash
python scripts/Baseline_Exp.py --dataset STL \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

Result path:

```text
Experiments_Results/Train_Clean_Test_Clean/STL/
```

Optional ablation that intentionally keeps gold rows in train:

```bash
python scripts/Baseline_Exp.py --dataset STL \
  --include-gold-in-train \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

Use this only as an ablation, not as the headline baseline.

## Noise Baselines

Feature-noise baselines:

```bash
python scripts/Baseline_Exp.py --dataset STL \
  --train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 \
  --test-dir Image_Data/Test_Clean_Data/STL \
  --early-stop-patience 3 \
  --selection-metric val-loss

python scripts/Baseline_Exp.py --dataset STL \
  --train-dir Image_Data/Train_Noise_Data/STL/feature_noise/brightness_0p75 \
  --test-dir Image_Data/Test_Clean_Data/STL \
  --early-stop-patience 3 \
  --selection-metric val-loss

python scripts/Baseline_Exp.py --dataset STL \
  --train-dir Image_Data/Train_Noise_Data/STL/feature_noise/gaussian_30p0 \
  --test-dir Image_Data/Test_Clean_Data/STL \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

Label-noise baseline:

```bash
python scripts/Baseline_Exp.py --dataset STL \
  --train-dir Image_Data/Train_Noise_Data/STL/label_noise/label_shuffle_0p2 \
  --test-dir Image_Data/Test_Clean_Data/STL \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

## Gold-Guided CL Templates

Generic feature-noise templates using script defaults:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss

python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/brightness_0p75 \
  --early-stop-patience 3 \
  --selection-metric val-loss

python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/gaussian_30p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

Generic label-noise template:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

## Current Best STL Runs

These are experiment-selected hyperparameters, not fixed method defaults.

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.9 \
  --lambda-gold 0.15 \
  --run-name feature_noise/blur/blur_3p0_method_beta09_lg015

python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/brightness_0p75 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.9 \
  --lambda-gold 0.1 \
  --run-name feature_noise/brightness/brightness_0p75_method_beta09_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/gaussian_30p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.5 \
  --lambda-gold 0.15 \
  --run-name feature_noise/gaussian/gaussian_30p0_method_beta05_lg015

python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.5 \
  --lambda-gold 0.25 \
  --run-name label_noise/label_shuffle/label_shuffle_0p2_method_beta05_lg025
```

## STL Hybrid Gold-Guided CL Seeded Sweep

These hybrid commands use the best single-noise STL parameters as anchors:
the corresponding feature-noise best setting, the label-noise best setting,
and one compromise setting. The gold evaluator checkpoint is passed explicitly
for reproducibility.

Blur + label shuffle:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/hybrid_noise/blur_3p0_label_shuffle_0p2 \
  --gold-evaluator-checkpoint Gold_Evaluators/STL/model.pt \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.90 \
  --lambda-gold 0.15 \
  --run-name hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_feature_best_beta09_lg015

python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/hybrid_noise/blur_3p0_label_shuffle_0p2 \
  --gold-evaluator-checkpoint Gold_Evaluators/STL/model.pt \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.50 \
  --lambda-gold 0.25 \
  --run-name hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_label_best_beta05_lg025

python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/hybrid_noise/blur_3p0_label_shuffle_0p2 \
  --gold-evaluator-checkpoint Gold_Evaluators/STL/model.pt \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.90 \
  --lambda-gold 0.20 \
  --run-name hybrid_noise/blur_label_shuffle/blur_3p0_label_shuffle_0p2_compromise_beta09_lg02
```

Brightness + label shuffle:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/hybrid_noise/brightness_0p75_label_shuffle_0p2 \
  --gold-evaluator-checkpoint Gold_Evaluators/STL/model.pt \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.90 \
  --lambda-gold 0.10 \
  --run-name hybrid_noise/brightness_label_shuffle/brightness_0p75_label_shuffle_0p2_feature_best_beta09_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/hybrid_noise/brightness_0p75_label_shuffle_0p2 \
  --gold-evaluator-checkpoint Gold_Evaluators/STL/model.pt \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.50 \
  --lambda-gold 0.25 \
  --run-name hybrid_noise/brightness_label_shuffle/brightness_0p75_label_shuffle_0p2_label_best_beta05_lg025

python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/hybrid_noise/brightness_0p75_label_shuffle_0p2 \
  --gold-evaluator-checkpoint Gold_Evaluators/STL/model.pt \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.90 \
  --lambda-gold 0.15 \
  --run-name hybrid_noise/brightness_label_shuffle/brightness_0p75_label_shuffle_0p2_compromise_beta09_lg015
```

Gaussian + label shuffle:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/hybrid_noise/gaussian_30p0_label_shuffle_0p2 \
  --gold-evaluator-checkpoint Gold_Evaluators/STL/model.pt \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.50 \
  --lambda-gold 0.15 \
  --run-name hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_feature_best_beta05_lg015

python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/hybrid_noise/gaussian_30p0_label_shuffle_0p2 \
  --gold-evaluator-checkpoint Gold_Evaluators/STL/model.pt \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.50 \
  --lambda-gold 0.25 \
  --run-name hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_label_best_beta05_lg025

python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/hybrid_noise/gaussian_30p0_label_shuffle_0p2 \
  --gold-evaluator-checkpoint Gold_Evaluators/STL/model.pt \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.50 \
  --lambda-gold 0.20 \
  --run-name hybrid_noise/gaussian_label_shuffle/gaussian_30p0_label_shuffle_0p2_compromise_beta05_lg02
```

Result summaries:

- [Blur results](feature_noise/stl_blur_3p0_results.md)
- [Brightness results](feature_noise/stl_brightness_0p75_results.md)
- [Gaussian results](feature_noise/stl_gaussian_30p0_results.md)
- [Label shuffle results](label_noise/stl_label_shuffle_0p2_results.md)
- [Blur + label shuffle hybrid results](hybrid_noise/stl_blur_3p0_label_shuffle_0p2_results.md)
- [Brightness + label shuffle hybrid results](hybrid_noise/stl_brightness_0p75_label_shuffle_0p2_results.md)
- [Gaussian + label shuffle hybrid results](hybrid_noise/stl_gaussian_30p0_label_shuffle_0p2_results.md)

## Optional STL Ablations

Dynamic non-negative controller:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 \
  --alpha-schedule linear \
  --alpha-start 0.5 \
  --alpha 2.0 \
  --run-name feature_noise/blur/blur_3p0_strong_gold_dynamic_alpha_linear
```

Freeze learner BatchNorm running stats during noisy mixed-data training:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 \
  --freeze-learner-batch-norm \
  --run-name feature_noise/blur/blur_3p0_strong_gold_freeze_bn
```

Old LR profile ablation:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset STL \
  --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 \
  --optimizer sgd \
  --base-lr 0.05 \
  --min-lr 0.001
```
