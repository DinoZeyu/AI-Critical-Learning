# Flower_102 Commands

Commands for the Flower_102 experiments. These assume the repo root is the working directory.

## Gold Data

Initial selection:

```bash
python scripts/Data_Prepration/LLM_Select_Gold.py --dataset Flower_102
```

Repair command used when strict thresholding leaves too few examples in some classes:

```bash
python scripts/Data_Prepration/LLM_Select_Gold.py --dataset Flower_102 \
  --quality-threshold 0.90 \
  --min-gold-per-class 5 \
  --fallback-quality-threshold 0.0
```

After changing the Flower_102 gold split, regenerate noisy train data and rerun the evaluator/baselines.

## Noisy Train Data

```bash
python scripts/Data_Prepration/make_noisy_data.py --dataset Flower_102 --split train --noise-type blur --value 3 --overwrite
python scripts/Data_Prepration/make_noisy_data.py --dataset Flower_102 --split train --noise-type brightness --value 0.75 --overwrite
python scripts/Data_Prepration/make_noisy_data.py --dataset Flower_102 --split train --noise-type gaussian --value 30 --overwrite
python scripts/Data_Prepration/make_noisy_data.py --dataset Flower_102 --split train --noise-type label_shuffle --value 0.2 --overwrite
```

Hybrid feature+label noisy train data:

```bash
python scripts/Data_Prepration/make_hybrid_noisy_data.py --dataset Flower_102 --split train --feature-noise-type blur --feature-value 3 --label-shuffle-fraction 0.2 --overwrite
python scripts/Data_Prepration/make_hybrid_noisy_data.py --dataset Flower_102 --split train --feature-noise-type brightness --feature-value 0.75 --label-shuffle-fraction 0.2 --overwrite
python scripts/Data_Prepration/make_hybrid_noisy_data.py --dataset Flower_102 --split train --feature-noise-type gaussian --feature-value 30 --label-shuffle-fraction 0.2 --overwrite
```

## Gold Evaluator

Current Flower_102 evaluator command:

```bash
python scripts/Train_Gold_Evaluator.py --dataset Flower_102 \
  --epochs 40 \
  --lr 0.0005 \
  --selection-metric val-loss \
  --train-augmentation feature-noise \
  --early-stop-patience 5 \
  --early-stop-min-delta 0.001
```

Current saved evaluator summary:

- Path: `Evaluators/Flower_102/model.pt`
- Best epoch: `33`
- Gold validation loss: `2.0351`
- Gold validation accuracy: `0.4701`

## Clean Baseline

Clean train -> clean test:

```bash
python scripts/Baseline_Exp.py --dataset Flower_102 \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

Result path:

```text
Experiments_Results/Train_Clean_Test_Clean/Flower_102/
```

Optional ablation that intentionally keeps gold rows in train:

```bash
python scripts/Baseline_Exp.py --dataset Flower_102 \
  --include-gold-in-train \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

Use this only as an ablation, not as the headline baseline.

Current Flower_102 experiments use the default `224` image size. A high-resolution
`448` clean baseline was tested and performed worse with the current SimpleCNN, so
the main Flower_102 runs should stay at `224` unless the backbone is changed.

## Noise Baselines

Feature-noise baselines:

```bash
python scripts/Baseline_Exp.py --dataset Flower_102 \
  --train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/blur_3p0 \
  --test-dir Image_Data/Test_Clean_Data/Flower_102 \
  --early-stop-patience 3 \
  --selection-metric val-loss

python scripts/Baseline_Exp.py --dataset Flower_102 \
  --train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 \
  --test-dir Image_Data/Test_Clean_Data/Flower_102 \
  --early-stop-patience 3 \
  --selection-metric val-loss

python scripts/Baseline_Exp.py --dataset Flower_102 \
  --train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 \
  --test-dir Image_Data/Test_Clean_Data/Flower_102 \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

Label-noise baseline:

```bash
python scripts/Baseline_Exp.py --dataset Flower_102 \
  --train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 \
  --test-dir Image_Data/Test_Clean_Data/Flower_102 \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

## Gold-Guided CL Templates

Generic feature-noise templates:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/blur_3p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 \
  --early-stop-patience 3 \
  --selection-metric val-loss

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

Generic label-noise template:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss
```

## First Flower_102 Sweeps

Use these after the Flower_102 noise baselines are complete.

Feature noise should start with a small beta sweep, because Flower_102 has a weaker evaluator than STL:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/blur_3p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.2 \
  --lambda-gold 0.1 \
  --run-name feature_noise/blur/blur_3p0_method_beta02_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/blur_3p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.5 \
  --lambda-gold 0.1 \
  --run-name feature_noise/blur/blur_3p0_method_beta05_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/blur_3p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.8 \
  --lambda-gold 0.1 \
  --run-name feature_noise/blur/blur_3p0_method_beta08_lg01
```

Brightness sweep:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.2 \
  --lambda-gold 0.1 \
  --run-name feature_noise/brightness/brightness_0p75_method_beta02_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.5 \
  --lambda-gold 0.1 \
  --run-name feature_noise/brightness/brightness_0p75_method_beta05_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.8 \
  --lambda-gold 0.1 \
  --run-name feature_noise/brightness/brightness_0p75_method_beta08_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.9 \
  --lambda-gold 0.1 \
  --run-name feature_noise/brightness/brightness_0p75_method_beta09_lg01
```

If `beta=0.8` is best, use this lambda sweep:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.8 \
  --lambda-gold 0.05 \
  --run-name feature_noise/brightness/brightness_0p75_method_beta08_lg005

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.8 \
  --lambda-gold 0.15 \
  --run-name feature_noise/brightness/brightness_0p75_method_beta08_lg015

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.8 \
  --lambda-gold 0.2 \
  --run-name feature_noise/brightness/brightness_0p75_method_beta08_lg02
```

If `beta=0.9` is best, reuse the same lambda sweep with `--beta 0.9` and run names like `brightness_0p75_method_beta09_lg015`.

Gaussian sweep:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.2 \
  --lambda-gold 0.1 \
  --run-name feature_noise/gaussian/gaussian_30p0_method_beta02_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.5 \
  --lambda-gold 0.1 \
  --run-name feature_noise/gaussian/gaussian_30p0_method_beta05_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.8 \
  --lambda-gold 0.1 \
  --run-name feature_noise/gaussian/gaussian_30p0_method_beta08_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.9 \
  --lambda-gold 0.1 \
  --run-name feature_noise/gaussian/gaussian_30p0_method_beta09_lg01
```

Gaussian lambda sweep used after selecting `beta=0.5`:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.5 \
  --lambda-gold 0.05 \
  --run-name feature_noise/gaussian/gaussian_30p0_method_beta05_lg005

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.5 \
  --lambda-gold 0.15 \
  --run-name feature_noise/gaussian/gaussian_30p0_method_beta05_lg015

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.5 \
  --lambda-gold 0.2 \
  --run-name feature_noise/gaussian/gaussian_30p0_method_beta05_lg02
```

Current Gaussian selected best:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.5 \
  --lambda-gold 0.15 \
  --run-name feature_noise/gaussian/gaussian_30p0_method_beta05_lg015
```

Label noise should start around balanced evaluator/prototype weighting. The images are clean but labels are corrupted, so use a beta sweep first and then tune `lambda_gold` around the best beta.

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.2 \
  --lambda-gold 0.1 \
  --run-name label_noise/label_shuffle/label_shuffle_0p2_method_beta02_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.5 \
  --lambda-gold 0.1 \
  --run-name label_noise/label_shuffle/label_shuffle_0p2_method_beta05_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.8 \
  --lambda-gold 0.1 \
  --run-name label_noise/label_shuffle/label_shuffle_0p2_method_beta08_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.9 \
  --lambda-gold 0.1 \
  --run-name label_noise/label_shuffle/label_shuffle_0p2_method_beta09_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.95 \
  --lambda-gold 0.1 \
  --run-name label_noise/label_shuffle/label_shuffle_0p2_method_beta095_lg01

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 1.0 \
  --lambda-gold 0.1 \
  --run-name label_noise/label_shuffle/label_shuffle_0p2_method_beta10_lg01
```

Label-noise lambda sweep around selected `beta=0.95`:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.95 \
  --lambda-gold 0.05 \
  --run-name label_noise/label_shuffle/label_shuffle_0p2_method_beta095_lg005

python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.95 \
  --lambda-gold 0.15 \
  --run-name label_noise/label_shuffle/label_shuffle_0p2_method_beta095_lg015
```

The planned `lambda_gold=0.20` run was skipped because both `0.05` and `0.15` were much weaker than `0.10`.

Current label-noise selected best:

```bash
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 \
  --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 \
  --early-stop-patience 3 \
  --selection-metric val-loss \
  --beta 0.95 \
  --lambda-gold 0.1 \
  --run-name label_noise/label_shuffle/label_shuffle_0p2_method_beta095_lg01
```
