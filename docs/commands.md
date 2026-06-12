# Select gold evaluator data from Train_Clean_Data, copy it to Gold_Data, and remove the selected rows from the train labels to avoid train/evaluator leakage.

python scripts/Data_Prepration/LLM_Select_Gold.py --dataset STL
python scripts/Data_Prepration/LLM_Select_Gold.py --dataset Flower_102

# Stronger STL gold set for feature-noise robustness.
# This reuses existing annotations.jsonl when available, so it should not call OpenAI again if all rows are already annotated.
# It raises the STL gold target to 30%, uses a slightly softer primary threshold, and fills per-class target shortfalls
# with top-scoring fallback rows that still pass fallback-quality-threshold.
# After this command, regenerate STL noisy data and rerun STL baselines/evaluator, because Train_Clean_Data/STL/labels.csv changes.

python scripts/Data_Prepration/LLM_Select_Gold.py --dataset STL --gold-ratio 0.30 --quality-threshold 0.90 --fill-target-shortfall --fallback-quality-threshold 0.85

# Flower_102 repair command if strict threshold=0.93 leaves many classes without gold data.
# This reuses existing annotations.jsonl, so it should not call OpenAI again if all rows are already annotated.
# It keeps high-quality rows at threshold=0.90, then supplements each class to at least 5 gold rows using class top-k fallback.
# fallback-quality-threshold=0.0 is needed because a few Flower classes have no high-scoring candidates from the LLM.
# After running this repair command, regenerate Flower_102 noisy data and retrain the Flower_102 gold evaluator,
# because Train_Clean_Data/Flower_102/labels.csv will exclude the newly selected gold rows.

python scripts/Data_Prepration/LLM_Select_Gold.py --dataset Flower_102 --quality-threshold 0.90 --min-gold-per-class 5 --fallback-quality-threshold 0.0





---------------------------------------------------------------------------------------------------------------------------------------------
# Baseline clean-train / clean-test experiments.
# Default behavior trains on Train_Clean_Data/<dataset>/labels.csv, where selected gold rows have already been removed.
# Test data is always Test_Clean_Data/<dataset>/labels.csv.
# Checkpoint selection is aligned with gold-guided runs:
# by default Baseline_Exp.py uses a stratified 10% split from Image_Data/Gold_Data/<dataset> as clean validation.
# Baseline runs are for comparison metrics, so they do not save model checkpoints by default.
# Pass --save-checkpoints only when you need to inspect a baseline model.pt or diagnostic model_peak_test.pt.
# For fair headline comparisons, use test_accuracy_at_selected_epoch.
# For method exploration, also inspect max_test_accuracy_observed.
# Results are saved to Experiments_Results/Train_Clean_Test_Clean/Baseline_Exp/<dataset>/clean/.

python scripts/Baseline_Exp.py --dataset STL --early-stop-patience 3 --selection-metric val-loss
python scripts/Baseline_Exp.py --dataset Flower_102 --early-stop-patience 3 --selection-metric val-loss

# Baseline ablation that intentionally includes the original gold rows in train.
# This uses Train_Clean_Data/<dataset>/labels_before_gold_split.csv instead of labels.csv.
# Use this only when you want to compare against a train set that still contains the gold rows.
# Because default validation comes from Gold_Data, this ablation can overlap with validation and should not be used as the headline baseline.

python scripts/Baseline_Exp.py --dataset STL --include-gold-in-train --early-stop-patience 3 --selection-metric val-loss
python scripts/Baseline_Exp.py --dataset Flower_102 --include-gold-in-train --early-stop-patience 3 --selection-metric val-loss

# Optional Flower_102 run with larger image resolution if GPU memory allows.
# Default Flower_102 image size is 224; STL defaults to its native 96.

python scripts/Baseline_Exp.py --dataset Flower_102 --image-size 384 --early-stop-patience 3 --selection-metric val-loss



---------------------------------------------------------------------------------------------------------------------------------------------
# Create train noisy datasets from Train_Clean_Data labels.
# Train_Clean_Data/<dataset>/labels.csv has selected gold rows removed, so the noisy train data also excludes gold rows.
# Feature noise keeps the same labels as Train_Clean_Data and only changes image pixels.
# Label noise keeps the same images and randomly changes a configurable fraction of labels to another existing class.
# Customize noise with --value:
# blur: Gaussian blur radius
# brightness: brightness factor
# gaussian: pixel noise sigma
# label_shuffle: fraction of labels to change
# Output layout:
# Image_Data/Train_Noise_Data/<dataset>/feature_noise/<noise_type>_<value>/
# Image_Data/Train_Noise_Data/<dataset>/label_noise/label_shuffle_<value>/

# STL feature noise examples. Change --value to choose a different strength.
# Use --overwrite after changing the STL gold split, because train labels have changed.
python scripts/Data_Prepration/make_noisy_data.py --dataset STL --split train --noise-type blur --value 3
python scripts/Data_Prepration/make_noisy_data.py --dataset STL --split train --noise-type brightness --value 0.75
python scripts/Data_Prepration/make_noisy_data.py --dataset STL --split train --noise-type gaussian --value 30

python scripts/Data_Prepration/make_noisy_data.py --dataset STL --split train --noise-type blur --value 3 --overwrite

# STL label noise example. Change --value to choose a different shuffle fraction.
python scripts/Data_Prepration/make_noisy_data.py --dataset STL --split train --noise-type label_shuffle --value 0.2

# Flower_102 feature noise examples. Change --value to choose a different strength.
# Use --overwrite after repairing Flower_102 gold data, because train labels have changed.
python scripts/Data_Prepration/make_noisy_data.py --dataset Flower_102 --split train --noise-type blur --value 3 --overwrite
python scripts/Data_Prepration/make_noisy_data.py --dataset Flower_102 --split train --noise-type brightness --value 0.75 --overwrite
python scripts/Data_Prepration/make_noisy_data.py --dataset Flower_102 --split train --noise-type gaussian --value 30 --overwrite

# Flower_102 label noise example. Change --value to choose a different shuffle fraction.
python scripts/Data_Prepration/make_noisy_data.py --dataset Flower_102 --split train --noise-type label_shuffle --value 0.2 --overwrite


---------------------------------------------------------------------------------------------------------------------------------------------
# Baseline train-noise / test-clean experiments.
# This uses the generic Baseline_Exp.py entrypoint with --train-dir pointing to a noisy train dataset.
# Test data remains clean: Image_Data/Test_Clean_Data/<dataset>/labels.csv.
# Checkpoint selection uses the same clean Gold_Data validation policy as clean baselines and gold-guided runs.
# Results are saved in separate subfolders:
# Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/<dataset>/feature_noise/<noise_run>/
# Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/<dataset>/label_noise/<noise_run>/

# STL train feature noise -> clean test.
python scripts/Baseline_Exp.py --dataset STL --train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 --test-dir Image_Data/Test_Clean_Data/STL --early-stop-patience 3 --selection-metric val-loss
python scripts/Baseline_Exp.py --dataset STL --train-dir Image_Data/Train_Noise_Data/STL/feature_noise/brightness_0p75 --test-dir Image_Data/Test_Clean_Data/STL --early-stop-patience 3 --selection-metric val-loss
python scripts/Baseline_Exp.py --dataset STL --train-dir Image_Data/Train_Noise_Data/STL/feature_noise/gaussian_30p0 --test-dir Image_Data/Test_Clean_Data/STL --early-stop-patience 3 --selection-metric val-loss

# STL train label noise -> clean test.
python scripts/Baseline_Exp.py --dataset STL --train-dir Image_Data/Train_Noise_Data/STL/label_noise/label_shuffle_0p2 --test-dir Image_Data/Test_Clean_Data/STL --early-stop-patience 3 --selection-metric val-loss

# Flower_102 train feature noise -> clean test.
python scripts/Baseline_Exp.py --dataset Flower_102 --train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/blur_3p0 --test-dir Image_Data/Test_Clean_Data/Flower_102 --early-stop-patience 3 --selection-metric val-loss
python scripts/Baseline_Exp.py --dataset Flower_102 --train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 --test-dir Image_Data/Test_Clean_Data/Flower_102 --early-stop-patience 3 --selection-metric val-loss
python scripts/Baseline_Exp.py --dataset Flower_102 --train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 --test-dir Image_Data/Test_Clean_Data/Flower_102 --early-stop-patience 3 --selection-metric val-loss

# Flower_102 train label noise -> clean test.
python scripts/Baseline_Exp.py --dataset Flower_102 --train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 --test-dir Image_Data/Test_Clean_Data/Flower_102 --early-stop-patience 3 --selection-metric val-loss


---------------------------------------------------------------------------------------------------------------------------------------------
# Step 1: train frozen gold evaluators from Image_Data/Gold_Data/<dataset>/labels.csv.
# By default, this uses a stratified 90/10 gold split:
# 90% trains the evaluator, and 10% is held out as gold validation.
# The held-out gold validation split is not used for gradient updates, but it selects the best evaluator checkpoint.
# Each dataset gets one evaluator checkpoint:
# Evaluators/<dataset>/model.pt
# The saved evaluator checkpoint is the best gold-validation model.
# Re-running a command overwrites that dataset's evaluator checkpoint and logs.
# Recommended default gold evaluator training uses epochs=20 with best-validation checkpoint selection.
# For stronger feature-noise experiments, use a larger repaired STL gold set and train the evaluator longer
# with feature-noise augmentation. This should improve the evaluator/prototype signal on blurred/noisy images.
# Flower_102 benefits from a slower, longer evaluator run after gold repair.

python scripts/Train_Gold_Evaluator.py --dataset STL --epochs 20
python scripts/Train_Gold_Evaluator.py --dataset STL --epochs 40 --lr 0.0005 --selection-metric val-loss --train-augmentation feature-noise --early-stop-patience 6 --early-stop-min-delta 0.001
python scripts/Train_Gold_Evaluator.py --dataset Flower_102 --epochs 20 --lr 0.0005

# Step 2: gold-guided critical learning experiments from PDF Sections 1.2-1.5.
# These commands load the fixed evaluator checkpoint from Step 1, freeze it, and train a separate learner on mixed/noisy train data.
# Test data remains clean: Image_Data/Test_Clean_Data/<dataset>/labels.csv.
# By default, 10% of Gold_Data is held out as clean validation for model selection and early stopping.
# This uses the same stratified split as Step 1, so the validation gold was not used to train the evaluator.
# The remaining 90% of Gold_Data is used for prototypes and gold stability loss.
# Default controller hyperparameters:
# beta=0.2 puts more weight on feature-prototype consistency than prediction probability.
# The evaluator computes a gold-consistency score; the critical controller maps that score into sample control.
# controller-mode=positive maps tanh control to [0, 1], so the model can learn less or skip a sample, but never negatively fit it.
# min-control=0.0 allows low-consistency samples to contribute zero gradient.
# alpha=2.0 controls controller sharpness, and tau=0.45 is the neutral gold-consistency threshold.
# lambda-gold=0.1 controls gold stability loss strength, and grad-clip=5.0 clips learner gradients.
# early-stop-patience=3 can be used for validation-selected checkpointing in feature noise runs.
# For fair headline comparisons with baseline, use the same selection metric and report test_accuracy_at_selected_epoch.
# For method-analysis runs that ask "how strong can this method get on noisy training data",
# keep the full epoch budget and report max_test_accuracy_observed / max_test_epoch_observed as a diagnostic.
# model.pt is the validation-selected checkpoint.
# model_peak_test.pt is a diagnostic peak clean-test checkpoint; use it for ablation analysis, not deployment selection.
# Gold artifacts are not copied into each run by default because Evaluators/<dataset>/model.pt and the
# computed gold prototypes are shared across runs with the same dataset/gold split/evaluator.
# Pass --save-gold-artifacts only when you need a self-contained debug folder.
# By default, checkpoint selection and early stopping use gold validation accuracy.
# If the held-out gold validation set is small and accuracy is too discrete, pass --selection-metric val-loss.
# To show that the method reduces train-noise damage, compare:
# clean-train baseline, noisy-train baseline, and gold-guided noisy-train peak/selected test accuracy.
# A useful summary is recovery_ratio = (ours - noisy_baseline) / (clean_baseline - noisy_baseline).
# lr remains 0.001 for AdamW; the old base_lr/gold_lr=0.05 are not used as defaults without the old scheduler setup.
# To test the old LR profile, pass --optimizer sgd --base-lr 0.05 --min-lr 0.001.
# This enables SGD momentum with cosine LR decay for the learner.
# Results are saved in separate subfolders:
# Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/<dataset>/feature_noise/<noise_run>/
# Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/<dataset>/label_noise/<noise_run>/

# STL train feature noise -> clean test with gold-guided critical learning.
# These are generic template commands using the script defaults. Use them when starting a fresh sweep.
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 --early-stop-patience 3 --selection-metric val-loss
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/brightness_0p75 --early-stop-patience 3 --selection-metric val-loss
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/gaussian_30p0 --early-stop-patience 3 --selection-metric val-loss

# Current selected STL feature-noise settings from the completed sweeps.
# These values are experiment-selected hyperparameters, not fixed method defaults.
# The result folders are aligned with docs/stl_blur_3p0_results.md,
# docs/stl_brightness_0p75_results.md, and docs/stl_gaussian_30p0_results.md.
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 --early-stop-patience 3 --selection-metric val-loss --beta 0.9 --lambda-gold 0.15 --run-name feature_noise/blur/blur_3p0_method_beta09_lg015
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/brightness_0p75 --early-stop-patience 3 --selection-metric val-loss --beta 0.9 --lambda-gold 0.1 --run-name feature_noise/brightness/brightness_0p75_method_beta09_lg01
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/gaussian_30p0 --early-stop-patience 3 --selection-metric val-loss --beta 0.5 --lambda-gold 0.15 --run-name feature_noise/gaussian/gaussian_30p0_method_beta05_lg015

# STL feature-noise method-analysis run: train all epochs, then compare peak test accuracy in metrics.json.
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 --run-name feature_noise/blur/blur_3p0_strong_gold_fixed_alpha

# Dynamic non-negative elastic controller from the PDF extension after removing negative anti-fitting:
# raw_i = tanh(alpha_k * (r_i - tau)), then controller-mode=positive maps raw_i to [0, 1].
# Increasing alpha_k makes the controller softer early and more selective after the learner has stabilized.
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 --alpha-schedule linear --alpha-start 0.5 --alpha 2.0 --run-name feature_noise/blur/blur_3p0_strong_gold_dynamic_alpha_linear

# Optional feature-noise ablation: freeze learner BatchNorm running stats during noisy mixed-data training.
# This tests whether clean-test instability comes from BatchNorm statistics drifting toward the blur domain.
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 --freeze-learner-batch-norm --run-name feature_noise/blur/blur_3p0_strong_gold_freeze_bn

# Optional validation-selected variant for small gold validation sets: select/early-stop by validation loss.
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 --early-stop-patience 3 --selection-metric val-loss --run-name feature_noise/blur/blur_3p0_strong_gold_val_loss_selection

# STL train label noise -> clean test with gold-guided critical learning.
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/label_noise/label_shuffle_0p2

# Flower_102 train feature noise -> clean test with gold-guided critical learning.
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/blur_3p0 --early-stop-patience 3
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/brightness_0p75 --early-stop-patience 3
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/feature_noise/gaussian_30p0 --early-stop-patience 3

# Flower_102 train label noise -> clean test with gold-guided critical learning.
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2

# Optional ablation for softer suppression. This keeps at least 20% of each sample's gradient.
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/label_noise/label_shuffle_0p2 --controller-mode positive --min-control 0.2
python scripts/Gold_Guided_Critical_Learning.py --dataset Flower_102 --mixed-train-dir Image_Data/Train_Noise_Data/Flower_102/label_noise/label_shuffle_0p2 --controller-mode positive --min-control 0.2

# Optional old-profile ablation for STL blur.
# Maps old params as beta=0.2, tau=0.45, alpha=2.0, lambda_g=0.1, anti_weight=0.0, grad_clip=5.0, epochs=20, base_lr=0.05, min_lr=0.001.
# gamma_gold=0.5 is not mapped here because the current PDF 1.2-1.5 implementation does not have a separate gamma_gold term.
python scripts/Gold_Guided_Critical_Learning.py --dataset STL --mixed-train-dir Image_Data/Train_Noise_Data/STL/feature_noise/blur_3p0 --optimizer sgd --base-lr 0.05 --min-lr 0.001
