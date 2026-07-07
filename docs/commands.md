# Experiment Commands

Dataset-specific command files:

- [STL commands](STL/commands.md)
- [Flower_102 commands](Flower_102/commands.md)
- [Supplemental ablation results](ablation/supplemental_ablation_results.md)

Shared notes:

- Clean-train / clean-test baselines save to `Experiments_Results/Train_Clean_Test_Clean/<dataset>/`.
- Noisy-train / clean-test baselines save to `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/<dataset>/...`.
- Gold-guided critical learning runs save to `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/<dataset>/...`.
- Gold-guided critical learning commands should explicitly use `--gold-evaluator-checkpoint Gold_Evaluators/<dataset>/model.pt`.
- Historical metrics that record `Evaluators/<dataset>/model.pt` refer to the same evaluator directory before it was renamed to `Gold_Evaluators/`.
- Headline comparisons should use `test_accuracy_at_selected_epoch`, selected by clean gold validation loss unless otherwise stated.
- `max_test_accuracy_observed` is diagnostic/oracle-style and should be used for analysis, not deployment checkpoint selection.

Supplemental ablation batch:

```bash
./run_supplemental_ablation_experiments.sh
```

This runs the 42 supplemental ablations: `lambda_gold=0`, `beta=0`, and
`beta=1` across both datasets and all seven noise settings. Final results are
stored under `Supplemental_Ablation_Results/<dataset>/<ablation>/...`.
