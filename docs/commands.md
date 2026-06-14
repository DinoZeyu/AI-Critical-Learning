# Experiment Commands

Dataset-specific command files:

- [STL commands](STL/commands.md)
- [Flower_102 commands](Flower_102/commands.md)

Shared notes:

- Clean-train / clean-test baselines save to `Experiments_Results/Train_Clean_Test_Clean/<dataset>/`.
- Noisy-train / clean-test baselines save to `Experiments_Results/Train_Noise_Test_Clean/Baseline_Exp/<dataset>/...`.
- Gold-guided critical learning runs save to `Experiments_Results/Train_Noise_Test_Clean/Gold_Guided_Critical_Learning/<dataset>/...`.
- Headline comparisons should use `test_accuracy_at_selected_epoch`, selected by clean gold validation loss unless otherwise stated.
- `max_test_accuracy_observed` is diagnostic/oracle-style and should be used for analysis, not deployment checkpoint selection.
