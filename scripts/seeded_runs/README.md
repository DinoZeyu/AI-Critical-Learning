# Seeded Experiment Runs

This folder contains the runnable command entry points for the three-seed
experiment reorganization.

The root-level seed containers are:

```text
seed_22/
seed_42/
seed_62/
```

Each seed container owns the artifacts affected by that seed:

```text
seed_<seed>/Train_Noise_Data/
seed_<seed>/Gold_Evaluators/
seed_<seed>/Experiments_Results/
seed_<seed>/Supplemental_Ablation_Results/
seed_<seed>/Noise_Baseline/
seed_<seed>/docs/
seed_<seed>/run_all_experiments.sh
```

Gold-guided and sweep results are stored directly by dataset under
`seed_<seed>/Experiments_Results/`.

The fixed shared data remains in the repository root:

```text
Image_Data/Train_Clean_Data/
Image_Data/Test_Clean_Data/
Image_Data/Gold_Data/
Clean_Baseline/
```

Run one full seed:

```bash
bash scripts/seeded_runs/run_seed_experiments.sh 22
```

Run selected stages only:

```bash
bash scripts/seeded_runs/run_seed_experiments.sh 22 "noise-data gold-evaluators"
bash scripts/seeded_runs/run_seed_experiments.sh 62 "noise-baselines ggcl supplemental-ablation"
```

Run all configured seeds:

```bash
bash scripts/seeded_runs/run_all_seeds.sh
```

Each seed container can also include a local one-command launcher:

```bash
bash seed_42/run_all_experiments.sh
```

By default existing outputs are skipped. Set `SKIP_EXISTING=0` to fail on an
existing output instead of skipping it. Set `REPLACE_EXISTING=1` when you
intentionally want to rerun a canonical result and replace the existing final
output only after the new raw result finishes successfully.

Rerun seed 42 supplemental ablations in the standard output locations:

```bash
REPLACE_EXISTING=1 bash seed_42/run_all_experiments.sh supplemental-ablation
```
