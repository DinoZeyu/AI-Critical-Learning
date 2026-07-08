# Experiment Commands

Use the seeded runners for all current experiments:

```bash
bash scripts/seeded_runs/run_seed_experiments.sh 42
bash scripts/seeded_runs/run_seed_experiments.sh 22
bash scripts/seeded_runs/run_seed_experiments.sh 62
```

Run all configured seeds:

```bash
bash scripts/seeded_runs/run_all_seeds.sh
```

Run from inside a seed container:

```bash
bash seed_22/run_all_experiments.sh
bash seed_42/run_all_experiments.sh
bash seed_62/run_all_experiments.sh
```

Rerun canonical seed 42 supplemental ablations and replace the existing
standard results:

```bash
REPLACE_EXISTING=1 bash seed_42/run_all_experiments.sh supplemental-ablation
```

Seed-owned artifacts are stored under root-level seed folders:

```text
seed_22/
seed_42/
seed_62/
```

Each seed folder uses the same internal layout:

```text
seed_<seed>/Train_Noise_Data/
seed_<seed>/Gold_Evaluators/
seed_<seed>/Experiments_Results/
seed_<seed>/Supplemental_Ablation_Results/
seed_<seed>/Noise_Baseline/
seed_<seed>/docs/
seed_<seed>/run_all_experiments.sh
```

Gold-guided and sweep results are stored directly under
`seed_<seed>/Experiments_Results/<dataset>/...`.

Shared clean/gold image data remains under root-level `Image_Data/`.
The shared train-clean/test-clean baseline results remain under root-level
`Clean_Baseline/`.

The GGCL and supplemental ablation parameter lists are stored in:

```text
scripts/seeded_runs/manifests/ggcl_seed42_runs.tsv
scripts/seeded_runs/manifests/supplemental_ablation_seed42_runs.tsv
```
