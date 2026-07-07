# Seed 42 Train Noise Data

Generated noisy image files are local-only and are not committed to Git.

Each noise configuration keeps a `labels.csv` manifest. That file records the
source image index/path, assigned label, noise type, noise value, and, for label
noise, whether label noise was applied.

The image files can be regenerated from the shared clean data with:

```bash
bash scripts/seeded_runs/run_seed_experiments.sh 42 noise-data
```
