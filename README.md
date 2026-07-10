# AI-Critical-Learning

Gold-Guided Critical Learning experiments for noisy image classification.

## Algorithm: GCL

**Inputs:** gold set $D_G$, noisy/mixed training set $D_t$, learner
$M_\theta$, learning rate $\eta$, threshold $\tau$, sharpness $\alpha$,
probability-similarity tradeoff $\beta$, minimum confidence $c_{\min}$, gold
stability weight $\lambda_G$, and training steps $K$.

**Output:** updated learner parameters $\theta_K$.

1. Train on $D_G$ to obtain the gold reference
   $\theta_G=\arg\min_\theta L_{\mathrm{ref}}(\theta;D_G)$.
2. Initialize the learner with $\theta_0 \leftarrow \theta_G$ and freeze a
   gold evaluator $\bar{\theta}_G \leftarrow \theta_G$.
3. Compute class prototypes $\{\mu_y^G\}$ from $D_G$ using
   $M_{\bar{\theta}_G}$.
4. For each training step $k=0,\ldots,K-1$:
   - Sample a mixed batch $B \subset D_t$ and a gold batch $B_G \subset D_G$.
   - For each $(x_i,y_i)\in B$, compute gold confidence from evaluator
     probability and prototype similarity:

     $$
     r_i=\beta r_i^{\mathrm{prob}}+(1-\beta)r_i^{\mathrm{sim}}.
     $$

   - Convert confidence into a smooth sample weight:

     $$
     \tilde{c}_i=\tanh(\alpha(r_i-\tau)),\qquad
     c_i=c_{\min}+(1-c_{\min})(\tilde{c}_i+1)/2.
     $$

   - Compute the gold stability loss:

     $$
     L_{\mathrm{stab}}(\theta_k;B_G)
     =\frac{1}{|B_G|}\sum_{j\in B_G}
     \ell(M_{\theta_k}(x_j^G),y_j^G).
     $$

   - Update the learner with confidence-weighted mixed gradients and gold
     stability regularization:

     $$
     g_k=
     \frac{1}{|B|}\sum_{i\in B}c_i\nabla_\theta
     \ell(M_{\theta_k}(x_i),y_i)
     +\lambda_G\nabla_\theta L_{\mathrm{stab}}(\theta_k;B_G),
     \qquad
     \theta_{k+1}\leftarrow \theta_k-\eta g_k.
     $$

## Environment

Use the project virtual environment, not system Python:

```bash
.venv/bin/python --version
.venv/bin/python Visualization/make_figures.py
```

Equivalent `uv` command when the environment is already synced:

```bash
UV_CACHE_DIR=.cache/uv uv run --no-sync python Visualization/make_figures.py
```

## Layout

```text
Image_Data/                 shared clean/gold image data
Clean_Baseline/             shared clean-training baselines
seed_22/, seed_42/, seed_62/ per-seed data, models, metrics, and docs
docs/                       multi-seed result summaries and CSV tables
Visualization/              figure-generation code and output figures
scripts/seeded_runs/        seed orchestration scripts
```

Generated noisy image pixels, model checkpoints, logs, and raw staging folders
are local-only. Metrics, histories, docs, and figure scripts are the relevant
tracked artifacts.

## Experiments

Run all configured seeds:

```bash
bash scripts/seeded_runs/run_all_seeds.sh
```

Run one seed:

```bash
bash scripts/seeded_runs/run_seed_experiments.sh 22
```

Run selected stages:

```bash
bash scripts/seeded_runs/run_seed_experiments.sh 62 "noise-baselines ggcl supplemental-ablation"
```

Existing completed outputs are skipped by default. Use `REPLACE_EXISTING=1`
only when intentionally replacing a finished canonical result.

## Results

Main multi-seed summaries:

- `docs/multi_seed_results_summary.md`
- `docs/multi_seed_main_setting_stats.csv`
- `docs/multi_seed_ggcl_exact_run_stats.csv`
- `docs/multi_seed_supplemental_exact_run_stats.csv`

Per-seed result docs live under `seed_<seed>/docs/`.

## Figures

Generate the single-column PDF figure for LaTeX:

```bash
.venv/bin/python Visualization/make_figures.py
```

Output is written to `Visualization/Figures/`. Use the PDF file in `main.tex`.
