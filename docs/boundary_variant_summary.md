# Boundary-Variant Ablation Summary

This note summarizes the compact boundary-variant comparison for the ablation section using the same selection rule as the paper's main sweep tables.

The selected reference configuration is the paper-standard fixed-configuration choice: for each dataset--corruption setting, choose the single fixed `(β, λ_G)` sweep run with the highest three-seed mean selected clean-test accuracy. This is the Method B comparison used by the current Table II/III/IV bold entries, not the per-seed oracle in `multi_seed_main_setting_stats.csv`.

Inputs:

- `docs/multi_seed_ggcl_exact_run_stats.csv` for the fixed `(β, λ_G)` selected sweep configuration.
- `docs/multi_seed_supplemental_exact_run_stats.csv` for the boundary variants.

For each of the 14 dataset--corruption settings and each boundary variant, the drop is:

```text
Drop = Sel_selected_config - Sel_boundary_variant
```

Positive values mean the selected sweep configuration has higher selected clean-test accuracy. Drops are reported in percentage points.

## Compact summary

| Variant | Boundary setting | Worse settings | Mean drop (p.p.) | Drop range (p.p.) |
| --- | --- | ---: | ---: | ---: |
| $\lambda_G = 0$ | No gold stability loss | 14/14 | 6.4 | 3.5--11.3 |
| $\beta = 0$ | Similarity only | 12/14 | 2.9 | -0.7--5.8 |
| $\beta = 1$ | Confidence only | 10/14 | 1.2 | -1.0--3.4 |

CSV outputs:

- Summary CSV: `docs/boundary_variant_summary.csv`
- Per-setting drop CSV: `docs/boundary_variant_setting_drops.csv`

## LaTeX table

```latex
\begin{table}[t]
\centering
\caption{Boundary-variant drops relative to the selected fixed-configuration sweep entry across the 14 dataset--corruption settings. Drops are computed as selected clean-test accuracy of the selected sweep configuration minus that of the boundary variant; positive values mean the selected configuration performs better.}
\label{tab:boundary_variant_summary}
\begin{tabular}{llccc}
\toprule
Variant & Boundary setting & Worse settings & Mean drop (p.p.) & Drop range (p.p.) \\
\midrule
$\lambda_G=0$ & No gold stability loss & 14/14 & 6.4 & 3.5--11.3 \\
$\beta=0$ & Similarity only & 12/14 & 2.9 & -0.7--5.8 \\
$\beta=1$ & Confidence only & 10/14 & 1.2 & -1.0--3.4 \\
\bottomrule
\end{tabular}
\end{table}
```

## Computation notes

- `Worse settings` counts settings with `Drop > 0`.
- `Mean drop` is the arithmetic mean of the 14 per-setting drops after converting accuracy differences to percentage points.
- `Drop range` is the minimum and maximum per-setting drop over the 14 settings.
- The selected reference configuration is computed from exact fixed-run means in `docs/multi_seed_ggcl_exact_run_stats.csv`; `docs/multi_seed_main_setting_stats.csv` is not used because it is a per-seed oracle summary.
