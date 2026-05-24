# Notebook Conversion Tracking: Python → R Quarto

This file tracks the conversion of Python Jupyter notebooks to R Quarto (`.qmd`) notebooks
within the `05-deep-causal-learners` series.

---

## Completed Conversions

### `05-04-01-DeepCausalML-timeseries-neural-granger-cusuality-r.qmd`

| Field | Details |
|---|---|
| **Source notebook** | `01_neural_granger_causality_models.ipynb` |
| **Output file** | `inst/notebooks/05-deep-causal-learners/05-00-deep-causal-learning-introduction-r_files/05-04-01-DeepCausalML-timeseries-neural-granger-cusuality-r.qmd` |
| **Conversion date** | 2026-05-19 |
| **Converted by** | Cursor AI Agent |
| **Format reference** | `05-03-05-DeepCausalML-causal-structural-learning-regularization-CASTLE-r.qmd` |

#### Content Overview

The notebook introduces **Neural Granger Causality** models and demonstrates their use
on S&P 500 sector ETF daily log-return data (2018–2023).

**Sections:**

1. Theory — Classical Granger Causality and the Neural Extension
2. The Four Models: cMLP, cLSTM, EconomySRU, NRI (with LaTeX formulations)
3. Implementation in R
4. Data: S&P 500 Sector ETF Daily Log Returns
5. Data Preparation: Lag Dataset
6. Fit All Four Neural Granger Models
7. Training Curves
8. Inferred Causal Matrices (heatmaps per model)
9. Validation: Model Performance Comparison
10. Consensus Causal Graph
11. Notes and Extensions
12. Summary and Conclusions
13. Resources
14. Scientific Terminology for Beginners

#### Python → R Translation Notes

| Python | R Equivalent | Notes |
|---|---|---|
| `yfinance.download()` | `tidyquant::tq_get()` | Same Yahoo Finance data source |
| `sklearn.preprocessing.StandardScaler` | `base::scale()` | Built-in R standardisation |
| `numpy.log(raw / raw.shift(1))` | `log(. / lag(.))` via `dplyr::mutate` | Same log-return formula |
| `torch.nn.Module` (manual) | `RCausalML::neural_granger_ml()` | Package wrapper handles all four models |
| `ComponentMLP`, `cMLP` class | `.deepnet_ngc_cmlp()` (internal) | Implemented in `R/causalDeepNet.R` |
| `ComponentLSTM`, `cLSTM` class | `.deepnet_ngc_clstm()` (internal) | Implemented in `R/causalDeepNet.R` |
| `EconomySRU` class | `.deepnet_ngc_economy_sru()` (internal) | Implemented in `R/causalDeepNet.R` |
| `MLPEncoder`, `MLPDecoder`, `NRI` class | `.deepnet_ngc_nri()` (internal) | Implemented in `R/causalDeepNet.R` |
| `train_model()` function | Internal `.deepnet_ngc_train_model()` | Called by `neural_granger_ml()` |
| `seaborn.heatmap()` | `ggplot2::geom_tile()` | Custom `plot_causal_heatmap()` helper |
| `matplotlib.pyplot.subplots()` | `patchwork` library | 2×2 facet layout |
| `pandas.DataFrame.describe()` | `sapply()` + summary stats | Base R equivalent |
| `numpy.linalg.norm(group, "fro")` | `norm(group, "F")` / `sqrt(sum(v^2))` | Already handled inside `causal_matrix()` |
| `DEVICE = torch.device(...)` | `device_use <- if (cuda_is_available()) "cuda" else "cpu"` | R torch device selection |

#### Key RCausalML Functions Used

| Function | Description |
|---|---|
| `neural_granger_ml()` | Unified training interface for all four models |
| `neuralGrangerML()` | Alias for `neural_granger_ml()` |
| `predict.neural_granger_ml()` | S3 predict method for the fitted object |

The returned `neural_granger_ml` object contains:
- `$models` — list of trained `nn_module` objects
- `$histories` — per-model list with `train_loss` and `val_loss` vectors
- `$val_mse` — named numeric vector of final validation MSE per model
- `$causal_matrices` — named list of $d \times d$ causal strength matrices
- `$lag`, `$var_names`, `$device` — metadata

#### Packages Required

```r
c('tidyverse', 'plyr', 'RCausalML', 'torch', 'tidyquant', 'reshape2', 'scales', 'patchwork')
```

#### Known Differences from Python Notebook

1. **NRI model** — Python version crashed kernel during SRU/NRI training due to CUDA memory issues.
   R version runs on CPU by default and is stable.
2. **Evaluation cell** — Python Cell 15 (`eval_mse`) and Cell 16 (plots) depend on completed training;
   in R these are folded into post-training cells with the `ngc_fit` object.
3. **Consensus graph** — Added as an R-specific section (not in the original Python notebook) to
   summarise cross-model agreement using min-max normalised average causal matrices.
4. **`build_lag_dataset` column order** — Python uses `window.T.flatten()` (column-major in C order);
   R replicates this with `as.vector(t(window))` so lag features are interleaved identically.

---

### `05-DeepCausalML-timeseries-structural-causal-model-SMC-r.qmd`

| Field | Details |
|---|---|
| **Source notebook** | `02_structural_causal_model_SCMs.ipynb` |
| **Output file** | `inst/notebooks/05-deep-causal-learners/05-DeepCausalML-timeseries-structural-causal-model-SMC-r.qmd` |
| **Conversion date** | 2026-05-19 |
| **Converted by** | Cursor AI Agent |
| **Format reference** | `05-03-05-DeepCausalML-causal-structural-learning-regularization-CASTLE-r.qmd` |

#### Content Overview

The notebook introduces **Structural Causal Models (SCMs) with Deep Components** and demonstrates three models
on S&P 500 sector ETF daily log-return data (2018–2024).

**Sections:**

1. Theory — SCMs, Pearl's hierarchy, Deep SCMs
2. Overview of three models: DeepSCM, DECI, DYNOTEARS (with LaTeX formulations)
3. Implementation in R
4. Data: S&P 500 Sector ETF Daily Log Returns
5. Data Preparation: Lag Dataset (3-D array)
6. Utilities: Acyclicity penalty, threshold helper, fixed-graph heuristic
7. Train DeepSCM (fixed graph from correlation)
8. Train DECI (joint graph + structural equations)
9. Intervention Analysis and ATE Estimation
10. Train DYNOTEARS (lagged DAG-constrained discovery)
11. Graph Recovery Evaluation and Visual Comparison
12. Notes and Extensions
13. Summary and Conclusions
14. Resources
15. Scientific Terminology for Beginners

#### Python → R Translation Notes

| Python | R Equivalent | Notes |
|---|---|---|
| `yfinance.download()` | `tidyquant::tq_get()` | Same Yahoo Finance data source |
| `sklearn.preprocessing.StandardScaler` | `base::scale()` | Built-in R standardisation |
| `numpy.log(raw / raw.shift(1))` | `log(. / lag(.))` via `dplyr::mutate` | Same log-return formula |
| `torch.nn.Module` (manual) | `RCausalML::deep_scm()` / `deci_model()` / `dynotears()` | Package wrappers handle all three models |
| `StructuralEquationNet`, `NoiseEncoder`, `DeepSCM` class | `.deepnet_deep_scm_module()` (internal) | Implemented in `R/causalDeepNet.R` |
| `DECIAdjacency`, `DECIEncoder`, `DECIDecoder`, `DECI` class | `.deepnet_deci_module()` (internal) | Implemented in `R/causalDeepNet.R` |
| `DynoTEARS` class + `train_dynotears()` | `.deepnet_dynotears_module()` (internal) | Implemented in `R/causalDeepNet.R` |
| `deci.compute_ate()` | `RCausalML::ate_deci()` | Public API wrapper |
| `deep_scm.intervene()` | `RCausalML::intervene_deep_scm()` | Public API wrapper |
| `seaborn.heatmap()` | `ggplot2::geom_tile()` via `plot_scm_dag()` | Built into {RCausalML} |
| `matplotlib.pyplot.subplots()` | `patchwork` library | Multi-panel layout |
| `networkx.DiGraph` / `nx.draw_networkx_*` | `igraph::graph_from_adjacency_matrix()` | Graph stats + circular layout plots |
| `evaluate_graph_recovery()` Python function | `RCausalML::evaluate_graph_recovery()` | Precision / Recall / F1 / SHD |
| `DEVICE = torch.device(...)` | `device_use <- if (cuda_is_available()) "cuda" else "cpu"` | R torch device selection |

#### Key RCausalML Functions Used

| Function | Description |
|---|---|
| `deep_scm()` / `deepSCM()` | Train fixed-graph variational SCM |
| `deci_model()` / `deciModel()` | Train DECI (joint graph + structural equations) |
| `dynotears()` / `dynoTEARS()` | Train DYNOTEARS (lagged DAG-constrained discovery) |
| `predict.deep_scm()` | S3 predict for DeepSCM |
| `predict.deci_model()` | S3 predict for DECI |
| `predict.dynotears()` | S3 predict for DYNOTEARS |
| `intervene_deep_scm()` | do-calculus intervention on DeepSCM |
| `ate_deci()` | ATE estimation via DECI Monte-Carlo |
| `plot_scm_dag()` | Heatmap visualisation of adjacency matrix |
| `evaluate_graph_recovery()` | Precision / Recall / F1 / SHD vs ground truth |

#### Packages Required

```r
c('tidyverse', 'plyr', 'RCausalML', 'torch', 'tidyquant', 'reshape2', 'scales', 'patchwork', 'igraph')
```

#### Known Differences from Python Notebook

1. **Data loaders** — Python uses `torch.utils.data.DataLoader`; R uses direct array slicing into `torch_tensor()` inside the {RCausalML} training loops.
2. **Graph visualisation** — Python uses `networkx` + `matplotlib`; R uses `igraph` for network diagrams and `ggplot2`/`patchwork` for heatmaps.
3. **`evaluate_graph_recovery`** — Python defines the function inline in Cell 23; R calls the {RCausalML} exported version with the same logic.
4. **`plot_dag` helper** — Python uses `seaborn.heatmap`; R uses `RCausalML::plot_scm_dag()` (ggplot2-backed).
5. **Intervention cell** — Python Cell 19 mixes DECI and DeepSCM interventions; the R version separates them into `ate_deci()` and `intervene_deep_scm()` calls for clarity.

---

## Pending / Planned Conversions

| Python notebook | Target R Quarto file | Status |
|---|---|---|
| (other notebooks in series) | TBD | Pending |

---

## Style Reference

All notebooks in the `05-deep-causal-learners` series follow the format established in:

```
inst/notebooks/05-deep-causal-learners/
  05-03-05-DeepCausalML-causal-structural-learning-regularization-CASTLE-r.qmd
```

Key formatting conventions:
- Top-level `# Title {.unnumbered}` header
- `## Overview` section with theory and LaTeX equations
- `## Implementation in R` section
- Quarto chunk labels: `{r, label: kebab-case-name, warning: false}`
- `devtools::load_all()` guard for local development
- Packages listed separately (list → install → verify → load pattern)
- Summary, Resources, and Scientific Terminology for Beginners sections at end
