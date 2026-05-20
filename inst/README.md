# CausalML-R Example Notebooks (Quarto)

These examples mirror the [Python CausalML examples](https://causalml.readthedocs.io/en/latest/examples.html) and [GitHub docs/examples](https://github.com/uber/causalml/tree/master/docs/examples). Each `.qmd` runs in R using the **CausalML** R package and includes comparison notes with Python CausalML where relevant.

## Data

- **Synthetic data** is generated in R via `synthetic_data()` and `make_uplift_classification()` (no download).
- **Real/semi-synthetic data** (e.g. IHDP, card.csv) can be downloaded from the Python CausalML repo:  
  [https://github.com/uber/causalml/tree/master/docs/examples/data](https://github.com/uber/causalml/tree/master/docs/examples/data)  
  Save files into `inst/examples/data/` (or set `data_dir` in each notebook). See [data/README.md](data/README.md).

## How to run

## Installation (required)

Before rendering notebooks, install `RCausalML` and its dependencies from the **package root** (directory containing `DESCRIPTION`):

```r
install_missing <- function(pkgs, repos = getOption("repos")) {
  pkgs <- unique(pkgs)
  pkgs <- pkgs[nzchar(pkgs)]
  pkgs <- setdiff(pkgs, rownames(installed.packages()))
  if (length(pkgs)) install.packages(pkgs, repos = repos)
  invisible(TRUE)
}

desc <- read.dcf("DESCRIPTION")[1, , drop = FALSE]
fields <- intersect(c("Depends", "Imports", "Suggests"), colnames(desc))
deps_raw <- paste(desc[1, fields], collapse = ",")
deps_raw <- gsub("\\s*\\([^\\)]+\\)", "", deps_raw)
deps <- trimws(unlist(strsplit(deps_raw, ",")))
deps <- setdiff(deps, c("R"))

install_missing(c("remotes"))
install_missing(deps)
remotes::install_local(".", dependencies = FALSE, upgrade = "never")
```

From the package root (or from `inst/examples/`):

```bash
quarto render "inst/examples/01-meta-learners-training-estimation-validation-visualization.qmd"
# or render all numbered examples:
quarto render inst/examples/[0-2][0-9]-*.qmd
```

In RStudio: open a `.qmd` and use "Render" or "Run All".

## Example Notebooks

| # | QMD file | Short description |
|---|----------|-------------------|
| 01 | `01-meta-learners-training-estimation-validation-visualization.qmd` | Meta-learners: training, ATE estimation, validation, visualization · [Python](https://causalml.readthedocs.io/en/latest/examples/meta_learners_with_synthetic_data.html) |
| 02 | `02-uplift-trees-synthetic.qmd` | Uplift trees/forests (KL, ED, Chi) with synthetic data · [Python](https://causalml.readthedocs.io/en/latest/examples/uplift_trees_with_synthetic_data.html) |
| 03 | `03-meta-learners-single-multiple-treatment.qmd` | Meta-learners — single and multiple treatment · [Python](https://causalml.readthedocs.io/en/latest/examples/meta_learners_with_synthetic_data_multiple_treatment.html) |
| 04 | `04-uplift-tree-forest-visualization.qmd` | Uplift trees and forests visualization · [Python](https://causalml.readthedocs.io/en/latest/examples/uplift_tree_visualization.html) |
| 05 | `05-model-interpretation-feature-importance-shap.qmd` | Model interpretation with feature importance and SHAP · [Python](https://causalml.readthedocs.io/en/latest/examples/feature_interpretations_example.html) |
| 06 | `06-uplift-curves-tmle.qmd` | Uplift curves with TMLE · [Python](https://causalml.readthedocs.io/en/latest/examples/validation_with_tmle.html) |
| 07 | `07-Dragonnet-TarNet-CRFNet-metalearner.qmd` | DragonNet, TARNet, CFRNet vs meta-learners · [Python](https://causalml.readthedocs.io/en/latest/examples/dragonnet_example.html) |
| 08 | `08-IV-2SLS-NLSYM.qmd` | 2SLS and IV with NLSYM and synthetic data · [Python](https://causalml.readthedocs.io/en/latest/examples/iv_nlsym_synthetic_data.html) |
| 09 | `09-sensitivity-analysis.qmd` | Sensitivity analysis (placebo, random cause, selection bias, random replace) · [Python](https://causalml.readthedocs.io/en/latest/examples/sensitivity_example_with_synthetic_data.html) |
| 10 | `10-counterfactual-unit-selection.qmd` | Unit selection based on counterfactual logic (Li & Pearl 2019) · [Python](https://causalml.readthedocs.io/en/latest/examples/counterfactual_unit_selection.html) |
| 11 | `11-counterfactual-value-estimation.qmd` | Counterfactual value estimation (Li & Pearl 2019) · [Python](https://causalml.readthedocs.io/en/latest/examples/counterfactual_value_optimization.html) |
| 12 | `12-feature-selection-uplift.qmd` | Feature selection for uplift trees (Zhao et al. 2020) · [Python](https://causalml.readthedocs.io/en/latest/examples/feature_selection.html) |
| 13 | `13-policy-learner-binary-treatment.qmd` | Policy learner (Athey & Wager 2018) — binary treatment · [Python](https://causalml.readthedocs.io/en/latest/examples/binary_policy_learner_example.html) |
| 14 | `14-CEVAE-GANITE-metalearner-benchmark.qmd` | CEVAE, GANITE and meta-learners benchmark (IHDP + synthetic) · [Python](https://causalml.readthedocs.io/en/latest/examples/cevae_example.html) |
| 15 | `15-dr-driv-xr-benchmark.qmd` | DR-learner vs DR-IV vs X-learner benchmark · [Python](https://causalml.readthedocs.io/en/latest/examples/dr_learner_with_synthetic_data.html) |
| 16 | `16-meta-learners-benchmark-nie-wager.qmd` | Meta-learner benchmarks (Nie and Wager 2020) · [Python](https://causalml.readthedocs.io/en/latest/examples/benchmark_simulation_studies.html) |
| 17 | `17-causal-trees-forests-treatment-effects-estimation-visualization.qmd` | Causal trees/forests — treatment effects and visualization · [Python](https://causalml.readthedocs.io/en/latest/examples/causal_trees_with_synthetic_data.html) |
| 18 | `18-causal-trees-interpretation-feature-importance-SHAP.qmd` | Causal trees/forests interpretation with feature importance and SHAP · [Python](https://causalml.readthedocs.io/en/latest/examples/causal_trees_interpretation.html) |
| 19 | `19-logistic-regression-data-generation-uplift-classification.qmd` | Logistic regression data generation for uplift classification · [Python](https://causalml.readthedocs.io/en/latest/examples/logistic_regression_based_data_generation_for_uplift_classification.html) |
| 20 | `20-qini-curves-multiple-costly-treatments-arms.qmd` | Qini curves with multiple costly treatment arms · [Python](https://causalml.readthedocs.io/en/latest/examples/qini_curves_for_costly_treatment_arms.html) |
| 21 | `21-propensity-calibration.qmd` | Propensity score calibration · [Python](https://causalml.readthedocs.io/en/latest/examples/calibration.html) |
| 22 | `22-benchmark-semi-synthetic-Schuler-et-al.qmd` | Meta-learner benchmarks with semi-synthetic data (Schuler et al.) · [Python](https://causalml.readthedocs.io/en/latest/examples/benchmark_semi_synthetic_simulation_studies.html) |
| 23 | `23-double-machine-learning-usecases-examples.qmd` | DML use cases: LinearDML, SparseLinearDML, CausalForestDML (synthetic and observational) |
| 24 | `24-orthoiv-driv-use-cases-examples.qmd` | OrthoIV, DMLIV, DRIV — IV-based CATE with instruments |
| 25 | `25-dynamic-double-ml.qmd` | Dynamic Double ML — sequential/panel treatments |
| 26 | `26-weighted-double-ml.qmd` | Weighted Double ML — summarized data, interpretability (raw-data workflow) |
| — | `case_study/01-customer-segmentation-estimate-individualized-Responses-Incentives.qmd` | Case study: customer segmentation and individualized incentive response (EconML + DoWhy style) |
| — | `uplift-classification-data.qmd` | Uplift classification data generation (`make_uplift_classification`) |

## Deep Neural Causal Models (`R/causalDeepNet.R`)

All models live in `R/causalDeepNet.R`. Models marked **torch** require the
R `torch` package (`install.packages("torch")`); models marked **torch / fallback**
use an `nnet` or `ranger` placeholder when `torch` is not installed.

### Treatment-Effect / ITE Estimators

| Function | Class | torch? | Description | Key reference |
|----------|-------|--------|-------------|---------------|
| `cevae()` | `cevae` | torch / fallback | **CEVAE** — Counterfactual VAE. Generative model \(z \sim p(z)\), \(x \sim p(x\|z)\), \(w \sim p(w\|z)\), \(y \sim p(y\|t,z)\). Twin outcome nets handle imbalanced treatment; training optimises ELBO + \(\log q(t\|x)\) + \(\log q(y\|t,x)\). | Louizos et al. (2017) NeurIPS |
| `fit_predict_cevae()` | — | torch / fallback | Convenience wrapper: calls `cevae()` then `predict()` in a single step, matching Python `CEVAE.fit_predict()`. | — |
| `dragonnet()` | `dragonnet` | torch / fallback | **DragonNet** — Shared 3-layer ELU representation with propensity head \(\hat{e}(X)\) and two outcome heads \(\hat{Y}(0)\), \(\hat{Y}(1)\). Optional targeted regularisation (learnable \(\varepsilon\)). Trains Adam → SGD. | Shi, Blei & Veitch (2019) NeurIPS |
| `tarnet()` | `tarnet` | torch / fallback | **TARNet** — Shared encoder + two separate outcome heads for treated and control units; no representation-balancing penalty. | Shalit, Johansson & Sontag (2017) ICML |
| `cfrnet()` | `cfrnet` | torch / fallback | **CFRNet** — TARNet plus MMD\(^2\) (RBF kernel) balancing penalty on the shared representation \(\Phi(X)\) between treatment groups. | Shalit, Johansson & Sontag (2017); clinicalml/cfrnet |
| `ganite()` | `ganite` | torch / fallback | **GANITE** — Two-stage GAN: generator + discriminator for counterfactual block, then inference network for ITE. Supports AdamW and dropout. | Yoon, Jordon & van der Schaar (2018) ICLR |
| `causalGAN()` | `causalGAN` | **torch** (required) | **CausalGAN** — Structural-equation GAN following \(X \to T \to Y\). Three generators (\(G_X\), \(G_T\), \(G_Y\)) model the interventional distribution; label-smoothed BCE adversarial loss. Predict returns interventional samples and ITE. | — |

### Latent-Variable Causal Models (VAE / Generative)

| Function | Class | torch? | Description | Key reference |
|----------|-------|--------|-------------|---------------|
| `ivae()` | `ivae` | **torch** (required) | **iVAE** — Identifiable VAE conditioned on auxiliary variable \(u\) (e.g. environment or time index). Encoder learns \(q(z\|x,u)\), prior \(p(z\|u) = \prod_k \mathcal{N}(\mu_k(u), \sigma_k(u)^2)\) via segment-conditional flow. Achieves identifiable latent-factor recovery under mild non-stationarity. | Khemakhem et al. (2020) ICML |
| `CausalVAE()` | `nn_module` | **torch** (required) | **CausalVAE module** — Builds the underlying `nn_module` instance with causal DAG constraint, acyclicity penalty \(h(A)\), and sparsity regularisation. Use `causal_vae()` for the high-level training wrapper. | — |
| `CausalVAE_ATE()` | `nn_module` | **torch** (required) | Same as `CausalVAE` with an additional outcome head for ATE estimation. | — |
| `causal_vae()` | `causal_vae` | **torch** (required) | **CausalVAE** — High-level wrapper. VAE with \(d \times d\) learned causal adjacency \(A\), NOTEARS-style DAG acyclicity penalty \(h(A) = \mathrm{tr}(e^{A \circ A}) - d\), and \(L_1\) sparsity. Beta-VAE KL weighting + causal penalty + warmup schedule. | Yang et al. (2021) |
| `causal_vae_opt()` | `causal_vae` | **torch** (required) | **CausalVAE-Opt** — Same architecture as `causal_vae()` with optimised speed-oriented defaults (\(\beta=2\), fewer epochs). | — |
| `generate_data()` | — | **torch** (required) | Generates synthetic latent-causal data \((z_1 \to z_2 \to z_3, x = f(z))\) for CausalVAE experiments. | — |
| `dscm()` | `dscm` | **torch** (required) | **DSCM** — Deep Structural Causal Model with abduction-action-prediction (AAP) counterfactual inference. Treatment mechanism \(p(T\|X)\) + outcome VAE \(q(Z\|X,T,Y)\) + potential-outcome heads \(Y(0), Y(1)\). Supports oracle supervision, KL warmup, and early stopping. | Pearl (2009); Pawlowski et al. (2020) |
| `causal_discrepancy_vae()` / `CausalDiscrepancyVAE()` | `causal_discrepancy_vae` | **torch** (required) | **CausalDiscrepancyVAE** — VAE with separate outcome heads for treated/control + MMD (multi-RBF kernel) balancing penalty on the latent representation \(Z\) between treatment groups. Combined loss: reconstruction + KL + MMD + outcome MSE. | — |
| `causal_egm()` / `CausalEGM()` | `causal_egm` | **torch** (required) | **CausalEGM** — Causal Encoding Generative Model with three disentangled latent factors: \(Z_c\) (confounding), \(Z_t\) (treatment-specific), \(Z_y\) (outcome-specific). Joint optimisation of reconstruction, treatment prediction, outcome regression, and discriminator-based disentanglement. | — |

### Causal Structure Learning (DAG Discovery)

| Function | Class / returns | torch? | Description | Key reference |
|----------|----------------|--------|-------------|---------------|
| `castle()` | `castle` | **torch** (required) | **CASTLE** — CAusal STructure LEarning Regularisation. Learns adjacency \(A\) by optimising prediction + \(L_1\) sparsity + NOTEARS acyclicity \(h(A)\) + neighbourhood reconstruction regularisation. Supports `summary()` and `plot()` via igraph/ggraph. | Kyono, Zhang & van der Schaar (2020) |
| `dagmaLinear()` | `dagma` | pure R (no torch) | **DAGMA-Linear** — Continuous DAG recovery via M-matrix characterisation: score function \(Q(W) = \log \det(s I - W \circ W)\). Supports L2 and logistic loss; optional edge inclusion/exclusion masks. | Bello, Aragam & Ravikumar (2022) NeurIPS |
| `DagmaMLP()` | `nn_module` | **torch** (required) | **DAGMA-MLP module** — Locally-connected MLP for nonlinear structural equations in DAGMA; maps node \(j\) to all-but-\(j\) inputs via \(d\) parallel linear layers. | Bello, Aragam & Ravikumar (2022) NeurIPS |
| `dagma()` | `dagma` | torch for MLP | **DAGMA unified wrapper** — dispatches to `dagmaLinear()` (`method = "linear"`) or `DagmaMLP` + Adam solver (`method = "nonlinear_mlp"`). | Bello, Aragam & Ravikumar (2022) NeurIPS |
| `causalStructureML()` | list | depends on method | **Unified DAG discovery API** — wraps NOTEARS-linear (`notears_linear`), NOTEARS-nonlinear-MLP, NOTEARS-nonlinear-Sobolev, DAG-GNN, and GraN-DAG under one interface. Returns adjacency matrix + diagnostics. | See individual method references |
| `causal_structure_ml_model_descriptions()` | data.frame | — | Returns a data.frame describing all methods available through `causalStructureML()`. | — |

> **NOTEARS** (linear, nonlinear-MLP, nonlinear-Sobolev) live in `R/notears.R`
> (`notears_linear`, `notears_nonlinear`).  
> **DAG-GNN** and **GraN-DAG** are implemented inside `causalDeepNet.R` and exposed through
> `causalStructureML()`.

### Neural Granger-Causality Models (Time Series)

| Function | Class | torch? | Description | Key reference |
|----------|-------|--------|-------------|---------------|
| `neural_granger_ml()` / `neuralGrangerML()` | `neural_granger_ml` | **torch** (required) | **Neural Granger ML** — unified interface for four models on multivariate time-series data. Returns per-model causal matrices, training/validation loss histories, and fitted model objects. | — |

The four sub-models selectable via `models = c("cmlp","clstm","economysru","nri")`:

| Model | Architecture | Sparsity mechanism |
|-------|--------------|--------------------|
| **cMLP** | One MLP per target variable; input is lagged-variable vector | Group-lasso over lag blocks |
| **cLSTM** | One LSTM per target; input is the full lag sequence | Sparse input-weight matrix via group-lasso |
| **EconomySRU** | Structured Recurrent Unit with learnable binary causal mask | Mask-based sparsity loss |
| **NRI** | Neural Relational Inference; latent edge-type probabilities via encoder–decoder GNN | KL penalty on edge-type distributions |

### Quick Usage Examples

```r
library(RCausalML)
set.seed(42)
d <- synthetic_data(mode = 1, n = 500, p = 5, sigma = 1)

# -- Treatment-effect estimators (require torch for neural path) --

# DragonNet
dn <- dragonnet(d$X, d$w, d$y, neurons = 100L, adam_epochs = 20L, sgd_epochs = 30L)
tau_dn <- predict(dn, d$X)

# TARNet
tn <- tarnet(d$X, d$w, d$y, epochs = 50L)
tau_tn <- predict(tn, d$X)

# CFRNet (TARNet + MMD balancing)
cn <- cfrnet(d$X, d$w, d$y, mmd_weight = 0.1, epochs = 50L)
tau_cn <- predict(cn, d$X)

# CEVAE
cv <- cevae(d$X, d$w, d$y, latent_dim = 5L, num_epochs = 20L)
tau_cv <- predict(cv, d$X)

# GANITE
gn <- ganite(d$X, d$w, d$y, h_dim = 32L, iterations = 200L)
tau_gn <- predict(gn, d$X)

# -- Latent-variable models (torch required) --

# CausalVAE
m_cvae <- causal_vae(d$X, latent_dim = 3L, num_epochs = 50L)

# Deep Structural Causal Model
m_dscm <- dscm(d$X, d$w, d$y, hidden_dim = 64L, num_epochs = 50L)
tau_dscm <- predict(m_dscm, d$X, d$w, d$y)$ite

# CausalDiscrepancyVAE
m_cdvae <- causal_discrepancy_vae(d$X, d$w, d$y, latent_dim = 8L, num_epochs = 30L)

# CausalEGM
m_egm <- causal_egm(d$X, d$w, d$y, dim_c = 4L, num_epochs = 50L)

# -- Causal structure learning --

# DAGMA linear (no torch needed)
set.seed(1)
X_dag <- matrix(rnorm(200 * 5), 200, 5)
dag_fit <- dagmaLinear(X_dag, lambda1 = 0.05)
print(dag_fit$adjacency)

# CASTLE (requires torch)
cs_fit <- castle(X_dag, y = X_dag[, 5], hidden_dim = 32L, epochs = 50L)
summary(cs_fit)

# Unified causalStructureML (NOTEARS-linear, no torch)
cs <- causalStructureML(as.data.frame(X_dag), method = "notears_linear")
print(cs$adjacency)

# -- Neural Granger causality --
set.seed(42)
ts_data <- matrix(rnorm(200 * 4), 200, 4)   # 200 time steps, 4 variables
ng <- neural_granger_ml(ts_data, lag = 3L, models = c("cmlp", "clstm"),
                         hidden = 16L, epochs = 20L)
print(ng$causal_matrices$cmlp)
```

---

## Comparison with Python CausalML

- **ATE/CATE**: Same data-generating process (Nie & Wager modes 1–5) where used, so ATE estimates should be in the same ballpark; small differences are expected (different base learners, e.g. XGBoost vs ranger, and randomness).
- **Uplift trees**: R uses the same criteria (KL, ED, Chi, CTS, DDP, IDDP, IT, CIT); implementation details differ (e.g. tree depth, min node size).
- **TMLE**: Not implemented in R package; use the `tmle` R package or Python CausalML for TMLE. Example 06 shows uplift curves and documents the TMLE option.
- **SHAP**: Use R packages `treeshap` or `fastshap` with the fitted ranger models for SHAP values; examples 05 and 18 use ranger variable importance as a stand-in.
- **Python reference**: [CausalML Examples](https://causalml.readthedocs.io/en/latest/examples.html), [GitHub examples](https://github.com/uber/causalml/tree/master/docs/examples).
