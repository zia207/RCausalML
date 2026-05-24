![](images/logo_RCausalML.png){fig-align="left" width="200"}

# RCausalML

**RCausalML** is an R package for uplift modeling and causal inference, inspired by Python’s [CausalML](https://causalml.readthedocs.io/en/latest/about.html) and Microsoft’s [EconML](https://www.microsoft.com/en-us/research/project/econml/). It features automated ML for nuisance models, policy learning, CATE interpretation, DR-learner family, and multiple neural causal models (CEVAE, DragonNet, GANITE, and more) with support for SCMs, DAG learning (NOTEARS, DAG-GNN, GraN-DAG), GNN and transformer-based causal models, and counterfactual analysis. Includes tools for causal structure learning and example notebooks. Most neural modules require the R `torch` package.

Causal inference and machine learning are closely linked areas in data science. Causal inference aims to identify cause-and-effect relationships, while machine learning seeks patterns and makes predictions, often based on associations rather than direct causes. To uncover real causal effects, causal inference uses methods like randomized controlled trials, instrumental variables, and propensity score matching, especially when running experiments is not possible or ethical. Machine learning is strong at working with large and complex datasets for tasks like classification, prediction, and pattern recognition, but it does not automatically tell us what causes what.

Lately, these two fields have come together as “causal machine learning,” which brings out the best of both. Machine learning can help causal inference by automatically adjusting for confounders and estimating treatment effects in complex data. At the same time, causal inference can make machine learning models easier to understand and more reliable. This combination is especially useful in areas such as healthcare, economics, and technology, where it is important to understand both what might happen and why. By working together, these methods help create AI systems that not only predict outcomes but also explain the reasons behind them.

## Overview

The package estimates the **Conditional Average Treatment Effect (CATE)** and related quantities from experimental or observational data using meta-learners, propensity scoring, matching, uplift trees, policy learning, and instrumental variables.

### EconML-aligned modules

Three core modules align with [EconML](https://www.microsoft.com/en-us/research/project/econml/)’s API and concepts:

-   [**R/DMLearner.R**](R/DMLearner.R) — **Double Machine Learning (DML)**: Chernozhukov et al.–style DML with cross-fitting; outcome and treatment are fitted with flexible ML (ranger, glmnet, xgb, lm), then a final stage (linear, sparse linear, kernel, nonparametric, or causal forest) estimates CATE from residuals. Includes **LinearDML**, **SparseLinearDML**, **KernelDML**, **NonParamDML**, **CausalForestDML**, plus native R **DynamicDMLearner** (panel/sequential treatments) and **OrthoIVLearner** / **DMLIVLearner** / **NonParamDMLIVLearner** for DML with instrumental variables. `predict.DMLearner` is robust to near-degenerate OLS final stages (aliased `lm` coefficients replaced with zero) and guarantees finite output for all callers including `explain_cate()` / `kernelshap`. When **DoubleML**, **mlr3**, **mlr3learners**, and **mlr3measures** are installed, **DiD-style DML** on $\Delta y = y_1 - y_0$ is available via **`doubleml_did_linear`**, **`doubleml_did_rf`**, **`doubleml_did_xgboost`**, with optional nuisance metrics from **`doubleml_did_eval_*`** and **`doubleml_did_eval_preds`**; **`doubleml_plr`**, **`doubleml_plr_fit_data`** (pre-built **DoubleMLData** + any mlr3 learners, including pipelines / ensembles), **`doubleml_plr_tune_data`** (with **mlr3tuning**), and **`doubleml_pliv`** fit partially linear regression and PLIV through the DoubleML R package; **`doubleml_data_from_data_frame`** / **`doubleml_data_from_matrix`** build **DoubleMLData**; **`doubleml_make_plr_CCDDHNR2018`**, **`doubleml_make_pliv_CHS2015`**, **`doubleml_fetch_401k`**, and **`doubleml_fetch_bonus`** provide example / simulated data aligned with DoubleML.

-   [**R/cate_interpreter.R**](R/cate_interpreter.R) — **CATE interpreters**: Single-tree interpretation of any CATE or policy model. **SingleTreeCateInterpreter** fits a regression tree to predicted CATE to summarize effect heterogeneity by covariates; **SingleTreePolicyInterpreter** fits a classification tree that predicts optimal treatment (who to treat) from covariates. API matches EconML’s `econml.cate_interpreter`.

-   [**R/policy_learner.R**](R/policy_learner.R) — **Policy learning**: Learns a treatment assignment policy to maximize expected outcome (welfare). **policy_learner** implements Athey & Wager (2018) doubly robust weighted classification; **DRPolicyTree** and **DRPolicyForest** provide cross-fitted outcome + policy tree/forest (EconML-style). Supports `fit`, `predict`, `predict_proba`, and (for tree/forest) `feature_importances` and `predict_value`.

-   [**R/automated_ml.R**](R/automated_ml.R) — **Automated ML**: Automate model selection for nuisance models (Y, T) in causal estimators. **EconAutoMLConfig** configures constraints (linear-only, sample-weights required) and task (regression/classification). **AutomatedMLModel** selects a learner via cross-validation from the whitelist (lm, glmnet, ranger, rpart), then fits and exposes `fit`, `predict`, and `predict_proba`. Use with **add_automated_ml** to pass configs into meta-learners. Local AutoML (no Azure); API aligned with EconML’s `econml.automated_ml`.

## Supported algorithms

| Category | Algorithms | R functions |
|----|----|----|
| **Tree-based** | Uplift RF (KL, ED, Chi) | `uplift_rf_kl`, `uplift_rf_ed`, `uplift_rf_chi` |
|  | Uplift RF on Contextual Treatment Selection | `uplift_rf_cts` |
|  | Unified uplift random forest (classification & regression) | `uplift_randomForest`, `uplift_randomForestClassifier`, `uplift_randomForestRegressor` ([R/uplift_randomForest.R](R/uplift_randomForest.R)) |
|  | Multi-treatment uplift RF | `uplift_rf_multi` |
|  | Uplift tree on ΔΔP (binary outcome) | `uplift_tree_ddp` |
|  | Uplift tree on IDDP (binary outcome) | `uplift_tree_iddp` |
|  | Interaction Tree, Causal Inference Tree | `interaction_tree`, `causal_inference_tree` |
| **Meta-learners** | S-, T-, X-, R-, DR-learner | `SLearner`, `TLearner`, `XLearner`, `RLearner`, `DRLearner`, `fit()`, `estimate_ate()` |
| **Double ML (DML)** | LinearDML, SparseLinearDML, KernelDML, NonParamDML, CausalForestDML; Dynamic DML (panel); DML with IV; optional DoubleML DiD / PLR / PLIV / data | `DMLearner`, `LinearDML`, `SparseLinearDML`, `KernelDML`, `NonParamDML`, `CausalForestDML`; `DynamicDMLearner` (panel, native R); `OrthoIVLearner`, `DMLIVLearner`, `NonParamDMLIVLearner` (IV, native R); `doubleml_plr`, `doubleml_plr_fit_data`, `doubleml_plr_tune_data`, `doubleml_pliv`, `doubleml_data_from_data_frame`, `doubleml_data_from_matrix`, `doubleml_make_plr_CCDDHNR2018`, `doubleml_make_pliv_CHS2015`, `doubleml_fetch_401k`, `doubleml_fetch_bonus`, `doubleml_did_linear` / `doubleml_did_rf` / `doubleml_did_xgboost`, `doubleml_did_eval_*`, `doubleml_did_eval_preds` (optional: DoubleML, mlr3, mlr3learners, mlr3tuning, paradox, mlr3measures). |
| **Policy learning** | Policy Learner (Athey & Wager, DR + weighted classifier) | `policy_learner`, `fit`, `predict`, `predict_proba` |
|  | DR Policy Tree (EconML-style: cross-fitted outcome + tree) | `DRPolicyTree`, `fit`, `predict`, `predict_proba`, `predict_value`, `feature_importances` |
|  | DR Policy Forest (ensemble of policy trees) | `DRPolicyForest`, `fit`, `predict`, `predict_proba`, `feature_importances` |
| **CATE interpretation** | Single-tree CATE interpreter (tree over predicted CATE) | `SingleTreeCateInterpreter`, `interpret`, `predict` |
|  | Single-tree policy interpreter (tree over optimal treatment) | `SingleTreePolicyInterpreter`, `interpret`, `treat`, `predict` |
| **Instrumental variables** | 2SLS, LATE, DRIV; OrthoIV, DMLIV, NonParamDMLIV (native R) | `iv_2sls`, `late_iv`, `driv_learner`; `OrthoIVLearner`, `DMLIVLearner`, `NonParamDMLIVLearner` (native R) |
| **Neural** | CEVAE, DragonNet, TARNet, CFRNet, GANITE, CausalGAN, DSCM, CausalDiscrepancyVAE, and neural Granger-causality models (cMLP, cLSTM, EconomySRU, NRI) in [R/causalDeepNet.R](R/causalDeepNet.R); tabular DCEVAE reference under `inst/dcevae/` (optional **reticulate** + Python **torch**) | `cevae`, `dragonnet`, `tarnet`, `cfrnet`, `ganite`, `causalGAN`, `dscm`, `causal_discrepancy_vae`, `CausalDiscrepancyVAE`, `neural_granger_ml`, `neuralGrangerML` (R `torch` when available) |
| **SCM with Deep Components** | DeepSCM (fixed-graph SCM, variational noise encoders, ELBO training, do-calculus interventions), DECI (jointly learns causal graph + structural equations, NOTEARS acyclicity penalty, Monte-Carlo ATE), DynoTEARS (lagged causal discovery, augmented-Lagrangian DAG constraint); graph-recovery utilities in [R/causalDeepNet.R](R/causalDeepNet.R) | `deep_scm`, `predict.deep_scm`, `intervene_deep_scm`, `deci_model`, `predict.deci_model`, `ate_deci`, `dynotears`, `predict.dynotears`, `evaluate_graph_recovery`, `plot_scm_dag`; aliases `deepSCM`, `deciModel`, `dynoTEARS` (requires `torch`) |
| **Attention/Transformer Causal Models** | TCDF (Temporal Causal Discovery Framework, Nauta et al. 2019 — causal dilated convolutions + variable-importance attention), CausalTransformer (autoregressive masking + inter-variable cross-attention for graph discovery), TFT (Temporal Fusion Transformer, Lim et al. 2021 — variable-selection networks, LSTM encoder, interpretable multi-head temporal attention); all in [R/causalDeepNet.R](R/causalDeepNet.R) | `attn_causal_model`, `predict.attn_causal_model`, `causal_matrix_attn`, `tcdf_model`, `causal_transformer_model`, `tft_model`; aliases `attnCausalModel`, `TCDFModel`, `CausalTransformerModel`, `TFTModel` (requires `torch`) |
| **RNN/LSTM Causal Models** | CausalLSTM (per-variable LSTM with learnable sparse adjacency mask, L1 sparsity penalty), RETAIN (Choi et al. 2016 — reverse-time dual-channel GRU: temporal α + variable β attention, interpretable attribution matrix), Intervention-Aware RNN (GRU regime detector + learned regime/intervention embeddings, regime-conditioned causal matrices); all in [R/causalDeepNet.R](R/causalDeepNet.R) | `rnn_causal_model`, `predict.rnn_causal_model`, `causal_matrix_rnn`, `causal_lstm_model`, `retain_model`, `intervention_rnn_model`; aliases `rnnCausalModel`, `CausalLSTMModel`, `RETAINModel`, `InterventionRNNModel` (requires `torch`) |
| **GNN Causal Models** | GVAR (Graph Vector Autoregression — lag-specific soft adjacency matrices with L1 sparsity + NOTEARS DAG penalties, two stacked GNN message-passing layers), CausalGNN / CD-GNN (GRU per-variable temporal encoder + bilinear-style graph learner + stacked edge-conditioned GNN layers with single-step GRU node update + LayerNorm, NOTEARS DAG + sparsity penalties), CUTS+ (variational Bernoulli graph posterior + joint imputation network for missing data + GRU encoder + edge-conv, MSE + KL-to-sparse-prior + NOTEARS DAG); all in [R/causalForest.R](R/causalForest.R) | `gnn_causal_model`, `predict.gnn_causal_model`, `causal_matrix_gnn`, `gvar_model`, `causal_gnn_model`, `cuts_model`; aliases `gnnCausalModel`, `GNNCausalModel`, `GVARModel`, `CausalGNNModel`, `CUTSModel` (requires `torch`) |
| **Counterfactual / Potential Outcomes Models** | DeepSynth (Neural Synthetic Control — GRU encoder, scaled dot-product attention over donor variables forms soft synthetic control; factual + counterfactual heads yield ITE and ATE), CRN (Counterfactual Recurrent Network — GRU encoder with adversarial treatment-balancing discriminator, decoder conditioned on representation and do(T) gives Y-hat(0)/Y-hat(1)), G-Net (Deep G-Computation — GRU backbone with covariate transition head and outcome head; counterfactual outcomes via sequential substitution of intervened treatment); all in [R/causalDeepNet.R](R/causalDeepNet.R) | `counterfactual_model`, `predict.counterfactual_model`, `ate_counterfactual`, `ite_counterfactual`, `deep_synth_model`, `crn_model`, `gnet_model`; aliases `CounterfactualModel`, `DeepSynthModel`, `CRNModel`, `GNetModel` (requires `torch`) |
| **Causal XGBoost** | Two-head DragonNet-style masked-MSE outcome model + ranger propensity; PEHE / ATE evaluation; save/load ([R/causalXGBoost.R](R/causalXGBoost.R)) | `CXGBoost` (R6), `PEHE`, `ATE`, `ATE_error`, `run_example` |
| **Multi-arm Causal Boosting** | K \>= 2 treatment arms: per-arm XGBoost outcomes + multiclass ranger propensity; univariate and multivariate Y ([R/multi_arm_causal_boost.R](R/multi_arm_causal_boost.R)) | `MultiArmCausalBoost` (R6), `multi_arm_PEHE`, `multi_arm_ATE`, `load_multi_arm_causal_boost`, `save_multi_arm_causal_boost`, `run_multi_arm_causal_boost_example` |
| **Interventional CRL** | Environment-conditioned VAE with analytical KL for binary fingerprints; interventional ELBO loss ([R/interventionalCRL.R](R/interventionalCRL.R)) | `InterventionalCRL` (torch nn_module), `interventional_elbo_loss` (requires `torch`) |
| **Temporal Causal VAE** | GRU encoder/decoder with learned causal adjacency, NOTEARS DAG penalty, sparsity regularisation ([R/tempoCausalVAE.R](R/tempoCausalVAE.R)) | `TemporalCausalVAE` (torch nn_module), `temporal_causal_loss` (requires `torch`) |
| **Temporal Causal Discovery (TCDF)** | Depthwise-TCN ADDSTCN for attention-based multivariate time-series causal discovery (Nauta et al.) ([R/temporalCausaDiscovery.R](R/temporalCausaDiscovery.R)) | `TCDF_ADDSTCN`, `TCDF_depthwise_net` and building blocks (requires `torch`) |
| **Treatment optimization** | Counterfactual Value Estimator, Unit Selection | `counterfactual_value_estimator`, `counterfactual_unit_selection`, `predict_best_treatment` |
| **Data & propensity** | Synthetic data, propensity, matching; textbook data (causaldata) | `synthetic_data`, `propensity_glmnet`/`propensity_glm`, `nearest_neighbor_match`, `create_table_one`; `load_causaldata`, `list_causaldata_datasets` (optional: install `causaldata`) |
| **Automated ML** | Auto-select nuisance models (Y, T) with constraints | `EconAutoMLConfig`, `AutomatedMLModel`, `set_automated_ml_workspace`, `add_automated_ml` (EconML-style, local) |
| **Causal structure learning** | NOTEARS (linear, nonlinear), DAG-GNN, GraN-DAG; unified API | [R/causalDeepNet.R](R/causalDeepNet.R): `notears_linear`, `notears_nonlinear`, `simulate_dag`, `count_accuracy`, `make_daggnn`, `daggnn_adj`, `GraNDAG`, `GraphDAG`; [R/causalStructureML.R](R/causalStructureML.R): `causalStructureML`, `causal_structure_ml_model_descriptions` |

## Installation

For a step-by-step install and troubleshooting guide (including **Ubuntu** and **Windows** instructions), see [HELP_RCausalML_0.3.0.md](HELP_RCausalML_0.3.0.md).

### From GitHub (easiest)

Install directly from [GitHub](https://github.com/zia207/RCausalML) using `remotes` or `devtools`:

``` r
# Using remotes (recommended)
install.packages("remotes")
remotes::install_github("zia207/RCausalML")

# Or using devtools
install.packages("devtools")
devtools::install_github("zia207/RCausalML")
```

To install a specific branch or tag:

``` r
# Install from a specific branch
remotes::install_github("zia207/RCausalML", ref = "main")

# Install with all suggested dependencies
remotes::install_github("zia207/RCausalML", dependencies = TRUE)
```

### Global (recommended): install missing dependencies + install RCausalML

Run this **from the package root** (the directory that contains `DESCRIPTION`). It will **install any missing R package dependencies** declared in `DESCRIPTION` (Imports/Depends/Suggests), then install `RCausalML`.

``` r
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

# remove version constraints like "(>= 1.2)" and split
deps_raw <- gsub("\\s*\\([^\\)]+\\)", "", deps_raw)
deps <- trimws(unlist(strsplit(deps_raw, ",")))
deps <- setdiff(deps, c("R"))

install_missing(c("remotes"))  # used below (lightweight)
install_missing(deps)

remotes::install_local(".", dependencies = FALSE, upgrade = "never")
```

### From source (manual)

After cloning or unpacking:

``` r
# From package root (directory containing DESCRIPTION):
install.packages(".", repos = NULL, type = "source")
# or:
# devtools::install(".")
```

### From built tarball (`RCausalML_0.3.0.tar.gz`)

Build the archive from the package root (writes `RCausalML_0.3.0.tar.gz` next to `DESCRIPTION`):

``` bash
cd RCausalML   # directory containing DESCRIPTION
R CMD build .
```

**Ubuntu:**

``` bash
R CMD INSTALL RCausalML_0.3.0.tar.gz
# or to a user library:
R CMD INSTALL -l ~/.Rlibrary RCausalML_0.3.0.tar.gz
```

**Windows (PowerShell / R console):**

``` r
# Rtools must be installed and on PATH
install.packages("RCausalML_0.3.0.tar.gz", repos = NULL, type = "source")
```

**Package name:** `RCausalML` (version 0.2.0). Load with `library(RCausalML)`.

**Dependencies:** `glmnet`, `ranger`, `rpart`, `R6`, `expm`, `igraph`, `htetree`. Optional: `grf`, `MatchIt`, `xgboost`, `reticulate`, `devtools` (for `load_all` in standalone test scripts). For **DoubleML** helpers in [R/DMLearner.R](R/DMLearner.R) (PLR, PLIV, DiD, `DoubleMLData` wrappers, simulators / fetch): install `DoubleML`, `mlr3`, `mlr3learners`, and `mlr3measures`. For **NOTEARS** ([R/causalDeepNet.R](R/causalDeepNet.R)): `expm`, `igraph`; `torch` for nonlinear NOTEARS (MLP/Sobolev). For **DAG-GNN** and **GraN-DAG** ([R/causalDeepNet.R](R/causalDeepNet.R)): `torch`, `R6`, `expm`; `ranger` and `progress` optional for GraN-DAG PNS and progress bars. For **causalStructureML** ([R/causalStructureML.R](R/causalStructureML.R)): same as NOTEARS / DAG-GNN / GraN-DAG depending on `method`. For **SHAP / shapviz** (see below): `kernelshap`, `shapviz`. For **causalDeepNet** ([R/causalDeepNet.R](R/causalDeepNet.R)): `torch (>= 0.10)` (CEVAE, DragonNet, TARNet, CFRNet, GANITE, CausalGAN, DSCM, CausalDiscrepancyVAE, neural Granger models `cMLP`, `cLSTM`, `EconomySRU`, `NRI` via `neural_granger_ml()`, and SCM Deep Components: `deep_scm`, `deci_model`, `dynotears`); `nnet` / `ranger` placeholders where applicable. For **SCM graph utilities** (`plot_scm_dag`): `reshape2` (optional, falls back to base graphics). For **tabular DCEVAE** (optional, not exported in this build): R package `reticulate` and a Python environment with `torch`, `numpy`, and `tqdm` (code under `inst/dcevae/DCEVAE_ours/`).

### Linux system libraries (only if installs fail compiling packages)

Some dependencies (e.g. `glmnet`, `torch`) may need compilers and headers. On Ubuntu/Debian, this usually fixes build errors:

``` bash
sudo apt-get update
sudo apt-get install -y build-essential gfortran cmake git \
  libcurl4-openssl-dev libssl-dev libxml2-dev
```

## SHAP / shapviz integration

You can apply **all shapviz functions** (importance, dependence, waterfall, force plots, etc.) to **any RCausalML CATE model** by computing SHAP values with `kernelshap` and then passing the result to `shapviz()`.

1.  Install optional packages: `install.packages(c("kernelshap", "shapviz"))`.
2.  Use **`explain_cate(model, X, ...)`** from RCausalML to get a `kernelshap` object for your fitted model and feature matrix `X`.
3.  Pass that object to **`shapviz()`** to build a `shapviz` object.
4.  Use any shapviz function: `sv_importance()`, `sv_dependence()`, `sv_waterfall()`, `sv_force()`, `sv_dependence2D()`, etc.

Works with meta-learners (S/T/X/R/DR), DML (Linear, SparseLinear, Kernel, NonParam, CausalForest, Dynamic, IV), causal forest, uplift trees/forests, policy learners, CATE interpreters, neural models (CEVAE, DragonNet, TARNet, CFRNet, GANITE, CausalGAN, DSCM, CausalDiscrepancyVAE, neural Granger models through `neural_granger_ml()`, DeepSCM / DECI / DynoTEARS via `deep_scm()` / `deci_model()` / `dynotears()`), and other fitters that have `predict(object, newdata)`. An optional tabular **DCEVAE** bridge (under `inst/`) uses a specialized `predict(fit, r, d, a, y)` signature rather than `newdata` alone when that code path is wired in.

## causaldata integration

RCausalML integrates with the [**causaldata**](https://github.com/NickCH-K/causaldata) R package (Huntington-Klein & Barrett), which provides example datasets from causal inference textbooks (*The Effect*, *Causal Inference: The Mixtape*, *Causal Inference: What If*). Use **`load_causaldata(name)`** to load a dataset in RCausalML-ready form (covariates `X`, treatment `w`, outcome `y`, and optionally instrument `Z`), and **`list_causaldata_datasets()`** to see supported names and variable roles.

``` r
library(RCausalML)
list_causaldata_datasets()   # nsw_mixtape, cps_mixtape, abortion, close_college, ...
d <- load_causaldata("nsw_mixtape")
sl <- SLearner(learner = "ranger")
sl <- fit(sl, d$X, d$w, d$y)
estimate_ate(sl, d$X, d$w, d$y)
# Always cite the data source (returned in d$citation) and the causaldata package:
citation("causaldata")
```

Install the data package: `install.packages("causaldata")`. When publishing or reporting results that use these datasets, cite **causaldata** and the original source given in the `citation` component of the list returned by `load_causaldata()`.

``` r
library(RCausalML)
library(kernelshap)
library(shapviz)

d <- synthetic_data(mode = 1, n = 500, p = 5, sigma = 1)
sl <- SLearner(learner = "ranger")
sl <- fit(sl, d$X, d$w, d$y)

# SHAP for CATE; then all shapviz plots
ks <- explain_cate(sl, d$X[1:100, ], bg_X = d$X, use_permshap = TRUE)
shp <- shapviz(ks)
sv_importance(shp)
sv_dependence(shp, "X1")
sv_waterfall(shp, 1)
sv_force(shp, 1)
```

See `?explain_cate` for arguments (`n_samples`, `bg_X`, `use_permshap`, and options passed to `kernelshap`/`permshap`).

## Performance

To run faster on multi-core machines (Unix/macOS), use the `n_cores` argument where supported:

-   **Meta-learners** (`fit_predict`, `estimate_ate` with `return_ci=TRUE` or `bootstrap_ci=TRUE`): bootstrap runs in parallel, e.g. `fit_predict(sl, X, w, y, return_ci = TRUE, n_cores = 4)` or `estimate_ate(sl, X, w, y, bootstrap_ci = TRUE, n_cores = 4)`.
-   **DR-Learner**: same `n_cores` for bootstrap in `fit_predict.DRLearner` and `estimate_ate.DRLearner`.
-   **Uplift forests** (`uplift_rf_kl`, `uplift_rf_ed`, `uplift_rf_chi`, `uplift_rf_cts`, `uplift_randomForest`): tree building runs in parallel, e.g. `uplift_randomForest(X, w, y, n_estimators = 100, n_cores = 4)`.

Parallelization uses `parallel::mclapply` (multi-core lapply). On Windows, `n_cores > 1` is ignored and runs remain sequential.

## Quick start

Run the quickstart example (see `inst/examples/quickstart.R`):

``` r
library(RCausalML)

# Synthetic data (Nie & Wager 2018 style)
set.seed(42)
d <- synthetic_data(mode = 1, n = 1000, p = 5, sigma = 1)
y <- d$y
X <- d$X
w <- d$w          # treatment (0/1)
e <- d$e          # propensity (for X-Learner, R-Learner, TMLE)

# S-Learner (LRSRegressor = SLearner(learner = "lm"))
lr <- LRSRegressor()
lr <- fit(lr, X, w, y)
ate_lr <- estimate_ate(lr, X, w, y, return_ci = TRUE)

# S-Learner with ranger
sl <- SLearner(learner = "ranger")
sl <- fit(sl, X, w, y)
ate_sl <- estimate_ate(sl, X, w, y, return_ci = TRUE)

# T-Learner
tl <- TLearner(learner = "ranger")
tl <- fit(tl, X, w, y)
ate_tl <- estimate_ate(tl, X, w, y, return_ci = TRUE)

# X-Learner (with propensity)
xl <- XLearner(learner = "ranger")
xl <- fit(xl, X, w, y, p = e)
ate_xl <- estimate_ate(xl, X, w, y, p = e, return_ci = TRUE)

# R-Learner
rl <- RLearner(learner = "ranger", n_fold = 5)
rl <- fit(rl, X, w, y, p = e)
ate_rl <- estimate_ate(rl, X, w, y, p = e, return_ci = TRUE)

# TMLE
tmle <- TMLELearner(learner = "ranger")
tmle <- fit(tmle, X, w, y, p = e)
ate_tmle <- estimate_ate(tmle, X, w, y, p = e, return_ci = TRUE)

# Propensity and CATE
ps <- propensity_glmnet(X, w, n_fold = 5)
cate_s <- predict(lr, X, verbose = FALSE)
cate_t <- predict(tl, X)
```

**Example output:**

```         
ATE (S-Learner / LRSRegressor): 0.60 (0.46, 0.75)
ATE (S-Learner ranger): 0.48 (0.41, 0.55)
ATE (T-Learner ranger): 0.56 (0.49, 0.64)
ATE (X-Learner ranger): 0.49 (0.42, 0.55)
ATE (R-Learner ranger): 0.52 (0.37, 0.68)
ATE (TMLE ranger): 0.44 (0.34, 0.54)
Propensity scores (first 5): 0.8517243 0.5155485 0.6182359 0.7123617 0.6122691
Mean CATE (S-Learner): 0.6048919   Mean CATE (T-Learner): 0.5633381
```

### Uplift forest and IV

``` r
# Unified uplift random forest (auto-detects classification vs regression)
uf <- uplift_randomForest(X, w, y, n_estimators = 100, evaluation_function = "KL")
cate_u <- predict(uf, X)

# Or explicitly for binary / continuous outcomes:
# uplift_randomForestClassifier(X, w, y, n_estimators = 100)
# uplift_randomForestRegressor(X, w, y, n_estimators = 100, evaluation_function = "TLearner")

# Legacy entry points (same underlying tree builders):
# uf <- uplift_rf_kl(X, w, y, n_trees = 100)
# cate_u <- predict(uf, X)

# 2SLS with instrument Z
# iv_fit <- iv_2sls(Y, W, Z, X)
# LATE: late_iv(Y, W, Z)
```

### Treatment optimization

``` r
# Policy learner (Athey & Wager: who to treat) — doubly robust + classifier
d <- synthetic_data(mode = 1, n = 800, p = 6, sigma = 1)
pl <- policy_learner(outcome_learner = "ranger", treatment_learner = "glmnet", policy_learner = "rpart")
pl <- fit(pl, d$X, d$w, d$y, control = rpart::rpart.control(maxdepth = 2))
policy_pred <- predict(pl, d$X)   # 0 = control, 1 = treat
prob_treat <- predict_proba(pl, d$X)

# DR Policy Tree (EconML-style): cross-fitted outcome + single policy tree
pt <- DRPolicyTree(model_regression = "ranger", cv = 2L, random_state = 42)
pt <- fit(pt, d$X, d$w, d$y)
pred_pt <- predict(pt, d$X)
proba_pt <- predict_proba(pt, d$X)
imp_pt <- feature_importances(pt)

# DR Policy Forest: ensemble of policy trees (majority vote)
pf <- DRPolicyForest(n_estimators = 50L, cv = 2L, max_samples = 0.6, random_state = 123)
pf <- fit(pf, d$X, d$w, d$y)
pred_pf <- predict(pf, d$X)

# CATE interpreter: single tree over CATE (interpret heterogeneity)
sl <- SLearner(learner = "ranger"); sl <- fit(sl, d$X, d$w, d$y)
interp_cate <- SingleTreeCateInterpreter(max_depth = 3L, min_samples_leaf = 10L)
interp_cate <- interpret(interp_cate, sl, d$X)
cate_tree <- predict(interp_cate, d$X)   # tree-predicted CATE

# Policy interpreter: single tree over optimal treatment (who to treat)
pol_interp <- SingleTreePolicyInterpreter(max_depth = 3L, min_samples_leaf = 10L)
pol_interp <- interpret(pol_interp, sl, d$X)
trt_rec <- treat(pol_interp, d$X)         # recommended treatment (0/1)
```

### Automated ML (EconML-style) {#automated-ml-econml-style}

Use **EconAutoMLConfig** and **AutomatedMLModel** to auto-select nuisance models (e.g. for outcome or treatment) with constraints (linear-only, sample-weights). Assign the result of `fit()` so the fitted model is stored.

``` r
# AutoML config: only models that support sample weights (e.g. for DR)
cfg <- EconAutoMLConfig(sample_weights_required = TRUE, task = "regression", show_output = TRUE)
m <- AutomatedMLModel(cfg)
m <- fit(m, X, y)           # CV selects best of lm, glmnet, ranger, rpart; refit on full data
pred <- predict(m, X)

# Classification (propensity): fit and predict_proba
cfg_cls <- EconAutoMLConfig(task = "classification", show_output = FALSE, n_folds = 3)
m_cls <- AutomatedMLModel(cfg_cls)
m_cls <- fit(m_cls, X, treatment)
prob <- predict_proba(m_cls, X)

# Replace configs with models in a list (e.g. for passing to meta-learners)
args <- add_automated_ml(list(learner = cfg), workspace = NULL)
```

``` r
# Counterfactual value estimator
# cve <- counterfactual_value_estimator(treatment, control_name, treatment_names,
#   y_proba = matrix(...), value = 1, conversion_cost = 0, impression_cost = 0)
# best_idx <- predict_best_treatment(cve)
```

## Testing Algorithms

From the package root you can run the neural-network test (`tests/test-neural.R` for [R/causalDeepNet.R](R/causalDeepNet.R)), the DR-Learner test (`tests/test-DRLearner.R`), the **DMLearner test** (`tests/test-DMLearner.R`), the policy learner test (`tests/test-policy-learner.R`), the **CATE interpreter test** (`tests/test-cate-interpreter.R`), the **NOTEARS test** (`tests/test-notears.R`), the **DAG-GNN test** (`tests/test-dag_gnn.R`), the **GraN-DAG test** (`tests/test-GraN_DAG.R`), the **causalStructureML test** (`tests/test-causalStructureML.R` for [R/causalStructureML.R](R/causalStructureML.R)), the **automated ML test** (see [Automated ML](#automated-ml-econml-style) for inline checks), the uplift trees test (`tests/test-uplift-trees.R`), the **uplift random forest test** (`tests/test-uplift_randomForest.R`), and the causal structure learning test (`tests/test-causal_structure_learning.R`) with synthetic data.

**Table of contents**

| Test | Description |
|----|----|
| [Dataset (synthetic)](#dataset-synthetic) | Synthetic DGP for tests (X, W, propensity, true CATE) |
| [Run uplift trees test](#run-uplift-trees-test) | Uplift trees and forests (DDP, IDDP, IT, CIT, KL, ED, CTS) |
| [Run DR-Learner test](#run-dr-learner-test) | DR-learner variants (Linear, SparseLinear, XGB, Forest) |
| [Run DMLearner test](#run-dmlearner-test) | Double ML (LinearDML, KernelDML, NonParamDML, CausalForestDML, Dynamic DML, IV) |
| [Run neural test](#run-neural-test) | Neural causal models in [R/causalDeepNet.R](R/causalDeepNet.R) (CEVAE, DragonNet, TARNet, CFRNet, GANITE, CausalGAN, DSCM, CausalDiscrepancyVAE, neural Granger models cMLP/cLSTM/EconomySRU/NRI, and SCM Deep Components: DeepSCM, DECI, DynoTEARS; optional DCEVAE via `reticulate` when sourced from tests) |
| [Run SCM deep test](#run-scm-deep-test) | SCM Deep Components: `deep_scm`, `deci_model`, `dynotears`, `evaluate_graph_recovery`, `plot_scm_dag`, intervention, ATE, graph recovery metrics ([tests/test-scm-deep.R](tests/test-scm-deep.R)) |
| [Run attention/transformer test](#run-attention-transformer-test) | Attention-Based/Transformer Causal Models: `attn_causal_model`, `tcdf_model`, `causal_transformer_model`, `tft_model`, `causal_matrix_attn`, `predict.attn_causal_model` ([R/causalDeepNet.R](R/causalDeepNet.R)) |
| [Run GNN causal model test](#run-gnn-causal-model-test) | GNN Causal Models: `gnn_causal_model`, `gvar_model`, `causal_gnn_model`, `cuts_model`, `causal_matrix_gnn`, `predict.gnn_causal_model` ([R/causalForest.R](R/causalForest.R)) |
| [Run RNN/LSTM causal model test](#run-rnnlstm-causal-model-test) | RNN/LSTM Causal Models: `rnn_causal_model`, `causal_lstm_model`, `retain_model`, `intervention_rnn_model`, `causal_matrix_rnn`, `predict.rnn_causal_model` ([R/causalDeepNet.R](R/causalDeepNet.R)) |
| [Run policy learner test](#run-policy-learner-test) | Policy learner, DR Policy Tree, DR Policy Forest |
| [Run CATE interpreter test](#run-cate-interpreter-test) | SingleTreeCateInterpreter, SingleTreePolicyInterpreter |
| [Run NOTEARS test](#run-notears-test) | NOTEARS in [R/causalDeepNet.R](R/causalDeepNet.R): utilities, linear, nonlinear (MLP), demo_linear |
| [Run DAG-GNN test](#run-dag-gnn-test) | DAG-GNN ([R/causalDeepNet.R](R/causalDeepNet.R)): device, forward, ELBO, make_daggnn, daggnn_adj |
| [Run GraN-DAG test](#run-gran-dag-test) | GraN-DAG ([R/causalDeepNet.R](R/causalDeepNet.R)): is_acyclic, constraint, NormalizationData, GraNDAG learn |
| [Run causalStructureML test](#run-causalstructureml-test) | Unified API ([R/causalStructureML.R](R/causalStructureML.R)): all `method` values, `causal_structure_ml_model_descriptions` |
| [Run causal structure learning test](#run-causal-structure-learning-test) | NOTEARS, DAG-GNN, GraN-DAG, optional Optuna tune |

### Dataset (synthetic) {#dataset-synthetic}

``` r
set.seed(123)
n <- 1000
p_x <- 5
X <- matrix(rnorm(n * p_x), n, p_x)
colnames(X) <- paste0("X", 1:p_x)
W <- matrix(rnorm(n * 5), n, 5)

# Propensity: Pr[T=1 | X, W]
propensity <- 1 / (1 + exp(-X[, 1] - 0.5 * W[, 1]))
T_bin <- rbinom(n, 1, propensity)

# True CATE: non-linear in X1
true_cate <- 1 + 0.5 * X[, 1] + 0.3 * (X[, 1]^2)
Y <- as.vector(2 + 0.3 * (W[, 1:3] %*% c(1, -0.5, 0.5)) + true_cate * T_bin + rnorm(n, 0, 0.5))

x_cols <- paste0("X", 1:p_x)
data <- data.frame(
  Y = Y, T = factor(T_bin), T_num = T_bin,
  as.data.frame(X), W1 = W[, 1], W2 = W[, 2], W3 = W[, 3], true_cate = true_cate
)
X_mat <- as.matrix(data[, x_cols])
treatment <- as.integer(data$T_num)
y <- as.numeric(data$Y)
```

### Run uplift trees test {#run-uplift-trees-test}

[`tests/test-uplift-trees.R`](tests/test-uplift-trees.R) runs **uplift trees and forests** (`R/uplift_trees.R`) on synthetic binary-outcome data from `make_uplift_classification()`: single trees (DDP, IDDP, IT, CIT), uplift RF (KL, ED, CTS), and multi-treatment `uplift_rf_multi()`.

``` r
# Minimal example (from test): synthetic data + single tree + forest
library(RCausalML)
set.seed(42)
out <- make_uplift_classification(
  treatment_name = c("control", "treatment1"),
  n_samples = 1200, n_classification_features = 8,
  n_uplift_increase_dict = list(treatment1 = 3),
  n_uplift_decrease_dict = list(treatment1 = 2),
  delta_uplift_increase_dict = list(treatment1 = 0.12),
  delta_uplift_decrease_dict = list(treatment1 = -0.08),
  random_seed = 42
)
df <- out$data
X <- as.matrix(df[, out$X_names])
w <- as.integer(df$treatment_group_key != "control")
y <- df$conversion
# Single trees
tree_ddp <- uplift_tree_ddp(X, w, y, min_node_size = 30, max_depth = 4)
tau_ddp <- predict(tree_ddp, X)
# Uplift RF with full_output
rf_kl <- uplift_rf_kl(X, w, y, n_trees = 5, min_node_size = 30, max_depth = 4, random_state = 123)
pred_full <- predict(rf_kl, X, full_output = TRUE)  # control, treatment1, delta_treatment1
```

``` bash
Rscript tests/test-uplift-trees.R
```

#### Uplift trees test results (n=1200, 25% test)

| Model                 | Mean predicted CATE (test) |
|-----------------------|----------------------------|
| uplift_tree_ddp       | -0.04                      |
| uplift_tree_iddp      | -0.04                      |
| interaction_tree      | -0.04                      |
| causal_inference_tree | -0.02                      |
| uplift_rf_kl          | -0.02                      |
| uplift_rf_ed          | -0.01                      |
| uplift_rf_cts         | -0.02                      |

**Example output:**

```         
========== Synthetic uplift data ==========
n = 1200, p = 8
Treatment rate: 0.486
Conversion (control): 0.509 | treatment: 0.487

---- Single trees: DDP, IDDP, IT, CIT ----
uplift_tree_ddp: fitted OK; mean predicted CATE (test) = -0.0409
uplift_tree_iddp: fitted OK; mean predicted CATE (test) = -0.0409
interaction_tree: fitted OK; mean predicted CATE (test) = -0.0426
causal_inference_tree: fitted OK; mean predicted CATE (test) = -0.0193

---- Uplift tree (DDP) structure (first few lines) ----
X2 >= 0.5270?
  yes -> X3 >= -0.5483?
      yes -> X1 >= 0.5769?
          yes -> [tau=0.3176, n=57]
          no  -> X7 >= -0.7006?
...

---- Uplift random forests (n_trees=5 for speed) ----
uplift_rf_kl: mean CATE (test) = -0.0208
  full_output: control/treatment1/delta columns OK
uplift_rf_ed: mean CATE (test) = -0.013
uplift_rf_cts: mean CATE (test) = -0.0223

---- Multi-treatment uplift forest ----
uplift_rf_multi: fitted OK
  full_output cols: control, treatment2, treatment1, recommended_treatment, delta_treatment2, delta_treatment1, max_delta
  recommended_treatment sample: treatment2 control treatment2 control treatment2 control

========== test-uplift-trees.R done ==========
```

### Run uplift random forest test {#run-uplift-random-forest-test}

[`tests/test-uplift_randomForest.R`](tests/test-uplift_randomForest.R) exercises **`uplift_randomForest()`** ([R/uplift_randomForest.R](R/uplift_randomForest.R)): binary classification (KL), multi-arm classification, regression (T-learner), and regression (IT trees).

``` bash
Rscript tests/test-uplift_randomForest.R
```

### Run DR-Learner test {#run-dr-learner-test}

[`tests/test-DRLearner.R`](tests/test-DRLearner.R) exercises all DR-Learner variants in `R/DRLearner.R` on the same synthetic DGP (X, W, propensity, true CATE non-linear in X1). It runs: **DRLearner** (ranger), **fit_predict** with return_components, **LinearDRLearner**, **SparseLinearDRLearner**, **XGBDRLearner** (if xgboost is installed), **ForestDRLearner**, DRLearner with estimated propensity, and **estimate_ate** with `pretrain = TRUE`. For linear variants it also checks **coef** and **intercept**.

``` bash
# Quick run (n=300)
QUICK=1 Rscript tests/test-DRLearner.R

# Full run (n=1000)
Rscript tests/test-DRLearner.R
```

#### DR-Learner test results (quick run, `QUICK=1`, n=300)

| Variant | Cor(pred CATE, true CATE) | MSE(pred vs true CATE) | ATE estimate (DR) |
|----|----|----|----|
| DRLearner (ranger) | 0.79 | 0.18 | 1.25 [1.12, 1.37] |
| LinearDRLearner | 0.74 | 0.21 | — |
| SparseLinearDRLearner | 0.75 | 0.44 | — |
| ForestDRLearner | \~0.79 | \~0.18 | — |
| XGBDRLearner | (if xgboost installed) | — | — |

True ATE ≈ 1.31. All DR variants run successfully; DRLearner/ForestDRLearner give the highest correlation with the true CATE in this setup.

### Run DMLearner test {#run-dmlearner-test}

[`tests/test-DMLearner.R`](tests/test-DMLearner.R) exercises the **Double Machine Learning** module (`R/DMLearner.R`): **LinearDML**, **SparseLinearDML**, **KernelDML**, **NonParamDML**, **CausalForestDML** (when `grf` is installed), base **DMLearner**, **fit_predict** with return_components, **estimate_ate**, and **coef** / **intercept** for linear variants. It also runs **DynamicDMLearner** (panel/sequential treatments) and **OrthoIVLearner**, **DMLIVLearner**, **NonParamDMLIVLearner** (DML with IV), all implemented in native R. Uses the same style of synthetic data (DGP with linear CATE in X1). API aligned with [EconML](https://econml.azurewebsites.net/) (Python); optional [DoubleML](https://docs.doubleml.org) (R) via `doubleml_plr()`.

``` bash
# Quick run (n=300)
QUICK=1 Rscript tests/test-DMLearner.R

# Full run (n=800)
Rscript tests/test-DMLearner.R
```

#### DMLearner test: minimal code example

``` r
library(RCausalML)
set.seed(42)
n <- 300
X <- matrix(rnorm(n * 5), n, 5)
colnames(X) <- paste0("X", 1:5)
treatment <- rbinom(n, 1, 1 / (1 + exp(-X[, 1])))
true_cate <- 0.3 + 0.5 * X[, 1]
y <- 2 + 0.3 * X[, 1] + true_cate * treatment + rnorm(n, 0, 0.5)

# LinearDML (EconML-style)
dml <- LinearDML(model_y = "ranger", model_t = "ranger", n_fold = 3, seed = 123)
dml <- fit(dml, X, treatment, y)
te <- predict(dml, X)
ate <- estimate_ate(dml, X, treatment, y, pretrain = TRUE, return_ci = TRUE)

# SparseLinearDML, KernelDML, NonParamDML, CausalForestDML (if grf installed)
# dml_sparse <- SparseLinearDML(n_fold = 3); dml_sparse <- fit(dml_sparse, X, treatment, y)
# dml_cf <- CausalForestDML(n_fold = 3); dml_cf <- fit(dml_cf, X, treatment, y)

# Dynamic DML (panel) and DML with IV (native R)
# DynamicDMLearner, OrthoIVLearner, DMLIVLearner, NonParamDMLIVLearner
# dyn <- DynamicDMLearner(cv = 2); dyn <- fit(dyn, Y, T, X, groups = unit_id)
# pred_dyn <- predict(dyn, X_new); eff <- effect(dyn, newdata = X_new, T0 = 0, T1 = 1)
# oiv <- OrthoIVLearner(cv = 2); oiv <- fit(oiv, Y = y, T = T, Z = Z, X = X)
# cate_iv <- predict(oiv, X_new)
```

#### DMLearner test results (quick run, `QUICK=1`, n=300)

| Variant | Cor(pred CATE, true CATE) | MSE(pred vs true CATE) | ATE estimate |
|----|----|----|----|
| LinearDML | 0.98 | 0.01 | 0.31 [0.26, 0.36] |
| SparseLinearDML | 1.00 | 0.06 | — |
| KernelDML | \~−0.04 | \~0.33 | — |
| NonParamDML (ranger) | 0.61 | 0.33 | — |
| CausalForestDML | 0.91 | 0.08 | 0.31 [0.28, 0.34] |
| DynamicDMLearner | (panel; native R) | — | — |
| OrthoIV / DMLIV / NonParamDMLIV | (IV; native R) | — | — |

True ATE ≈ 0.29. LinearDML and CausalForestDML give strong correlation with the true CATE in this setup. **DynamicDMLearner** (panel) and **OrthoIVLearner**, **DMLIVLearner**, **NonParamDMLIVLearner** (IV) are implemented in native R (no Python required).

**Example output:**

```         
========== DMLearner test: n = 300, p = 5 ==========
True ATE: 0.2891

---- 1. LinearDML ----
  Cor(pred CATE, true CATE): 0.9765
  MSE(pred vs true CATE): 0.013
  coef (first 3): 0.4455, -0.0647, 0.0215
  intercept: 0.3129
  ATE: 0.3129 [0.2622, 0.3636]

---- 2. SparseLinearDML ----
  Cor(pred CATE, true CATE): 1
  MSE(pred vs true CATE): 0.0611
  ...

---- 5. CausalForestDML ----
  Cor(pred CATE, true CATE): 0.905
  MSE(pred vs true CATE): 0.0781
  ATE: 0.3087 [0.279, 0.3385]

========== DMLearner test summary ==========
True ATE: 0.2891
LinearDML ATE: 0.3129
All DMLearner variants ran successfully.
========== test-DMLearner.R done ==========
```

### Run neural test {#run-neural-test}

[`tests/test-neural.R`](tests/test-neural.R) runs the **neural network–based algorithms** in RCausalML. It sets `RCAUSALML_SOURCE_ROOT` so **DCEVAE** can find `inst/python` and `inst/dcevae` when the script `source()`s [R/causalDeepNet.R](R/causalDeepNet.R) without installing the package. Available models:

-   **CEVAE** — Counterfactual Variational Autoencoder; generative model with latent confounders (uses **torch** when available). Ref: [Louizos et al. (2017)](http://papers.nips.cc/paper/7223-causal-effect-inference-with-deep-latent-variable-models.pdf).
-   **DragonNet** — Shared representation plus three heads and targeted regularization (uses **torch** when available).
-   **TARNet** — Shared representation plus two heads (treatment-agnostic).
-   **CFRNet** — TARNet with MMD balancing.
-   **GANITE** — Causal inference with GANs: Generator, Discriminator, and Inference net (uses **torch** when available). The test passes **`iterations`** (not `epochs`) to control training length.
-   **CausalGAN** — Structural equation GAN for $(X, T, Y)$ with node-wise generators and interventional sampling (uses **torch** when available). The test path can run with small `epochs` for quick validation.
-   **DSCM** — Deep Structural Causal Model with latent abduction and abduction-action-prediction counterfactual inference (uses **torch** when available).
-   **CausalDiscrepancyVAE** — Discrepancy-regularized VAE with latent MMD balancing and treatment-specific outcome heads (uses **torch** when available).
-   **Neural Granger models** — `neural_granger_ml()` wrapper for **cMLP**, **cLSTM**, **EconomySRU**, and **NRI** on multivariate time series (uses **torch** when available).
-   **DCEVAE** — Tabular deep counterfactual equivariant VAE (Python implementation under `inst/dcevae/DCEVAE_ours/`, called via **`reticulate`**). Skipped if **`reticulate`** is not installed or Python lacks `torch` / `numpy` / `tqdm`.

``` bash
# Quick run (fewer epochs / iterations)
QUICK=1 Rscript tests/test-neural.R

# Full run
Rscript tests/test-neural.R
```

#### Neural test (n=300, fewer epochs)

| Model | Cor(pred CATE, true CATE) | PEHE | Mean pred CATE |
|----|----|----|----|
| CEVAE | 0.855 | 0.40 | 1.43 |
| DragonNet | 0.915 | 0.31 | 1.41 |
| TARNet | 0.910 | 0.38 | 1.41 |
| CFRNet | 0.923 | 0.33 | 1.46 |
| GANITE | (varies; quick mode uses 80 iterations) | — | — |
| CausalGAN | (varies; quick mode can use 20-40 epochs) | — | — |
| DSCM | (new torch model; deterministic ITE + AAP counterfactual prediction) | — | — |
| CausalDiscrepancyVAE | (new torch model; evaluate with `predict(..., type = "ite")`) | — | — |
| DCEVAE | (counterfactual y contrast via `predict(fit, r, d, a, y)`; not same metric as CATE cor.) | — | — |

True ATE (mean of `true_cate`) ≈ 1.30. The first five learners target CATE vs the synthetic DGP; **DCEVAE** is exercised as fit + `predict` on a binary outcome and split covariates (`r`, `d`) when optional dependencies are present.

### Run attention/transformer test

[`R/causalDeepNet.R`](R/causalDeepNet.R) exposes three attention-based causal models for multivariate time series (v0.2.0, from `03_attention_transforner_causalML.ipynb`):

-   **TCDFNet** (`tcdf_model`) — stacked causal dilated convolutions + attention head; inspired by Nauta et al. (2019).
-   **CausalTransformer** (`causal_transformer_model`) — Transformer encoder with autoregressive mask + inter-variable cross-attention.
-   **TFT** (`tft_model`) — Temporal Fusion Transformer (Lim et al., 2021): variable-selection networks, LSTM encoder, multi-head temporal attention.

All three are also available through the unified wrapper `attn_causal_model()`. Causal graphs (variable-to-variable influence matrices) are extracted with `causal_matrix_attn()`. Requires `torch`.

``` r
library(RCausalML)
library(torch)

set.seed(42)
d <- 5L; T_len <- 300L
A <- matrix(0, d, d)
for (i in seq_len(d)) { A[i,i] <- 0.45; if (i > 1) A[i, i-1] <- 0.25 }
x <- matrix(0, T_len, d)
x[1, ] <- rnorm(d)
for (t in 2:T_len) x[t, ] <- x[t-1, ] %*% t(A) + 0.15 * rnorm(d)
colnames(x) <- paste0("V", seq_len(d))

# Fit all three models jointly
fit <- attn_causal_model(x, lag = 10L,
                         models = c("tcdf", "causal_transformer", "tft"),
                         epochs = 20L, d_model = 64L, hidden = 48L, verbose = TRUE)

# Validation MSE per model
print(fit$val_mse)

# Causal matrices (variable-to-variable influence)
print(round(causal_matrix_attn(fit, "tcdf"), 3))
print(round(causal_matrix_attn(fit, "causal_transformer"), 3))
print(round(causal_matrix_attn(fit, "tft"), 3))

# One-step-ahead predictions
ds   <- RCausalML:::.deepnet_attn_build_dataset(x, lag = 10L)
preds <- predict(fit, model = "causal_transformer", newdata = ds$X[1:5, , ])

# Convenience single-model wrappers
fit_tcdf <- tcdf_model(x, lag = 10L, epochs = 20L)
fit_ctrf <- causal_transformer_model(x, lag = 10L, epochs = 20L, d_model = 64L)
fit_tft  <- tft_model(x, lag = 10L, epochs = 20L, d_model = 64L)
```

#### Attention/transformer test results (n=300, d=5, lag=10, 5 epochs)

| Model             | Val MSE                    |
|-------------------|----------------------------|
| TCDF              | \~0.025                    |
| CausalTransformer | \~0.025                    |
| TFT               | \~0.18 (needs more epochs) |

### Run RNN/LSTM causal model test {#run-rnnlstm-causal-model-test}

[`R/causalDeepNet.R`](R/causalDeepNet.R) exposes three RNN/LSTM-based causal models for multivariate time series (v0.2.0, from `04_rnn_lstm_causalML.ipynb`):

-   **CausalLSTM** (`causal_lstm_model`) — per-variable LSTM with a learnable sparse causal-adjacency mask $G \in [0,1]^{d \times d}$; the mask is regularised with an L1 sparsity penalty so only true predictors survive.
-   **RETAIN** (`retain_model`) — REverse Time AttentIoN (Choi et al., 2016): two-channel GRU in reverse time — temporal attention $\alpha$ (which step matters) + variable attention $\beta$ (which variable at that step) — produces a per-step attribution matrix for interpretability.
-   **Intervention-Aware RNN** (`intervention_rnn_model`) — GRU regime detector infers latent system states; learned regime embeddings and an explicit binary/continuous intervention-indicator channel are concatenated to the input before the LSTM forecaster; exposes regime-conditioned causal effect matrices.

All three are available through the unified wrapper `rnn_causal_model()`. Causal graphs are extracted with `causal_matrix_rnn()`. Requires `torch`.

``` r
library(RCausalML)
library(torch)

set.seed(42)
d <- 5L; T_len <- 300L
A <- matrix(0, d, d)
for (i in seq_len(d)) { A[i,i] <- 0.45; if (i > 1) A[i, i-1] <- 0.25 }
x <- matrix(0, T_len, d)
x[1, ] <- rnorm(d)
for (t in 2:T_len) x[t, ] <- x[t-1, ] %*% t(A) + 0.15 * rnorm(d)
colnames(x) <- paste0("V", seq_len(d))

# Intervention vector (e.g., volatility-spike proxy, top 25% quantile)
interv <- as.numeric(apply(x, 1, function(r) sum(r^2)) >
                     quantile(apply(x, 1, function(r) sum(r^2)), 0.75))

# Fit all three models jointly
fit <- rnn_causal_model(x, lag = 10L,
                        models = c("causal_lstm", "retain", "intervention_rnn"),
                        intervention = interv,
                        epochs = 30L, hidden = 32L, verbose = TRUE)

# Validation MSE per model
print(fit$val_mse)

# Causal matrices (variable-to-variable influence)
print(round(causal_matrix_rnn(fit, "causal_lstm"), 3))
print(round(causal_matrix_rnn(fit, "retain"), 3))
print(round(causal_matrix_rnn(fit, "intervention_rnn"), 3))

# One-step-ahead predictions
ds    <- RCausalML:::.deepnet_rnn_build_dataset(x, lag = 10L, intervention = interv)
preds <- predict(fit, model = "causal_lstm", newdata = ds$X[1:5, , ])

# Convenience single-model wrappers
fit_lstm   <- causal_lstm_model(x, lag = 10L, epochs = 30L)
fit_retain <- retain_model(x, lag = 10L, epochs = 30L)
fit_irnn   <- intervention_rnn_model(x, lag = 10L, epochs = 30L,
                                     intervention = interv)
```

#### RNN/LSTM test results (n=300, d=5, lag=10, 30 epochs)

| Model           | Val MSE |
|-----------------|---------|
| CausalLSTM      | \~0.030 |
| RETAIN          | \~0.030 |
| InterventionRNN | \~0.030 |

### Run policy learner test {#run-policy-learner-test}

[`tests/test-policy-learner.R`](tests/test-policy-learner.R) runs all **policy learners** in `R/policy_learner.R` on synthetic data from `synthetic_data(mode = 1, ...)`: (1) **policy_learner** (Athey & Wager: DR score + weighted classifier), (2) **DRPolicyTree** (EconML-style: cross-fitted outcome + policy tree), (3) **DRPolicyForest** (ensemble of policy trees, majority vote). The test compares each model's policy value and agreement with the oracle rule "treat when τ \> 0".

``` bash
Rscript tests/test-policy-learner.R
```

#### Policy learner test results (n=800, p=6, mode 1)

| Model          | Value (policy) | Value (oracle) | Agreement with oracle |
|----------------|----------------|----------------|-----------------------|
| policy_learner | 0.488          | 0.488          | 1.00                  |
| DRPolicyTree   | 0.488          | 0.488          | 1.00                  |
| DRPolicyForest | 0.488          | 0.488          | 1.00                  |

`fit`, `predict`, and `predict_proba` run successfully. Mean CATE (true) ≈ 0.49; treatment rate ≈ 0.50. All three learners achieve full agreement with the oracle in this setup. For tree/forest, `feature_importances` is also available for tree/forest.

### Run CATE interpreter test {#run-cate-interpreter-test}

[`tests/test-cate-interpreter.R`](tests/test-cate-interpreter.R) runs the **CATE interpreters** in `R/cate_interpreter.R` (EconML-style): (1) **SingleTreeCateInterpreter** — fits a single regression tree to the predicted CATE from any fitted estimator (e.g. SLearner, TLearner) to summarize effect heterogeneity; (2) **SingleTreePolicyInterpreter** — fits a single classification tree over the optimal treatment (control vs treat) from CATE to summarize the policy. The test uses `synthetic_data(mode = 1, ...)` and checks `interpret()`, `predict()`, `treat()`, `node_dict_`, `policy_value_`, and error handling (e.g. `treat()` before `interpret()`).

``` bash
Rscript tests/test-cate-interpreter.R
```

#### CATE interpreter test (n=400, p=6, mode 1)

| Component | Checks |
|----|----|
| SingleTreeCateInterpreter + SLearner | `interpret()`, `tree_model_`, `node_dict_`, `predict(X)` and `predict(newdata)` |
| SingleTreeCateInterpreter + TLearner | Same with TLearner as CATE estimator |
| SingleTreePolicyInterpreter + SLearner | `interpret()`, `policy_value_`, `always_treat_value_`, `treat(X)`, `predict(X)` |
| sample_treatment_costs | Policy interpreter with scalar cost |
| Error handling | `treat()` before `interpret()` raises error |
| data.frame X | CATE interpreter accepts data.frame |
| node_dict\_ | Entries have `mean` and `std` |

All seven test blocks pass. CATE interpreter API aligns with [EconML cate_interpreter](https://econml.azurewebsites.net/) (SingleTreeCateInterpreter, SingleTreePolicyInterpreter).

### Run DAG-GNN test {#run-dag-gnn-test}

[`tests/test-dag_gnn.R`](tests/test-dag_gnn.R) runs the **DAG-GNN** implementation in [R/causalDeepNet.R](R/causalDeepNet.R). It checks: **get_daggnn_device**, **preprocess_adj** / **matrix_poly**, **DAGGNN** forward pass, **elbo_loss** / **h_func**, **make_daggnn**, and **daggnn_adj**. Requires **torch**.

``` bash
Rscript tests/test-dag_gnn.R
```

### Run GraN-DAG test {#run-gran-dag-test}

[`tests/test-GraN_DAG.R`](tests/test-GraN_DAG.R) runs the **GraN-DAG** (gradient-based neural DAG learner) implementation in [R/causalDeepNet.R](R/causalDeepNet.R). It checks: **is_acyclic**, **compute_constraint**, **NormalizationData**, **GraNDAG** constructor, and **GraNDAG\$learn()** with a short training run. Requires **torch**, **R6**, **expm**; optional **ranger** and **progress** for PNS and progress bars.

``` bash
# Quick run (fewer iterations)
QUICK=1 Rscript tests/test-GraN_DAG.R

# Full run
Rscript tests/test-GraN_DAG.R
```

### Run causalStructureML test {#run-causalstructureml-test}

[`tests/test-causalStructureML.R`](tests/test-causalStructureML.R) exercises the **unified causal structure learning API** in [R/causalStructureML.R](R/causalStructureML.R): **`causal_structure_ml_model_descriptions()`**, **`causalStructureML()`** with `method` set to `notears_linear`, `dag_gnn`, `notears_nonlinear_mlp`, `notears_nonlinear_sobolev`, and `grandag`, plus **data.frame** input. Uses **`devtools::load_all()`**. Requires **expm**, **igraph**, and **torch** (for all methods except `notears_linear`); install **`devtools`** for the test runner.

``` bash
QUICK=1 Rscript tests/test-causalStructureML.R
Rscript tests/test-causalStructureML.R
```

### Run NOTEARS test {#run-notears-test}

[`tests/test-notears.R`](tests/test-notears.R) runs the **NOTEARS** implementation in [R/causalDeepNet.R](R/causalDeepNet.R). It checks: (1) **set_random_seed**, (2) **is_dag**, (3) **simulate_dag** (ER, SF), (4) **simulate_parameter**, (5) **simulate_linear_sem** (gauss, uniform), (6) **count_accuracy**, (7) **notears_linear** (L-BFGS-B + augmented Lagrangian, l2 loss), (8) **demo_linear**, and (9) **notears_nonlinear** with **NotearsMLP** when **torch** is available (Adam + L-BFGS, custom autograd for trace(expm)). Requires **expm** and **igraph**; **torch** is optional for the nonlinear section.

``` bash
# Full run (includes nonlinear if torch installed)
Rscript tests/test-notears.R

# Quick run (skip nonlinear)
QUICK=1 Rscript tests/test-notears.R
```

Install **expm** and **igraph** for NOTEARS: `install.packages(c("expm", "igraph"))`. For nonlinear NOTEARS (MLP/Sobolev), install **torch**: `install.packages("torch")`.

### Run causal structure learning test {#run-causal-structure-learning-test}

[`tests/test-causal_structure_learning.R`](tests/test-causal_structure_learning.R) runs **causal structure learning** algorithms (pure R implementations in `R/Causal_structure_learning.R`):

-   **notrears (linear)** — NOTEARS linear (Zheng et al.); requires **expm** and uses `optim` (L-BFGS-B) with augmented Lagrangian.
-   **notrears (nonlinear)** — NOTEARS with learnable adjacency (R **torch**); acyclicity via matrix polynomial.
-   **DAG_GNN** — DAG structure learning with GNN (Yu et al.); VAE + acyclicity (R **torch**).
-   **GraN-DAG** — GraN-DAG (Lachapelle et al.); gradient-based neural DAG learning (R **torch**).
-   **notrears with tune** — Optional Optuna (reticulate) hyperparameter tuning when not in `QUICK` mode; requires **expm** for NOTEARS linear.

``` bash
# Quick run (small n, d; no tune)
QUICK=1 Rscript tests/test-causal_structure_learning.R

# Full run (includes optional Optuna tune)
Rscript tests/test-causal_structure_learning.R
```

#### Causal structure learning test results (n=200, d=6)

| Section | Result |
|----|----|
| 1\. notrears (linear) | Requires `install.packages("expm")`. With expm: returns `adjacency` (d×d), `binary_adjacency`. |
| 2\. notrears (nonlinear) | OK (torch). Returns weighted and binary adjacency; NNZ (edges) depends on data and `w_threshold`. |
| 3\. DAG_GNN | OK (torch). Returns learned adjacency; NNZ (edges) and final loss reported. |
| 4\. GraN-DAG (GraN-DAG) | OK. Returns `adjacency`, `binary_adjacency`; NNZ (edges) reported. |
| 5\. notrears with tune (optuna) | Optional; runs 3 Optuna trials when not `QUICK`. Requires expm for NOTEARS linear. |

**Example output** (all sections pass with **expm**, **torch**, and optional **reticulate** + **optuna** for tune):

```         
========== Causal structure learning tests: n = 200, d = 6 ==========

---- 1. notrears (linear) ----
  Adjacency dim: 6 x 6
  NNZ (edges): 0
  OK

---- 2. notrears (nonlinear) ----
  NNZ (edges): 0
  OK

---- 3. DAG_GNN ----
  NNZ (edges): 20
  Final loss: 0.144771
  OK

---- 4. GraN-DAG (GraN-DAG) ----
  NNZ (edges): 30
  OK

---- 5. notrears with tune (optuna) [optional] ----
  Best trial: ... Best value: ...
  Best lambda1: ...
  OK

========== test-causal_structure_learning.R done ==========
```

Install **expm** for NOTEARS linear and tune: `install.packages("expm")`. Dependencies: **torch** (R), **expm** (NOTEARS linear), **reticulate** + **optuna** (Python, only when `tune=TRUE`).

## Example Notebooks (`inst/examples/`)

Example notebooks (`.qmd`) live under **`inst/examples/Notebook/`**, grouped **by topic** in subfolders, and mirror the [Python CausalML examples](https://causalml.readthedocs.io/en/latest/examples.html). Real/semi-synthetic data (e.g. IHDP, card.csv) can be downloaded from [causalml/docs/examples/data](https://github.com/uber/causalml/tree/master/docs/examples/data) and placed in `inst/examples/data/`. See [**inst/examples/README.md**](inst/examples/README.md) for more detail.

**Topic index:** [**inst/examples/Notebook/INDEX.qmd**](inst/examples/Notebook/INDEX.qmd) — entry point that lists all notebooks by topic with links. Render from package root:

``` bash
quarto render inst/examples/Notebook/INDEX.qmd
```

**Notebooks by topic** (in `inst/examples/Notebook/`):

| Topic folder | Notebooks | Description |
|----|----|----|
| **01-Meta-Learners** | 01–05 | Meta-learners (S/T/X/R): training, ATE/CATE, validation, interpretation; benchmarks (Nie & Wager, Schuler et al.) |
| **02-Neural-Network** | 06–07 | DragonNet, TARNet, CFRNet; CEVAE, GANITE vs meta-learners (IHDP + synthetic); see also `DCEVAE()` in [R/causalDeepNet.R](R/causalDeepNet.R) (Quarto deep-causal learners under `inst/notebooks/05-deep-causal-learners/`) |
| **03-Uplift and causal trees-forests** | 00, 08–14 | Data generation, uplift trees/forests, visualization, TMLE curves, feature selection, causal trees, SHAP, Qini |
| **04-Instrumental variables, Doubly Robust Learning and Double ML** | 15–20 | IV/2SLS, DR/DRIV benchmark, DML use cases, OrthoIV/DRIV, dynamic DML, weighted DML |
| **05-Counterfactual reasoning and policy learning** | 21–23 | Unit selection, value estimation (Li & Pearl), policy learner (Athey & Wager) |
| **06-Sensitivity and calibration** | 24–25 | Sensitivity analysis (placebo, selection bias, etc.); propensity calibration |
| **08-Causal Structure Learning Models** | 29+ | NOTEARS, DAG-GNN, GraN-DAG (R and Python examples) |

**Case studies** (in `inst/examples/Case_study/`): customer segmentation and individualized incentive response; uplift classification data generation (`make_uplift_classification`). See [inst/examples/Case_study/](inst/examples/Case_study/) for files.

**Render all notebooks in a topic folder** (from package root), e.g.:

``` bash
quarto render "inst/examples/Notebook/01-Meta-Learners/*.qmd"
quarto render "inst/examples/Notebook/03-Uplift and causal trees-forests/*.qmd"
```

## Acknowledgement

This package is an independent R port inspired by the Python **CausalML** project. We thank the [CausalML (Python) authors and contributors](https://github.com/uber/causalml) for the original library, documentation, and examples. The R package mirrors their API and examples where practical and acknowledges Python CausalML as the primary reference. Example notebooks in `inst/examples/` are aligned with the [Python CausalML documentation](https://causalml.readthedocs.io/en/latest/examples.html).

-   **Python CausalML documentation:** [causalml.readthedocs.io](https://causalml.readthedocs.io/en/latest/about.html)
-   **Python CausalML source:** [github.com/uber/causalml](https://github.com/uber/causalml)

## References

-   Athey, S., & Wager, S. (2018). *Policy learning with observational data.* arXiv:1702.02896.
-   Chen et al. (2020). *CausalML: Python Package for Causal Machine Learning.* arXiv:2002.11631.
-   Künzel et al. (2019). *Metalearners for estimating heterogeneous treatment effects using machine learning.* PNAS.
-   Nie & Wager (2018). *Quasi-Oracle Estimation of Heterogeneous Treatment Effects.* arXiv:1712.04912.

## License

Apache License 2.0

------------------------------------------------------------------------

**Contact:** Zia Ahmed, PhD, Upatta Analytics, NY, USA

![](images/upatta_logo-01.png){width="200"}
