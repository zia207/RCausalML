# RCausalML Package Help — All Functions

`RCausalML` (current development version: **0.3.0**) is an R port of the
Python CausalML library.  It provides uplift modelling, CATE estimation,
causal structure learning, deep causal networks, and policy-learning tools.
The sections below cover every exported function grouped by topic.

---

## 1. Installation

### Ubuntu (Linux) — Prerequisites

Install system libraries required by R packages with native code **before**
installing RCausalML or its dependencies:

```bash
sudo apt update && sudo apt install -y \
  r-base r-base-dev                \
  libssl-dev libcurl4-openssl-dev  \
  libxml2-dev libgit2-dev          \
  libfontconfig1-dev               \
  libharfbuzz-dev libfribidi-dev   \
  libfreetype6-dev libpng-dev      \
  libtiff5-dev libjpeg-dev         \
  liblapack-dev libblas-dev        \
  libgfortran5                     \
  python3 python3-pip python3-dev  \
  cmake                            \
  build-essential gfortran
```

> **R version:** RCausalML requires R ≥ 4.0.0.
> Install the latest R from [CRAN](https://cran.r-project.org/bin/linux/ubuntu/)
> or the official Ubuntu PPA:
> ```bash
> sudo apt install -y --no-install-recommends software-properties-common dirmngr
> wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
>   | sudo tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
> sudo add-apt-repository \
>   "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
> sudo apt update && sudo apt install -y r-base r-base-dev
> ```

#### Ubuntu — Install RCausalML from tarball

```bash
# Install from a terminal
R CMD INSTALL RCausalML_0.3.0.tar.gz

# Or to a local library (no root required)
mkdir -p ~/.Rlibrary
R CMD INSTALL -l ~/.Rlibrary RCausalML_0.3.0.tar.gz
```

```r
# Inside R
install.packages("RCausalML_0.3.0.tar.gz", repos = NULL, type = "source")
```

#### Ubuntu — Install core dependencies first (recommended)

```r
options(repos = c(CRAN = "https://cloud.r-project.org"))

core_pkgs <- c(
  "glmnet", "ranger", "rpart", "R6", "expm", "igraph", "progress",
  "htetree", "rlang", "future", "future.apply",
  "ggplot2", "ggraph", "parsnip", "coro", "MASS", "Matrix"
)
install.packages(core_pkgs)

# Optional but recommended
install.packages(c(
  "xgboost", "grf", "DoubleML",
  "mlr3", "mlr3learners", "mlr3tuning", "paradox", "mlr3measures",
  "MatchIt", "torch", "reticulate",
  "kernelshap", "shapviz", "causaldata",
  "nnet", "rpart.plot", "reshape2", "shapr"
))

# torch backend (CPU by default; GPU requires matching CUDA toolkit)
torch::install_torch()
```

#### Ubuntu — reticulate / Python setup

```r
# One-time Miniconda install (only needed for reticulate-backed models)
reticulate::install_miniconda()
reticulate::conda_create("r-rcausalml", python_version = "3.10")
reticulate::use_condaenv("r-rcausalml", required = TRUE)
reticulate::conda_install("r-rcausalml",
  c("torch", "numpy", "scipy", "scikit-learn"))
```

---

### Windows — Prerequisites

1. **Install R** (≥ 4.0.0) from <https://cran.r-project.org/bin/windows/base/>.
2. **Install Rtools** matching your R version from
   <https://cran.r-project.org/bin/windows/Rtools/>:
   - R 4.4.x → **Rtools44**
   - R 4.3.x → **Rtools43**
   - R 4.2.x → **Rtools42**
3. During Rtools installation, tick **"Add Rtools to system PATH"**
   (or run `writeLines('PATH="${RTOOLS44_HOME}\\usr\\bin;${PATH}"', con = "~/.Renviron")`
   from R).
4. Verify Rtools is on PATH by opening R and running:
   ```r
   Sys.which("make")   # should return a non-empty path
   ```

#### Windows — Install RCausalML from tarball

Open **R** (or RStudio) and run:

```r
# Ensure CRAN mirror is set
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install from source tarball (Rtools required for packages with native code)
install.packages("RCausalML_0.3.0.tar.gz", repos = NULL, type = "source")
```

Or from the **Windows Command Prompt / PowerShell**:

```powershell
R CMD INSTALL RCausalML_0.3.0.tar.gz

# Install to a user library (no admin rights needed)
R CMD INSTALL --library="%USERPROFILE%\R\library" RCausalML_0.3.0.tar.gz
```

#### Windows — Install core dependencies (binary, recommended)

On Windows, always prefer `type = "binary"` to avoid compiling from source:

```r
options(repos = c(CRAN = "https://cloud.r-project.org"))

core_pkgs <- c(
  "glmnet", "ranger", "rpart", "R6", "expm", "igraph", "progress",
  "htetree", "rlang", "future", "future.apply",
  "ggplot2", "ggraph", "parsnip", "coro", "MASS", "Matrix"
)
install.packages(core_pkgs, type = "binary")

# Optional but recommended
install.packages(c(
  "xgboost", "grf", "DoubleML",
  "mlr3", "mlr3learners", "mlr3tuning", "paradox", "mlr3measures",
  "MatchIt", "torch", "reticulate",
  "kernelshap", "shapviz", "causaldata",
  "nnet", "rpart.plot", "reshape2", "shapr"
), type = "binary")

# torch backend — downloads LibTorch DLL automatically
torch::install_torch()
```

#### Windows — reticulate / Python setup

```r
# One-time Miniconda install
reticulate::install_miniconda()
reticulate::conda_create("r-rcausalml", python_version = "3.10")
reticulate::use_condaenv("r-rcausalml", required = TRUE)
reticulate::conda_install("r-rcausalml",
  c("torch", "numpy", "scipy", "scikit-learn"))
```

> **Note:** If you see `DLL load failed` errors for `torch` on Windows, make
> sure the Microsoft Visual C++ Redistributable (≥ 2019) is installed from
> <https://aka.ms/vs/17/release/vc_redist.x64.exe>.

---

### Load

```r
library(RCausalML)

# If installed to a local library
.libPaths(c("~/.Rlibrary", .libPaths()))   # Ubuntu
.libPaths(c(file.path(Sys.getenv("USERPROFILE"), "R", "library"), .libPaths()))  # Windows
library(RCausalML)
```

---

## 2. Dependencies

### Core (Imports — always required)

`stats`, `utils`, `Matrix`, `parallel`, `MASS`, `glmnet (>= 4.0)`,
`ranger (>= 0.12)`, `rpart (>= 4.1)`, `R6`, `expm`, `igraph`, `progress`,
`htetree`, `rlang`, `future`, `future.apply`, `ggplot2`, `ggraph`, `parsnip`, `coro`

### Optional (Suggests — install as needed)

```r
install.packages(c(
  "xgboost",      # CXGBoost, MultiArmCausalBoost, BoostedIVForest
  "grf",          # grf_causal_forest, causal_survival_forest, etc.
  "DoubleML",     # doubleml_* bridge functions
  "mlr3", "mlr3learners", "mlr3tuning", "paradox", "mlr3measures",
  "MatchIt",      # match_by_group, match_optimizer
  "torch",        # all deep / neural causal models (>= 0.10)
  "reticulate",   # Python-backed models (DCEVAE, etc.)
  "kernelshap", "shapviz",   # explain_cate / SHAP integration
  "causaldata",   # load_causaldata, list_causaldata_datasets
  "nnet", "rpart.plot", "reshape2", "shapr"
))
```

---

## 3. Quick Smoke Test

```r
library(RCausalML)
set.seed(42)
d   <- synthetic_data(mode = 1, n = 400, p = 5, sigma = 1)
m   <- SLearner(learner = "ranger")
m   <- fit(m, d$X, d$w, d$y)
ate <- estimate_ate(m, d$X, d$w, d$y)
print(ate)
```

---

## 4. Synthetic & Simulation Data

| Function | Description |
|---|---|
| `synthetic_data(mode, n, p, sigma, ...)` | Generate a synthetic CATE dataset (modes 1–6). Returns `list(X, w, y, tau)`. |
| `simulate_nuisance_and_easy_treatment(n, p, ...)` | Nuisance-heavy DGP with easy propensity. |
| `simulate_randomized_trial(n, p, ...)` | Balanced RCT simulation. |
| `simulate_easy_propensity_difficult_baseline(n, p, ...)` | Easy propensity / hard outcome. |
| `simulate_unrelated_treatment_control(n, p, ...)` | No treatment effect DGP. |
| `simulate_hidden_confounder(n, p, ...)` | DGP with a hidden confounder. |
| `make_uplift_classification(n, n_features, ...)` | Binary uplift classification dataset. |
| `make_uplift_regression(n, n_features, ...)` | Continuous uplift regression dataset. |
| `generate_data(n, sem_type, d, graph_type, ...)` | Generate data from a specified SEM/DAG (used by causal structure learning). |
| `simulate_dag(d, s0, graph_type)` | Simulate a random DAG adjacency matrix. |
| `simulate_linear_sem(W, n, sem_type, ...)` | Simulate linear SEM from a DAG `W`. |
| `simulate_nonlinear_sem(B, n, sem_type, ...)` | Simulate nonlinear SEM. |
| `simulate_parameter(d, graph_type, ...)` | Sample random SEM parameters. |

---

## 5. Propensity Score Estimation & Matching

| Function | Description |
|---|---|
| `compute_propensity_score(X, w, learner, ...)` | Estimate propensity scores with a given learner. |
| `propensity_glmnet(X, w, ...)` | Lasso/ElasticNet propensity (via `glmnet`). |
| `propensity_glm(X, w, ...)` | Logistic regression propensity. |
| `propensity_model(type, ...)` | Factory: returns a propensity model object by type string. |
| `propensity_model_logistic_regression(clip_bounds, ...)` | Logistic propensity model constructor. |
| `propensity_model_elastic_net(clip_bounds, ...)` | ElasticNet propensity model constructor. |
| `propensity_model_gradient_boosted(clip_bounds, ...)` | Gradient-boosted propensity model constructor. |
| `nearest_neighbor_match(X, w, n_matches, ...)` | Nearest-neighbour matching (1:k). |
| `create_table_one(data, treatment, vars, ...)` | Descriptive balance table (Table 1). |
| `smd(x, treatment)` | Standardised mean difference. |
| `match_optimizer(data, ...)` | Optimise matching via `MatchIt`. |
| `match_by_group(data, group, ...)` | Subgroup-stratified matching. |

---

## 6. Meta-Learners

All meta-learner objects share the generic S3 API:
`fit(model, X, w, y, ...)`, `predict(model, X, ...)`,
`estimate_ate(model, X, w, y, ...)`, `fit_predict(model, X, w, y, ...)`.

| Constructor | Description |
|---|---|
| `SLearner(learner, ...)` | Single-model learner — fits one model on `(X, w)`. |
| `TLearner(learner, ...)` | Two-model learner — separate outcome models per arm. |
| `XLearner(learner, ...)` | X-learner (Künzel et al., 2019) with propensity weighting. |
| `DomainAdaptationLearner(learner, ...)` | Domain-adaptation weighting for CATE. |
| `RLearner(learner, ...)` | R-learner (Robinson 1988; Nie & Wager 2021). |
| `r_learner_continuous(X, y, w, ...)` | Functional API for the R-learner on continuous treatment. |
| `DRLearner(learner, ...)` | Doubly-robust learner with outcome + propensity models. |
| `ForestDRLearner(...)` | DR-learner with random-forest nuisance models. |
| `LinearDRLearner(...)` | DR-learner with linear final model; supports `coef()`, `intercept()`. |
| `SparseLinearDRLearner(...)` | Lasso-regularised linear DR-learner. |
| `TMLELearner(learner, ...)` | TMLE-based CATE estimator. |
| `XGBDRLearner(...)` | DR-learner with XGBoost nuisance + final models. |

---

## 7. Double Machine Learning (DML)

DML learners implement `fit()`, `predict()`, `estimate_ate()`, and where
applicable `coef()`, `intercept()`, `effect()`.

| Constructor | Description |
|---|---|
| `DMLearner(learner_y, learner_w, ...)` | Generic DML wrapper (native R). |
| `LinearDML(model_y, model_t, ...)` | Linear final-stage DML (EconML-compatible). |
| `SparseLinearDML(model_y, model_t, ...)` | Lasso final-stage DML. |
| `KernelDML(model_y, model_t, ...)` | Kernel regression final-stage DML. |
| `NonParamDML(model_y, model_t, ...)` | Non-parametric final-stage DML (causal forest). |
| `CausalForestDML(model_y, model_t, ...)` | DML with GRF causal forest second stage. |
| `DynamicDMLearner(...)` | Panel/dynamic DML for time-series data; `effect()` method. |

---

## 8. DoubleML Package Bridges

These functions wrap the **DoubleML** R package for partial-linear, IV, and
DiD models, and can be used standalone or inside `DMLearner` pipelines.

| Function | Description |
|---|---|
| `doubleml_plr(data, ml_l, ml_m, ...)` | Fit a partially linear regression (PLR) via DoubleML. |
| `doubleml_plr_fit_data(data, ml_l, ml_m, ...)` | PLR fit from a `DoubleMLData` object. |
| `doubleml_plr_tune_data(data, ml_l, ml_m, ...)` | PLR with tuned hyperparameters. |
| `doubleml_pliv(data, ml_l, ml_m, ml_r, ...)` | Partially linear IV regression. |
| `doubleml_data_from_data_frame(df, y_col, d_col, ...)` | Build `DoubleMLData` from a data frame. |
| `doubleml_data_from_matrix(X, y, d, ...)` | Build `DoubleMLData` from matrices. |
| `doubleml_make_plr_CCDDHNR2018(n, ...)` | Simulate the CCDDHNR 2018 PLR benchmark dataset. |
| `doubleml_make_pliv_CHS2015(n, ...)` | Simulate the CHS 2015 PLIV benchmark dataset. |
| `doubleml_fetch_401k(...)` | Load the 401k savings data (Poterba et al.). |
| `doubleml_fetch_bonus(...)` | Load the Bonus experiment data (Bilias, 2000). |
| `doubleml_did_linear(data, ...)` | DiD-style DML with linear nuisance models. |
| `doubleml_did_rf(data, ...)` | DiD-style DML with random forest nuisance. |
| `doubleml_did_xgboost(data, ...)` | DiD-style DML with XGBoost nuisance. |
| `doubleml_did_eval_preds(fit, data, ...)` | Evaluate predictions from a DiD-DML fit. |
| `doubleml_did_eval_linear(data, ...)` | Evaluate linear DiD-DML model. |
| `doubleml_did_eval_rf(data, ...)` | Evaluate random-forest DiD-DML model. |
| `doubleml_did_eval_xgboost(data, ...)` | Evaluate XGBoost DiD-DML model. |

---

## 9. Instrumental Variable (IV) Methods

| Function | Description |
|---|---|
| `OrthoIVLearner(...)` | Orthogonal IV learner (generalised Wald / LATE). |
| `DMLIVLearner(...)` | DML-based IV learner. |
| `NonParamDMLIVLearner(...)` | Non-parametric DML IV learner. |
| `iv_2sls(y, D, Z, X, ...)` | Two-stage least squares (2SLS). |
| `late_iv(y, D, Z, X, ...)` | Local average treatment effect via IV. |
| `driv_learner(X, y, w, Z, ...)` | Deep DRIV learner combining IV with doubly-robust scores. |
| `BoostedIVForest` | R6 class: XGBoost-backed instrumental variable forest. |
| `boosted_iv_forest(X, y, w, Z, ...)` | Functional constructor for `BoostedIVForest`. |
| `save_boosted_iv(model, path)` | Serialise a `BoostedIVForest` to disk. |
| `load_boosted_iv(path)` | Load a saved `BoostedIVForest`. |
| `register_boosted_iv_parsnip()` | Register `BoostedIVForest` as a `parsnip` engine. |

---

## 10. Causal Forest / GRF Integration

These functions wrap the **grf** package and add convenience helpers.

| Function | Description |
|---|---|
| `causal_forest(X, Y, W, ...)` | Generalised random forest causal forest. |
| `causal_tree(X, Y, W, ...)` | Single causal tree (honest splitting). |
| `average_treatment_effect(model, ...)` | ATE from a causal forest. |
| `grf_causal_forest(X, Y, W, ...)` | Direct interface to `grf::causal_forest`. |
| `grf_average_treatment_effect(forest, ...)` | ATE/ATT via `grf`. |
| `best_linear_projection(forest, ...)` | Best linear projection of CATE on covariates. |
| `test_calibration(forest, ...)` | Calibration test for a causal forest. |
| `causal_survival_forest(X, Y, W, D, ...)` | Causal survival forest for censored outcomes. |
| `instrumental_forest(X, Y, W, Z, ...)` | GRF instrumental variable forest. |
| `multi_arm_causal_forest(X, Y, W, ...)` | Multi-arm causal forest. |
| `rank_average_treatment_effect(forest, priorities, ...)` | RATE statistic (targeting operator). |
| `rank_average_treatment_effect.fit(forest, ...)` | Compute RATE from fitted forest. |
| `variable_importance(forest, ...)` | Variable importance from a GRF forest. |
| `get_forest_weights(forest, newdata, ...)` | Forest kernel weights. |
| `get_leaf_node(forest, newdata)` | Leaf node IDs for new observations. |
| `get_tree(forest, index)` | Extract a single tree. |
| `split_frequencies(forest, max_depth)` | Split frequency table. |
| `causal_forest_confounder(X, Y, W, ...)` | Causal forest with confounder adjustment. |
| `predict.causal_forest(object, newdata, ...)` | Predict CATE from a `causal_forest` object. |
| `print.causal_forest(x, ...)` | Print method. |
| `summary.causal_forest(object, ...)` | Summary method. |
| `plot.grf_tree(x, ...)` | Plot a single GRF tree. |
| `plot.rank_average_treatment_effect(x, ...)` | Plot RATE/AUTOC curve. |

---

## 11. Causal XGBoost (`CXGBoost`)

`CXGBoost` is an **R6 class** implementing a two-head DragonNet-style
masked-MSE outcome model with a `ranger`-based propensity estimator.

```r
mod <- CXGBoost$new(nrounds = 100, n_trees = 200, ...)
mod$fit(X, w, y)
tau <- mod$predict(X)
mod$summary()
```

| Symbol | Description |
|---|---|
| `CXGBoost` | R6 class constructor. Fields: `nrounds`, `n_trees`, `eta`, `max_depth`. Methods: `fit()`, `predict()`, `summary()`. |
| `save_cxgboost(model, path)` | Serialise a `CXGBoost` model. |
| `load_cxgboost(path)` | Load a saved `CXGBoost` model. |
| `load_cxgboost_extensions()` | Register optional XGBoost extensions. |
| `run_cxgboost_example()` | Run the built-in `CXGBoost` demo. |
| `run_example()` | Alias / general demo runner. |

---

## 12. Multi-Arm Causal Boosting (`MultiArmCausalBoost`)

`MultiArmCausalBoost` extends CXGBoost to K ≥ 2 treatment arms with
separate outcome models per arm and a multiclass propensity model.

```r
mod <- MultiArmCausalBoost$new(nrounds = 50)
mod$fit(X, y, W)          # W is a factor with K levels
preds <- mod$predict(X)   # list with $contrasts named by arm
```

| Symbol | Description |
|---|---|
| `MultiArmCausalBoost` | R6 class. Methods: `fit(X, y, W)`, `predict(X)`. |
| `multi_arm_PEHE(tau_hat, tau_true)` | Multi-arm PEHE metric. |
| `multi_arm_ATE(tau_hat)` | Multi-arm ATE metric. |
| `save_multi_arm_causal_boost(model, path)` | Serialise model. |
| `load_multi_arm_causal_boost(path)` | Load model. |
| `run_multi_arm_causal_boost_example()` | Built-in demo. |

---

## 13. XGBoost Regressor Variants (EconML-style)

These functional constructors return model objects compatible with the
`fit` / `predict` generics.

| Function | Description |
|---|---|
| `LRSRegressor(...)` | Lasso/Ridge/Stacking regressor. |
| `XGBTRegressor(...)` | XGBoost treatment regressor. |
| `XGBSRegressor(...)` | XGBoost stacking regressor. |
| `XGBXRegressor(...)` | XGBoost X-learner regressor. |
| `XGBRRegressor(...)` | XGBoost R-learner regressor. |
| `XGBDRRegressor(...)` | XGBoost DR-learner regressor. |
| `XGBDRLearner(...)` | Full DR-learner built on XGBoost regressors. |

---

## 14. Uplift Trees & Forests

### Uplift Random Forests

| Function | Description |
|---|---|
| `uplift_rf_kl(X, y, w, ...)` | Uplift RF with KL-divergence split criterion. |
| `uplift_rf_ed(X, y, w, ...)` | Uplift RF with Euclidean distance criterion. |
| `uplift_rf_chi(X, y, w, ...)` | Uplift RF with chi-squared criterion. |
| `uplift_rf_cts(X, y, w, ...)` | Uplift RF with CTS (contextual treatment selection). |
| `uplift_rf_multi(X, y, w, ...)` | Multi-treatment uplift forest. |

### Uplift Trees

| Function | Description |
|---|---|
| `uplift_tree_ddp(X, y, w, ...)` | Uplift tree with DDP (delta-delta-p) splitting. |
| `uplift_tree_iddp(X, y, w, ...)` | Uplift tree with interaction DDP splitting. |
| `interaction_tree(X, y, w, ...)` | Interaction tree (Athey & Imbens 2016). |
| `causal_inference_tree(X, y, w, ...)` | Causal inference tree (honest). |

### Tree Utilities

| Function | Description |
|---|---|
| `uplift_tree_string(tree)` | Pretty-print an uplift tree as a string. |
| `uplift_tree_plot(tree, ...)` | Plot an uplift tree. |
| `uplift_tree_to_rpart(tree)` | Convert to an `rpart`-compatible object. |
| `uplift_forest_plot(forest, ...)` | Plot variable importance for an uplift forest. |

---

## 15. Policy Learning

| Function | Description |
|---|---|
| `policy_learner(...)` | Generic doubly-robust policy learner. |
| `DRPolicyTree(...)` | Doubly-robust policy tree. |
| `DRPolicyForest(...)` | Doubly-robust policy forest. |
| `predict_proba(model, X, ...)` | Predicted treatment probabilities. |
| `predict_value(model, X, ...)` | Predicted policy value. |
| `feature_importances(model, ...)` | Feature importances (tree/forest). |
| `policy_feature_names(model)` | Feature names used by the policy model. |

---

## 16. CATE Interpretation

| Function | Description |
|---|---|
| `SingleTreeCateInterpreter(...)` | Fit a single tree to approximate a CATE model. |
| `SingleTreePolicyInterpreter(...)` | Fit a single policy-tree approximation. |
| `interpret(model, X, ...)` | Extract interpretable rules from an interpreter object. |
| `treat(model, X, ...)` | Recommended treatment assignment from a `SingleTreePolicyInterpreter`. |
| `predict.SingleTreeCateInterpreter(object, X, ...)` | Predict CATE from interpreter. |
| `predict.SingleTreePolicyInterpreter(object, X, ...)` | Predict treatment from interpreter. |
| `explain_cate(object, X, bg, ...)` | SHAP-based CATE explanation via `kernelshap` / `shapviz`. |

---

## 17. Generic S3 API

The following generics are defined in `RCausalML` and dispatch to model-specific methods.

| Generic | Key Methods Available |
|---|---|
| `fit(model, X, w, y, ...)` | SLearner, TLearner, XLearner, DomainAdaptationLearner, RLearner, DRLearner, DMLearner, DynamicDMLearner, OrthoIVLearner, DMLIVLearner, NonParamDMLIVLearner, TMLELearner, AutomatedMLModel, policy_learner, DRPolicyTree, DRPolicyForest |
| `predict(model, X, ...)` | All learners above + causal_forest, uplift_rf/tree variants, driv_learner, cevae, dragonnet, tarnet, cfrnet, ganite, causalGAN, causal_egm, causal_discrepancy_vae, neural_granger_ml, dscm, deep_scm, deci_model, dynotears, attn_causal_model, rnn_causal_model, gnn_causal_model, counterfactual_model |
| `estimate_ate(model, X, w, y, ...)` | SLearner, TLearner, XLearner, DomainAdaptationLearner, RLearner, DRLearner, DMLearner, TMLELearner |
| `fit_predict(model, X, w, y, ...)` | SLearner, TLearner, XLearner, DomainAdaptationLearner, RLearner, DRLearner, DMLearner |
| `coef(model, ...)` | LinearDML, LinearDRLearner, SparseLinearDML, SparseLinearDRLearner |
| `intercept(model, ...)` | LinearDML, LinearDRLearner, SparseLinearDML, SparseLinearDRLearner |
| `effect(model, ...)` | DynamicDMLearner |
| `predict_best_treatment(model, X, ...)` | counterfactual_value_estimator |

---

## 18. Counterfactual Value & Unit Selection

| Function | Description |
|---|---|
| `counterfactual_value_estimator(X, y, w, ...)` | Estimate value of counterfactual treatments. |
| `counterfactual_unit_selection(X, y, w, ...)` | Select units that benefit most from treatment. |
| `predict_best_treatment(model, X, ...)` | Predict the best treatment arm per unit. |

---

## 19. Evaluation Metrics

| Function | Description |
|---|---|
| `PEHE(tau_hat, tau_true)` | Precision in estimation of heterogeneous effects. |
| `ATE(tau_hat)` | Average treatment effect from estimated ITE vector. |
| `ATE_error(tau_hat, tau_true)` | ATE estimation error. |
| `regression_metrics(y_true, y_pred, ...)` | RMSE, MAE, R² for continuous outcomes. |
| `classification_metrics(y_true, y_pred, ...)` | Accuracy, AUC, F1 for binary outcomes. |
| `qini_curve(y, w, score, ...)` | Compute Qini curve values. |
| `qini_score_vec(y, w, score)` | Qini coefficient (vector input). |
| `qini_score(y, w, score, ...)` | Qini coefficient with optional normalisation. |
| `plot_qini(qini, ...)` | Plot the Qini curve. |
| `gain_curve(y, w, score, ...)` | Cumulative gain curve. |
| `plot_gain(gain, ...)` | Plot the gain curve. |
| `get_cumlift(y, w, score, ...)` | Cumulative lift values. |
| `get_cumgain(y, w, score, ...)` | Cumulative gain values. |
| `multi_arm_PEHE(tau_hat, tau_true)` | Multi-arm PEHE. |
| `multi_arm_ATE(tau_hat)` | Multi-arm ATE. |
| `count_accuracy(B_true, B_est)` | Graph-recovery accuracy metrics (TP, FP, FN, SHD). |

---

## 20. Sensitivity Analysis

| Function | Description |
|---|---|
| `sensitivity(model, ...)` | Rosenbaum-style sensitivity analysis. |
| `one_sided(model, gamma, ...)` | One-sided sensitivity bound. |
| `alignment(model, gamma, ...)` | Alignment sensitivity bound. |
| `one_sided_att(model, gamma, ...)` | ATT one-sided bound. |
| `alignment_att(model, gamma, ...)` | ATT alignment bound. |
| `causalsens_selection_bias(y, w, X, ...)` | Selection-bias sensitivity (causalsens). |
| `sensitivity_analysis(model, ...)` | Full sensitivity analysis suite. |
| `plot_sensitivity(result, ...)` | Plot sensitivity analysis results. |
| `partial_rsqs_confounding(model, ...)` | Partial R² bounds for unmeasured confounding. |

---

## 21. Feature Filters

| Function | Description |
|---|---|
| `filter_F(X, y, w, ...)` | F-statistic filter for treatment-correlated features. |
| `filter_LR(X, y, w, ...)` | Likelihood-ratio filter. |
| `filter_D(X, y, w, ...)` | Difference-in-means filter. |
| `get_importance(model, ...)` | Variable importance from a fitted model. |

---

## 22. Automated ML (AzureML / EconML style)

| Function | Description |
|---|---|
| `set_automated_ml_workspace(...)` | Configure an AzureML workspace. Alias: `setAutomatedMLWorkspace`. |
| `EconAutoMLConfig(...)` | Build an AutoML configuration object. |
| `AutomatedMLModel(...)` | AutoML model wrapper (fit/predict/predict_proba). |
| `add_automated_ml(pipeline, ...)` | Add AutoML step to a pipeline. Alias: `addAutomatedML`. |

---

## 23. Causal DAG Discovery (`causalStructureML`)

`causalStructureML` is the unified interface to NOTEARS, DAGMA, GraN-DAG,
and DAG-GNN structure-learning algorithms.

```r
result <- causalStructureML(X, method = "notears_linear")
causal_structure_ml_model_descriptions()   # list all supported methods
```

### Main Entry Points

| Function | Description |
|---|---|
| `causalStructureML(X, method, ...)` | Fit a causal structure model. Returns adjacency matrix + metadata. |
| `causal_structure_ml_model_descriptions()` | Print descriptions of all supported methods. |

### NOTEARS (`R/notears.R`)

| Function | Description |
|---|---|
| `NOTEARS(X, lambda, ...)` | Main NOTEARS functional wrapper. |
| `notears_linear(X, lambda, ...)` | NOTEARS with linear SEMs (Zheng et al., 2018). |
| `notears_nonlinear(X, lambda, ...)` | NOTEARS with nonlinear MLP/Sobolev SEMs. |
| `is_dag(W)` | Check whether a matrix is a DAG. |
| `demo_linear(n, d, ...)` | Demo: linear NOTEARS. |
| `demo_nonlinear_mlp(n, d, ...)` | Demo: nonlinear NOTEARS (MLP). |
| `demo_nonlinear_sobolev(n, d, ...)` | Demo: nonlinear NOTEARS (Sobolev). |

### DAGMA (`R/notears.R` / `R/causalStructureML.R`)

| Function | Description |
|---|---|
| `dagmaLinear(X, lambda, ...)` | DAGMA with linear model (Bello et al., 2022). |
| `DagmaMLP(X, lambda, ...)` | DAGMA with MLP model. |
| `dagma(X, ...)` | Convenience alias for DAGMA. |

### CASTLE (`R/causalStructureML.R`)

| Function | Description |
|---|---|
| `castle(X, ...)` | CASTLE structure learner (Kyono et al., 2020). |
| `predict.castle(object, newdata, ...)` | Predict from a CASTLE model. |
| `summary.castle(object, ...)` | Summary of CASTLE fit. |
| `plot.castle(x, ...)` | Plot estimated DAG from CASTLE. |

### DAG Utilities

| Function | Description |
|---|---|
| `evaluate_graph_recovery(B_true, B_est)` | SHD, TPR, FPR, precision, recall. |
| `plot_scm_dag(B, labels, ...)` | Plot a structural causal model DAG. |

---

## 24. Graph Layout & Drawing

These functions mirror the `networkx` drawing API for visualising causal graphs.

| Function | Description |
|---|---|
| `draw_network(graph, layout, ...)` | Draw a causal graph using `igraph` / `ggraph`. |
| `draw_networkx_nodes(graph, pos, ...)` | Draw graph nodes. |
| `draw_networkx_edges(graph, pos, ...)` | Draw graph edges. |
| `draw_networkx_labels(graph, pos, ...)` | Draw node labels. |

### Layout Algorithms

| Function | Description |
|---|---|
| `random_layout(graph, ...)` | Random positions. |
| `circular_layout(graph, ...)` | Nodes on a circle. |
| `shell_layout(graph, ...)` | Concentric shells. |
| `bipartite_layout(graph, ...)` | Bipartite two-column layout. |
| `multipartite_layout(graph, ...)` | Multi-layer layout. |
| `spring_layout(graph, ...)` | Fruchterman-Reingold force-directed (alias). |
| `fruchterman_reingold_layout(graph, ...)` | Force-directed (Fruchterman & Reingold). |
| `kamada_kawai_layout(graph, ...)` | Kamada-Kawai energy layout. |
| `spectral_layout(graph, ...)` | Spectral decomposition layout. |
| `spiral_layout(graph, ...)` | Archimedean spiral. |
| `bfs_layout(graph, start, ...)` | Breadth-first-search tree layout. |
| `rescale_layout(pos, scale)` | Rescale a layout matrix. |
| `rescale_layout_dict(pos, scale)` | Rescale a named-list layout. |

---

## 25. Deep Causal Networks (require `torch`)

### Potential-Outcomes / Treatment-Effect Networks (`R/causalDeepNet.R`)

| Function | Description |
|---|---|
| `cevae(X, y, w, ...)` | CEVAE (Louizos et al., 2017) — VAE for causal effect estimation with latent confounders. |
| `dragonnet(X, y, w, ...)` | DragonNet (Shi et al., 2019) — shared representation with treatment & outcome heads. |
| `tarnet(X, y, w, ...)` | TARNet — treatment-agnostic representation network. |
| `cfrnet(X, y, w, ...)` | CFRNet — counterfactual regression network with IPM regularisation. |
| `ganite(X, y, w, ...)` | GANITE (Yoon et al., 2018) — GAN-based ITE estimation. |
| `causalGAN(X, y, w, ...)` | CausalGAN — structural-interventional GAN for causal modelling. |

### VAE / Latent-Factor Models (`R/causalDeepNet.R`)

| Function | Description |
|---|---|
| `CausalVAE(obs_dim, latent_dim, ...)` | Causal VAE with disentangled latent structure. |
| `CausalVAE_ATE(model, X, w)` | Estimate ATE from a fitted CausalVAE. |
| `train_causalvae(model, X, y, w, ...)` | Training loop for `CausalVAE`. |
| `causal_vae(X, y, w, ...)` | Functional interface to CausalVAE. |
| `causal_vae_opt(X, y, w, ...)` | CausalVAE with hyperparameter optimisation. |
| `estimate_ate_causalvae_ate(model, X, w)` | ATE estimation utility. |
| `tune_hyperparameters(model, X, y, w, ...)` | Hyperparameter tuning wrapper. |
| `print_optimization_summary(result)` | Print tuning summary. |
| `loss_function(model, X, y, w, ...)` | Generic loss computation. |
| `CausalEGM(input_dim, ...)` | Causal EGM — disentangled latent causal-factor model. |
| `causal_egm(X, y, w, ...)` | Functional interface to CausalEGM. |
| `CausalDiscrepancyVAE(obs_dim, ...)` | Discrepancy VAE with latent MMD balancing. |
| `causal_discrepancy_vae(X, y, w, ...)` | Functional interface. |
| `dscm(X, y, w, ...)` | Deep structural causal model (DSCM). |
| `ivae(X, y, s, ...)` | iVAE — identifiable VAE with auxiliary variable. |

### Temporal Causal VAE (`R/tempoCausalVAE.R`)

| Function | Description |
|---|---|
| `TemporalCausalVAE(obs_dim, latent_dim, hidden_dim, ...)` | GRU encoder/decoder with learned causal adjacency and NOTEARS DAG penalty. |
| `temporal_causal_loss(out, x_seq, ...)` | ELBO + DAG-penalty loss for `TemporalCausalVAE`. |

### Interventional CRL (`R/interventionalCRL.R`)

| Function | Description |
|---|---|
| `InterventionalCRL` | R6 class: environment-conditioned VAE with binary fingerprint inputs and interventional ELBO loss. |
| `interventional_elbo_loss(model, x, e, ...)` | Compute interventional ELBO loss. |

### Temporal Causal Discovery (`R/temporalCausaDiscovery.R`)

| Function | Description |
|---|---|
| `TCDF_find_causes(data, ...)` | ADDSTCN-based temporal causal discovery (Nauta et al., 2019). |
| `neural_granger_ml(X, ...)` | Neural Granger causality (cMLP / cLSTM / NRI). |
| `neuralGrangerML(...)` | CamelCase alias for `neural_granger_ml`. |

### Neural Granger Models (inside `R/causalDeepNet.R`)

| Function | Description |
|---|---|
| `predict.neural_granger_ml(object, X, ...)` | Predict from a neural Granger model. |

---

## 26. Structural Causal Models (SCMs) with Deep Components

### DeepSCM (`R/causalDeepNet.R`)

| Function | Description |
|---|---|
| `deep_scm(dag, obs_dim, ...)` | Fixed-graph SCM with variational noise encoders. Alias: `deepSCM`. |
| `predict.deep_scm(object, X, ...)` | Reconstruct observations. |
| `intervene_deep_scm(model, do_vars, ...)` | Apply hard interventions `do(·)` and sample counterfactuals. |

### DECI (`R/causalDeepNet.R`)

| Function | Description |
|---|---|
| `deci_model(obs_dim, ...)` | DECI: jointly learns graph and structural equations with NOTEARS penalty. Alias: `deciModel`. |
| `predict.deci_model(object, X, ...)` | Predict from DECI. |
| `ate_deci(model, X, do_t, ...)` | ATE under `do(T)` from DECI. |

### DynoTEARS (`R/causalDeepNet.R`)

| Function | Description |
|---|---|
| `dynotears(X_lag, obs_dim, ...)` | Lagged causal discovery with augmented-Lagrangian DAG constraint. Alias: `dynoTEARS`. |
| `predict.dynotears(object, X, ...)` | Predict from DynoTEARS. |

---

## 27. Attention / Transformer Causal Models

All require `torch`.

| Function | Alias | Description |
|---|---|---|
| `attn_causal_model(input_dim, ...)` | `attnCausalModel` | Generic attention-based causal model factory. |
| `tcdf_model(input_dim, ...)` | `TCDFModel` | TCDF — dilated causal convolutions + variable-importance attention (Nauta et al., 2019). |
| `causal_transformer_model(input_dim, ...)` | `CausalTransformerModel` | Transformer with autoregressive masking and inter-variable cross-attention. |
| `tft_model(input_dim, ...)` | `TFTModel` | Temporal Fusion Transformer (Lim et al., 2021): variable selection, LSTM encoder, multi-head temporal attention. |
| `predict.attn_causal_model(object, X, ...)` | — | Predict from any attention causal model. |
| `causal_matrix_attn(model, X, ...)` | — | Extract causal attention matrix. |

---

## 28. RNN / LSTM Causal Models

All require `torch`.

| Function | Alias | Description |
|---|---|---|
| `rnn_causal_model(input_dim, ...)` | `rnnCausalModel` | Generic RNN causal model factory. |
| `causal_lstm_model(input_dim, ...)` | `CausalLSTMModel` | CausalLSTM with learnable sparse adjacency mask gating per-variable LSTM inputs. |
| `retain_model(input_dim, ...)` | `RETAINModel` | RETAIN (Choi et al., 2016) — reverse-time dual-channel GRU attention (temporal α + variable β). |
| `intervention_rnn_model(input_dim, ...)` | `InterventionRNNModel` | Intervention-aware RNN with GRU regime detector and explicit intervention-indicator channel. |
| `predict.rnn_causal_model(object, X, ...)` | — | Predict from any RNN causal model. |
| `causal_matrix_rnn(model, X, ...)` | — | Extract learned sparse causal adjacency matrix. |

---

## 29. Graph Neural Network (GNN) Causal Models

All require `torch`.

| Function | Alias | Description |
|---|---|---|
| `gnn_causal_model(input_dim, ...)` | `gnnCausalModel` / `GNNCausalModel` | Generic GNN causal model factory. |
| `gvar_model(input_dim, ...)` | `GVARModel` | GVAR — lag-specific soft adjacency matrices with L1 + NOTEARS penalties. |
| `causal_gnn_model(input_dim, ...)` | `CausalGNNModel` | CD-GNN — GRU temporal encoder + bilinear graph learner + stacked edge-conditioned GNN layers. |
| `cuts_model(input_dim, ...)` | `CUTSModel` | CUTS+ — variational Bernoulli graph posterior + imputation network for missing data. |
| `predict.gnn_causal_model(object, X, ...)` | — | Predict from any GNN causal model. |
| `causal_matrix_gnn(model, X, ...)` | — | Extract learned causal adjacency matrix. |

---

## 30. Counterfactual / Potential-Outcomes Deep Models

All require `torch`.

| Function | Alias | Description |
|---|---|---|
| `counterfactual_model(type, input_dim, ...)` | `CounterfactualModel` | Factory: create DeepSynth, CRN, or G-Net model. |
| `deep_synth_model(input_dim, ...)` | `DeepSynthModel` | Neural Synthetic Control — GRU + scaled dot-product attention over donor variables; ITE from factual/counterfactual heads. |
| `crn_model(input_dim, ...)` | `CRNModel` | Counterfactual Recurrent Network — GRU + adversarial treatment-balancing discriminator. |
| `gnet_model(input_dim, ...)` | `GNetModel` | G-Net (Deep G-Computation) — GRU + covariate transition + outcome heads; sequential substitution for counterfactuals. |
| `predict.counterfactual_model(object, X, ...)` | — | Generate counterfactual predictions. |
| `ate_counterfactual(model, X, w, ...)` | — | ATE from a counterfactual model. |
| `ite_counterfactual(model, X, w, ...)` | — | ITE vector from a counterfactual model. |

---

## 31. Causaldata Integration

| Function | Description |
|---|---|
| `list_causaldata_datasets()` | List all datasets available in the `causaldata` package. |
| `load_causaldata(name, ...)` | Load a named `causaldata` dataset as a data frame. |

---

## 32. Utility Functions

| Function | Description |
|---|---|
| `set_random_seed(seed)` | Set R and (if available) torch random seeds simultaneously. |
| `smd(x, treatment)` | Standardised mean difference (also in propensity section). |
| `propensity_model(...)` | Propensity model factory (see §5). |

---

## 33. Example Workflows

### Meta-Learner Comparison

```r
library(RCausalML)
set.seed(1)
d <- synthetic_data(mode = 2, n = 500, p = 6)

models <- list(
  S = SLearner(learner = "ranger"),
  T = TLearner(learner = "ranger"),
  X = XLearner(learner = "ranger"),
  R = RLearner(learner = "ranger")
)

for (nm in names(models)) {
  models[[nm]] <- fit(models[[nm]], d$X, d$w, d$y)
  cat(nm, "ATE:", estimate_ate(models[[nm]], d$X, d$w, d$y), "\n")
}
```

### DML with Linear Final Stage

```r
library(RCausalML)
d <- synthetic_data(mode = 1, n = 600, p = 5)
m <- LinearDML()
m <- fit(m, d$X, d$w, d$y)
cat("CATE coefs:", coef(m), "\n")
```

### Causal Forest + SHAP

```r
library(RCausalML)
library(shapviz)
d   <- synthetic_data(mode = 3, n = 800, p = 8)
cf  <- causal_forest(d$X, d$y, d$w)
shp <- explain_cate(cf, d$X, bg = d$X[1:50, ])
sv_importance(shp)
```

### DoubleML Bridge

```r
library(RCausalML)
data <- doubleml_fetch_401k()
fit  <- doubleml_plr(data, ml_l = "ranger", ml_m = "ranger")
print(fit)
```

### Causal XGBoost

```r
library(RCausalML)
library(xgboost)
set.seed(1)
n <- 300; p <- 5
X  <- matrix(rnorm(n * p), n, p)
w  <- rbinom(n, 1, 0.5)
y  <- 2 * w + X[, 1] + rnorm(n)
mod <- CXGBoost$new(nrounds = 50, n_trees = 100)
mod$fit(X, w, y)
cat("ATE:", mean(mod$predict(X)), "\n")
mod$summary()
```

### Multi-Arm Causal Boosting

```r
library(RCausalML)
set.seed(2)
n <- 200; p <- 4
X <- matrix(rnorm(n * p), n, p)
W <- factor(sample(c("control","trt1","trt2"), n, replace = TRUE))
y <- ifelse(W == "trt1", 1, ifelse(W == "trt2", 2, 0)) + rnorm(n)
mod <- MultiArmCausalBoost$new(nrounds = 30)
mod$fit(X, y, W)
preds <- mod$predict(X)
cat("Contrasts:", names(preds$contrasts), "\n")
```

### Temporal Causal VAE (requires torch)

```r
if (requireNamespace("torch", quietly = TRUE)) {
  library(torch)
  library(RCausalML)
  model <- TemporalCausalVAE(obs_dim = 3L, latent_dim = 3L, hidden_dim = 64L)
  x_seq <- torch_randn(8L, 10L, 3L)
  out   <- model(x_seq)
  cat("Latent shape:", paste(out$z$shape, collapse = "x"), "\n")
}
```

### NOTEARS Structure Learning

```r
library(RCausalML)
set.seed(42)
dat <- generate_data(n = 200, sem_type = "linear-gauss", d = 5, graph_type = "ER")
res <- causalStructureML(dat$X, method = "notears_linear")
evaluate_graph_recovery(dat$B, res$W_est)
plot_scm_dag(res$W_est)
```

---

## 34. Common Issues & Fixes

### Cross-platform

| Issue | Fix |
|---|---|
| **CRAN mirror unreachable** | `options(repos = c(CRAN = "https://cloud.r-project.org"))` |
| **`xgboost` not found** | `install.packages("xgboost")` |
| **`torch` not found for deep nets** | `install.packages("torch"); torch::install_torch()` |
| **CPU/CUDA warnings** | CPU execution is expected when CUDA is unavailable — most algorithms still run on CPU. |
| **`grf` not found** | `install.packages("grf")` |
| **`DoubleML` not found** | `install.packages("DoubleML")` |
| **`MatchIt` not found** | `install.packages("MatchIt")` |
| **`shapviz`/`kernelshap` not found** | `install.packages(c("kernelshap", "shapviz"))` |
| **`htetree` not found** | `install.packages("htetree")` or `remotes::install_github("xiaomanluo/htetree")` |
| **`coro` not found** | `install.packages("coro")` |

### Ubuntu-specific

| Issue | Fix |
|---|---|
| **Permission denied to system R library** | Install to user library: `R CMD INSTALL -l ~/.Rlibrary RCausalML_0.3.0.tar.gz` |
| **`igraph` fails to compile** | `sudo apt install -y libxml2-dev` then `install.packages("igraph")` |
| **`ggraph` fails** | `sudo apt install -y libfontconfig1-dev libharfbuzz-dev libfribidi-dev` |
| **`glmnet` / `Matrix` LAPACK errors** | `sudo apt install -y liblapack-dev libblas-dev libgfortran5` |
| **`curl` / `httr` SSL errors** | `sudo apt install -y libssl-dev libcurl4-openssl-dev` |
| **`reticulate` Python not found** | `sudo apt install -y python3 python3-dev python3-pip`; set `RETICULATE_PYTHON` in `.Renviron` |
| **`torch::install_torch()` download fails** | Set `options(timeout = 300)` before calling `install_torch()` |
| **R version too old** | Add CRAN PPA (see §1 Ubuntu Prerequisites) and `sudo apt upgrade r-base` |
| **`cmake` not found (GraN-DAG / DAG-GNN)** | `sudo apt install -y cmake` |

### Windows-specific

| Issue | Fix |
|---|---|
| **`make` not found / Rtools not on PATH** | In R: `writeLines('PATH="${RTOOLS44_HOME}\\usr\\bin;${PATH}"', "~/.Renviron")`; restart R |
| **DLL load failed for `torch`** | Install VC++ Redistributable 2019+ (<https://aka.ms/vs/17/release/vc_redist.x64.exe>); then `torch::install_torch()` |
| **`torch::install_torch()` stalls** | Set `options(timeout = 600)`; ensure antivirus allows LibTorch download |
| **Binary package not available for R x.y** | `options(repos = c(CRAN = "https://packagemanager.posit.co/cran/latest"))` |
| **`igraph` compile errors (source)** | `install.packages("igraph", type = "binary")` |
| **Admin rights needed for system library** | `install.packages(..., lib = Sys.getenv("R_LIBS_USER"))` |
| **`reticulate` Python not found** | `reticulate::install_miniconda()` once; then `reticulate::use_condaenv("r-rcausalml")` |
| **Long path errors on Windows** | Enable long paths as admin: `reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1` |
| **`grf` compile errors** | `install.packages("grf", type = "binary")` |

---

## 35. Where to Look Next

- Package documentation: `README.md`
- Change log: `NEWS.md`
- In-R help: `?SLearner`, `?DMLearner`, `?causalStructureML`, `?CXGBoost`, etc.
- Test scripts: `tests/`
- Python reference: <https://causalml.readthedocs.io/>
- Python source: <https://github.com/uber/causalml>
