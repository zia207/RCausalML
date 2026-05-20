# RCausalML 0.3.0

### Cross-platform install support

- **`configure.win`**: Added a Windows-specific configure script (POSIX sh via
  Rtools) that mirrors the Unix `configure` script.  On interactive installs it
  reports missing/outdatable R package dependencies and installs them using
  `type = "binary"` (preferred on Windows to avoid source compilation).
  Post-install guidance for `torch::install_torch()` and
  `reticulate::install_miniconda()` is printed automatically when those
  packages are in the target set.

- **Ubuntu prerequisites**: The help file (`HELP_RCausalML_0.3.0.md`) now
  documents the full set of system-level `apt` packages required before
  installing R dependencies with native code (`libssl-dev`,
  `libcurl4-openssl-dev`, `libxml2-dev`, LAPACK/BLAS, freetype/harfbuzz,
  python3-dev, cmake, etc.) and includes the CRAN PPA recipe for a current
  R build.

- **Windows prerequisites**: The help file now documents Rtools installation
  (Rtools44 for R 4.4.x, Rtools43 for R 4.3.x, etc.), PATH configuration,
  binary-package preference (`type = "binary"`), VC++ Redistributable
  requirements for `torch`, Miniconda setup for `reticulate`, and long-path
  registry fix.

- **Expanded troubleshooting**: Section 34 of the help file is now split into
  Cross-platform, Ubuntu-specific, and Windows-specific sub-tables covering
  the most common install failures on each platform.

### Package maintenance

- **DESCRIPTION**: version **0.3.0** (2026-05-20).
- **`HELP_RCausalML_0.3.0.md`**: new help file replacing the 0.1.8 version;
  all tarball-install code snippets updated to reference `RCausalML_0.3.0.tar.gz`.

---

# RCausalML 0.1.9

### New models — Structural Causal Models (SCMs) with Deep Components (`R/causalDeepNet.R`)

Three new model families converted from Python (Pearl's SCM hierarchy — association,
intervention, counterfactual) and two graph-utility functions:

- **DeepSCM** (`deep_scm()`): Fixed-graph Structural Causal Model. Each variable's structural
  equation is a small MLP (`StructuralEquationNet`) conditioned on its graph-parents and a
  per-variable latent noise vector inferred by a variational encoder (`NoiseEncoder`). Training
  minimises an ELBO-style objective (MSE reconstruction + KL). Supports `predict.deep_scm()` and
  `intervene_deep_scm()` for do-calculus interventions (returns predictions under low/high values
  and the mean delta). Adjacency matrix can be supplied or derived from a correlation heuristic.
  CamelCase alias: `deepSCM()`.

- **DECI** (`deci_model()`): Deep End-to-end Causal Inference. Jointly learns a soft causal
  adjacency matrix (`DECIAdjacency`) and nonlinear structural equations
  (`DECIEncoder` + `DECIDecoder`) from lagged time-series windows. Uses a NOTEARS
  acyclicity penalty \eqn{h(A) = tr(exp(A ∘ A)) − d} plus KL and L1 sparsity in the ELBO
  objective. Returns `A_soft` and `A_binary` (thresholded). Supports `predict.deci_model()` and
  `ate_deci()` for Monte-Carlo average treatment effect estimation. CamelCase alias: `deciModel()`.

- **DynoTEARS** (`dynotears()`): Lagged causal discovery for multivariate time series. Learns
  lag-specific weight matrices \eqn{W^{(1)}, ..., W^{(p)}} with an augmented-Lagrangian DAG
  constraint \eqn{h(W) = tr(exp((Σ|W^{(k)}|) ∘ (Σ|W^{(k)}|))) − d}, L1 sparsity, and
  progressive rho/alpha updates. Returns `A_binary`, `W_agg`, and training diagnostics
  (`train_losses`, `dag_vals`). Supports `predict.dynotears()`. CamelCase alias: `dynoTEARS()`.

- **`evaluate_graph_recovery(A_true, A_pred, name)`**: Computes TP, FP, FN, Precision, Recall, F1,
  and Structural Hamming Distance (SHD) between a predicted binary adjacency matrix and a
  ground-truth DAG.

- **`plot_scm_dag(A, var_names, title)`**: Visualises a weighted or binary adjacency matrix as a
  heatmap (ggplot2 + reshape2 when available, base-graphics fallback). Diagonal entries masked.

### Package maintenance

- **DESCRIPTION**: version **0.1.9** (2026-05-19); extended description to cover DeepSCM, DECI,
  DynoTEARS, `evaluate_graph_recovery`, and `plot_scm_dag`; added `reshape2` to `Suggests`.
- **NAMESPACE**: exported `deep_scm`, `predict.deep_scm`, `intervene_deep_scm`, `deci_model`,
  `predict.deci_model`, `ate_deci`, `dynotears`, `predict.dynotears`, `evaluate_graph_recovery`,
  `plot_scm_dag`, `deepSCM`, `deciModel`, `dynoTEARS`; registered S3 methods
  `predict.deep_scm`, `predict.deci_model`, `predict.dynotears`.
- **README**: updated intro, algorithms table (new **SCM with Deep Components** row), dependencies
  section, and neural-test description.

---

## RCausalML 0.1.8

### New modules

- **Causal XGBoost** (`R/causalXGBoost.R`): `CXGBoost` R6 class implementing a two-head
  DragonNet-style masked-MSE causal outcome model with XGBoost and a ranger propensity
  scorer. New methods: `fit()`, `predict()`, `summary()`, `evaluate()` (PEHE + ATE vs
  ground-truth), `plot_importance()` (ggplot2 variable-importance chart), `save_model()` /
  `load_model()`, `clone_reset()`. Exported evaluation metrics `PEHE()` and `ATE()` accept
  both matrix and vector calling conventions. Bug fixes vs v1.0: Hessian always > 0,
  ranger column selection by `which.max`, verbose fully respected in parallel workers,
  XGBoost label flattening. Exported: `CXGBoost`, `PEHE`, `ATE`, `ATE_error`, `run_example`.

- **Multi-arm Causal Boosting** (`R/multi_arm_causal_boost.R`): `MultiArmCausalBoost` R6
  class extending binary CXGBoost to K ≥ 2 treatment arms. Fits separate XGBoost outcome
  models per arm and a multiclass ranger propensity forest. Supports univariate and
  multivariate Y, factor treatment, NA median imputation, and treatment contrast reports
  relative to a chosen baseline arm. Methods: `fit()`, `predict()`, `predict_ate()`,
  `summary()`, `plot_importance()`, `save()` / `load()`. Exported: `MultiArmCausalBoost`,
  `multi_arm_PEHE`, `multi_arm_ATE`, `load_multi_arm_causal_boost`,
  `save_multi_arm_causal_boost`, `run_multi_arm_causal_boost_example`.

- **Interventional Causal Representation Learning** (`R/interventionalCRL.R`):
  `InterventionalCRL` torch `nn_module` — environment-conditioned VAE with two-hidden-layer
  encoder/decoder and an analytically tractable KL divergence between `q(z|x,e)` and the
  environment prior `p(z|e)`. Designed for binary Morgan-fingerprint inputs (Bernoulli BCE
  reconstruction). `interventional_elbo_loss()` computes the negative ELBO (BCE + KL) as a
  scalar tensor. Optional `device` auto-selection (CUDA / CPU). Exported via `@export` in
  `tempoCausalVAE.R` (`TemporalCausalVAE`, `temporal_causal_loss`).

- **Temporal Causal VAE** (`R/tempoCausalVAE.R`): `TemporalCausalVAE` torch `nn_module` —
  GRU encoder/decoder with a learned `latent_dim × latent_dim` causal adjacency matrix,
  NOTEARS-style DAG acyclicity penalty, and L1 sparsity on the adjacency. `temporal_causal_loss()`
  computes the combined ELBO + DAG + sparsity loss. Beta-VAE KL weighting (`beta`) and
  separate adjacency penalty weight (`gamma`) are configurable. Exported: `TemporalCausalVAE`,
  `temporal_causal_loss`.

- **Temporal Causal Discovery Framework** (`R/temporalCausaDiscovery.R`): R/torch port of
  TCDF (Nauta et al., <https://github.com/M-Nauta/TCDF>). Implements depthwise causal TCN
  building blocks (`TCDF_chomp1d`, `TCDF_first_block`, `TCDF_temporal_block`,
  `TCDF_last_block`, `TCDF_depthwise_net`) and the `TCDF_ADDSTCN` attention-based dilated
  TCN for channel-wise causal discovery in multivariate time series. Requires `torch`.

### Package maintenance

- **DESCRIPTION**: version **0.1.8** (2026-05-19); extended description to cover all new
  modules added since 0.1.7.
- **Tests**: added `tests/test-cxgboost.R`, `tests/test-multi-arm-causal-boost.R`,
  `tests/test-interventionalCRL.R`, `tests/test-tempoCausalVAE.R`,
  `tests/test-temporalCausaDiscovery.R`.

---

## RCausalML 0.1.7

### Package maintenance

- **DESCRIPTION** / release: version **0.1.7** (2026-04-01); refreshed source tarball.
- **Installation help**: added release help guide **`HELP_RCausalML_0.1.7.md`** and linked it from `README.md` for tarball install, dependency approval behavior, and troubleshooting.
- **NAMESPACE**: export **TMLELearner**, **LinearDRLearner**, **SparseLinearDRLearner**, **XGBDRLearner**, and **ForestDRLearner**; register **coef** / **intercept** S3 methods for **LinearDRLearner** and **SparseLinearDRLearner** (aligned with `@export` in `R/meta_learners.R`).
- **DR-learner**: implementation consolidated in **`R/meta_learners.R`** (standalone `R/DRLearner.R` removed).

---

## CausalML 0.1.4

### CEVAE (Counterfactual Variational Autoencoder)

- **cevae()**: Full CEVAE implementation when **torch** is installed, aligned with Python CausalML/Pyro: generative model \(z \sim p(z)\), \(x \sim p(x|z)\), \(w \sim p(w|z)\), \(y \sim p(y|t,z)\) with twin outcome nets for imbalanced treatment. Training uses ELBO plus \(\log q(t|x)\) and \(\log q(y|t,x)\) (TraceCausalEffect_ELBO style). New arguments: `outcome_dist`, `latent_dim`, `hidden_dim`, `num_epochs`, `num_layers`, `batch_size`, `learning_rate`, `learning_rate_decay`, `num_samples`, `weight_decay`, `verbose`. Falls back to nnet or ranger placeholder when torch is not available.
- **predict.cevae()**: For torch CEVAE, ITE via Monte Carlo samples from the guide and model; optional `num_samples` in predict.
- **fit_predict_cevae()**: New convenience function for fit-then-predict in one call (matches Python `CEVAE.fit_predict(X, treatment, y)`).

---

## CausalML 0.1.3

- Package version bump and maintenance update.

---

## CausalML 0.1.2

### DragonNet (torch)

- **dragonnet()**: Full DragonNet implementation when the **torch** package is installed: shared representation (3 ELU layers), propensity head, two outcome heads \(\hat{Y}(0)\) and \(\hat{Y}(1)\), and learnable epsilon for targeted regularization. Training follows CausalML (Adam then SGD with momentum). New arguments: `neurons`, `reg_l2`, `targeted_reg`, `ratio_tar`, `batch_size`, `val_split`, `adam_epochs`/`sgd_epochs`, `verbose`. Falls back to nnet or ranger placeholder when torch is not available.
- **predict.dragonnet()**: Supports torch models; new option `propensity = TRUE` to return estimated propensity scores (torch only).

### Other

- **gain_curve()** (utils.R): Robust to list-like or length-mismatch inputs via `as.numeric(unlist(pred))` and length alignment with `tau`.
- **DESCRIPTION**: Added `torch (>= 0.10)` to Suggests.

---

## CausalML 0.1.1

### Documentation / Examples

- **07-dragonnet-metalearner.qmd**: Updated to align with the Python CausalML [DragonNet vs Meta-Learners example](https://causalml.readthedocs.io/en/latest/examples/dragonnet_example.html) and [DragonNet implementation](https://github.com/uber/causalml/blob/master/causalml/inference/tf/dragonnet.py). Uses IHDP semi-synthetic data (treatment, y_factual, y_cfactual, x1–x25), elastic-net propensity, S/T/X/R-learners and DragonNet placeholder; reports ATE, MAE, AUUC, and cumulative gain plots. Adds a synthetic section with `simulate_nuisance_and_easy_treatment()` and train/validation metrics.

---

## CausalML 0.1.0

- Initial release: meta-learners (S, T, X, R, DR), propensity estimation, uplift trees, synthetic data, instrumental variables (2SLS), matching, DragonNet/CEVAE placeholders, uplift curves and metrics (get_cumgain, plot_gain, qini_score, etc.).
