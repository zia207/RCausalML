# RCausalML: Double Machine Learning (DML) — aligned with PyWhy EconML `econml.dml`
# ---------------------------------------------------------------------------
# Theory (see EconML `econml/dml/__init__.py` and `econml/dml/dml.py`): ML estimates
# residual outcome Y - E[Y|X,W] and residual treatment T - E[T|X,W]; the CATE
# surface Theta(X) is fit by regressing those residuals so that
#   Y - E[Y|X,W] ~ Theta(X) * (T - E[T|X,W])  (linear Theta in phi(X) for parametric DML).
# Parametric classes mirror EconML: DML / LinearDML / SparseLinearDML / KernelDML /
# NonParamDML / CausalForestDML (`_BaseDML`, `_FinalWrapper` cross_product in Python).
# Matrix T => joint multi-treatment DML (final lm / glmnet / kernel only), same
# cross-product design as EconML `cross_product(F, T)`.
#
# References (EconML docstring): Chernozhukov et al. arXiv:1608.00060 (DML);
#   Nie & Wager arXiv:1712.04912 (R-learner family); Chernozhukov et al. panel;
#   Chernozhukov et al. arXiv:1806.04823 (high-dimensional second stage);
#   Foster & Syrgkanis arXiv:1901.09036 (orthogonal statistical learning).
#
# This file also implements DynamicDMLearner, OrthoIVLearner, DMLIVLearner,
# NonParamDMLIVLearner (IV / panel patterns; nuisances on [X,W] where noted).
# First-stage strings: "lm", "ranger", "glmnet", "xgb" (logistic path for discrete T).
# DoubleML (R): https://docs.doubleml.org — PLR/PLIV helpers (doubleml_plr, doubleml_plr_fit_data,
# doubleml_plr_tune_data, doubleml_pliv), data wrappers (doubleml_data_from_*), CHS2015 / CCDDHNR2018
# simulators. PLR with mlr3pipelines GraphLearners uses the same cross-fitted nuisance path as
# DoubleML’s R_dml/helper.R (dml_cv_predict / initiate_task); this module is standalone R DML.

#' Check that grf is available (for CausalForestDML)
#' @noRd
check_grf_dml <- function() {
  if (!requireNamespace("grf", quietly = TRUE))
    stop("Package 'grf' is required for CausalForestDML. Install with install.packages('grf').")
}

# --------------- Internal: first-stage and residual-on-residual ---------------

#' Out-of-sample predictions for outcome E[Y|X] (DML first stage)
#' @noRd
dml_fit_outcome <- function(X_tr, y_tr, X_te, learner = "ranger", ...) {
  df <- as.data.frame(X_tr)
  df$y <- y_tr
  if (learner == "lm") {
    m <- stats::lm(y ~ ., data = df)
    list(type = "lm", model = m)
  } else if (learner == "ranger") {
    m <- ranger::ranger(y ~ ., data = df, ...)
    list(type = "ranger", model = m)
  } else if (learner == "glmnet") {
    m <- glmnet::cv.glmnet(as.matrix(X_tr), y_tr, nfolds = min(5L, max(1L, nrow(X_tr) - 1)))
    list(type = "glmnet", model = m)
  } else if (learner == "xgb") {
    check_xgboost_dr()
    m <- do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(X_tr), y_tr, ..., default_nrounds = 50L))
    list(type = "xgb", model = m)
  } else {
    stop("model_y must be 'lm', 'ranger', 'glmnet', or 'xgb'")
  }
}

#' Out-of-sample predictions for treatment E[T|X] (DML first stage)
#' @noRd
dml_fit_treatment <- function(X_tr, t_tr, X_te, learner = "ranger", ...) {
  t_tr <- as.integer(t_tr)
  if (learner == "lm") {
    df <- as.data.frame(X_tr)
    df$t <- t_tr
    m <- stats::lm(t ~ ., data = df)
    list(type = "lm", model = m)
  } else if (learner == "ranger") {
    df <- as.data.frame(X_tr)
    df$t <- t_tr
    m <- ranger::ranger(t ~ ., data = df, ...)
    list(type = "ranger", model = m)
  } else if (learner == "glmnet") {
    m <- glmnet::cv.glmnet(X_tr, t_tr, family = "binomial", nfolds = min(5L, max(1L, length(unique(t_tr)))))
    list(type = "glmnet", model = m)
  } else if (learner == "xgb") {
    check_xgboost_dr()
    # Regression on 0/1 estimates E[T|X]; xgboost::xgboost() (>=3) does not
    # accept binary:logistic with the simplified API.
    m <- do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(X_tr), as.numeric(t_tr), ..., default_nrounds = 50L))
    list(type = "xgb", model = m)
  } else {
    stop("model_t must be 'lm', 'ranger', 'glmnet', or 'xgb'")
  }
}

#' Predict from outcome or treatment first-stage model
#' @noRd
dml_pred_nuisance <- function(m, X_new, type = c("outcome", "treatment")) {
  if (is.null(m)) return(numeric(nrow(X_new)))
  type <- match.arg(type)
  if (inherits(X_new, "data.frame")) X_new <- as.matrix(X_new)
  df_new <- as.data.frame(X_new)
  if (m$type == "lm") {
    pred <- as.vector(predict(m$model, newdata = df_new))
  } else if (m$type == "ranger") {
    pred <- predict(m$model, data = df_new)$predictions
  } else if (m$type == "glmnet") {
    pred <- as.vector(predict(m$model, newx = X_new, s = "lambda.1se", type = "response")[, 1])
    if (type == "treatment") pred <- pmax(1e-6, pmin(1 - 1e-6, pred))
  } else if (m$type == "xgb") {
    pred <- as.vector(predict(m$model, newdata = X_new))
    if (type == "treatment") pred <- pmax(1e-6, pmin(1 - 1e-6, pred))
  } else {
    return(numeric(nrow(X_new)))
  }
  pred
}

#' Number of treatment dimensions (1 for vector residual treatment)
#' @noRd
dml_ncol_res_t <- function(res_t) {
  if (is.matrix(res_t)) ncol(res_t) else 1L
}

#' Featurizer for CATE: optional intercept column (EconML-style phi(X))
#' @noRd
dml_phi_X <- function(X, fit_cate_intercept) {
  phi <- if (isTRUE(fit_cate_intercept)) cbind(intercept = 1, X) else X
  if (is.null(colnames(phi))) colnames(phi) <- paste0("phi", seq_len(ncol(phi)))
  phi
}

#' Joint design matrix: vec(tilde(T) %*% phi(X)^T) — EconML `cross_product` / `_FinalWrapper._combine`
#' @noRd
dml_joint_res_design <- function(res_t, phi) {
  if (is.matrix(res_t)) {
    do.call(cbind, lapply(seq_len(ncol(res_t)), function(j) res_t[, j] * phi))
  } else {
    as.vector(res_t) * phi
  }
}

#' Parse X, treatment, y for DML (vector or matrix continuous treatment)
#' @noRd
dml_parse_treatment_y <- function(X, treatment, y) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  y <- if (inherits(y, "data.frame")) as.numeric(y[[1L]]) else as.numeric(as.vector(y))
  if (inherits(treatment, "data.frame")) {
    Tmat <- as.matrix(treatment)
    if (ncol(Tmat) == 1L) {
      conv <- convert_to_numeric(as.data.frame(X), treatment[[1L]], y)
      return(list(X = conv$X, y = conv$y, treatment = conv$treatment,
                  multi_treatment = FALSE, Tmat = NULL, treatment_names = NULL))
    }
    if (!is.numeric(Tmat)) stop("Multivariate treatment columns must be numeric.")
    if (nrow(Tmat) != nrow(X) || length(y) != nrow(X))
      stop("nrow(X), nrow(treatment), and length(y) must match for multivariate treatment.")
    for (j in seq_len(ncol(Tmat))) check_continuous_treatment(Tmat[, j])
    tn <- colnames(treatment)
    if (is.null(tn)) tn <- paste0("T", seq_len(ncol(Tmat)))
    return(list(X = X, y = y, treatment = NULL, multi_treatment = TRUE,
                Tmat = Tmat, treatment_names = tn))
  }
  if (is.matrix(treatment)) {
    if (!is.numeric(treatment)) stop("Treatment matrix must be numeric.")
    if (ncol(treatment) == 1L) {
      conv <- convert_to_numeric(as.data.frame(X), treatment[, 1L], y)
      return(list(X = conv$X, y = conv$y, treatment = conv$treatment,
                  multi_treatment = FALSE, Tmat = NULL, treatment_names = NULL))
    }
    if (nrow(treatment) != nrow(X) || length(y) != nrow(X))
      stop("nrow(X), nrow(treatment), and length(y) must match for multivariate treatment.")
    for (j in seq_len(ncol(treatment))) check_continuous_treatment(treatment[, j])
    tn <- colnames(treatment)
    if (is.null(tn)) tn <- paste0("T", seq_len(ncol(treatment)))
    return(list(X = X, y = y, treatment = NULL, multi_treatment = TRUE,
                Tmat = treatment, treatment_names = tn))
  }
  conv <- convert_to_numeric(as.data.frame(X), treatment, y)
  list(X = conv$X, y = conv$y, treatment = conv$treatment,
       multi_treatment = FALSE, Tmat = NULL, treatment_names = NULL)
}

#' @noRd
dml_assert_final_supports_multit <- function(obj) {
  ok <- obj$model_final %in% c("lm", "glmnet", "kernel")
  if (!ok) {
    stop("Multiple continuous treatments require model_final \"lm\", \"glmnet\", or \"kernel\" (e.g. LinearDML, SparseLinearDML, KernelDML).",
         call. = FALSE)
  }
}

#' Scalar treatment: strict 0/1 binary, small integer discrete multi-arm, or continuous
#' @noRd
dml_scalar_treatment_mode <- function(treatment) {
  x <- as.numeric(na.omit(as.vector(treatment)))
  u <- sort(unique(x))
  n_u <- length(u)
  if (n_u == 0L) stop("treatment has no non-NA values.")
  if (n_u <= 2L && all(u %in% c(0, 1))) return("binary01")
  int_like <- max(abs(u - round(u))) < 1e-8
  if (n_u <= 10L && int_like) return("discrete")
  "continuous"
}

# --------------- DMLearner (base) ---------------

#' DMLearner: Double machine learning (EconML \code{econml.dml.DML} family)
#'
#' Special case of an R-learner style two-stage procedure: cross-fit nuisances
#' \eqn{\hat{q}(X) \approx E[Y \mid X]} and \eqn{\hat{f}(X) \approx E[T \mid X]}, form
#' residuals \eqn{\tilde{Y} = Y - \hat{q}(X)}, \eqn{\tilde{T} = T - \hat{f}(X)}, then
#' fit a CATE model solving (linear parametric case)
#' \eqn{\hat{\theta} = \arg\min E_n[(\tilde{Y} - \Theta(\phi(X)) \cdot \tilde{T})^2]}
#' i.e. OLS on the cross-product design \eqn{\tilde{T} \otimes \phi(X)} (EconML
#' \code{cross_product}; Python \code{_FinalWrapper._combine}). Discrete treatment
#' uses classification-style \eqn{\hat{f}} (probabilities / residualized binary \eqn{T});
#' continuous \eqn{T} uses regression for \eqn{\hat{f}}.
#'
#' @param model_y character; first-stage outcome model (\code{"lm"}, \code{"ranger"}, \code{"glmnet"}, \code{"xgb"}) — EconML \code{model_y}
#' @param model_t character; first-stage treatment model — EconML \code{model_t} (logistic-type when \eqn{T} is binary/discrete)
#' @param model_final character; final CATE stage (\code{"lm"}, \code{"glmnet"}, \code{"ranger"}, \code{"xgb"}, \code{"kernel"}, \code{"causal_forest"}) — EconML \code{model_final}
#' @param fit_cate_intercept logical; intercept in \eqn{\phi(X)} for linear final stages (EconML \code{fit_cate_intercept})
#' @param control_name baseline level for discrete \eqn{T} (EconML \code{categories} first level)
#' @param n_fold cross-fitting folds (EconML \code{cv})
#' @param ate_alpha confidence level for ATE intervals
#' @param seed random seed for fold assignment (\code{random_state})
#' @param alpha_glmnet L1 mix for \code{SparseLinearDML} / glmnet final
#' @param kernel_dim,kernel_bw random-feature kernel (EconML \code{KernelDML})
#'
#' @details
#' **Multi-treatment:** pass \code{treatment} as an \eqn{n \times d} matrix or data frame (\eqn{d > 1}).
#' First stage fits \eqn{E[T_j \mid X]} per column; final stage is joint regression on
#' \code{dml_joint_res_design(res_t, phi)}. Only \code{model_final} \code{"lm"}, \code{"glmnet"}, \code{"kernel"}.
#'
#' **EconML not mirrored here (string API):** optional controls \code{W} in nuisances (use \code{cbind(X,W)} manually);
#' \code{featurizer} / \code{treatment_featurizer} transformers; \code{mc_iters} / \code{mc_agg};
#' \code{discrete_outcome}; \code{sample_weight} / \code{freq_weight} / \code{sample_var} in \code{fit};
#' \code{sensitivity_summary}; Ray. Extensions can follow the same residual-on-residual structure as in
#' \code{econml/dml/dml.py}.
#'
#' @return Object of class \code{DMLearner}; fitted nuisances in \code{models_y}, \code{models_t}; final fit in \code{model_cate} (cf. EconML \code{model_cate} / \code{model_final_}).
#'
#' @references
#' Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W (2016).
#' Double/debiased machine learning for treatment and causal parameters. arXiv:1608.00060.
#'
#' Nie X, Wager S (2021). Quasi-oracle estimation of heterogeneous treatment effects.
#' \emph{Biometrika} 108(2), 299--319. (R-learner / orthogonal learner family.)
#'
#' EconML: \url{https://github.com/py-why/EconML} — module \code{econml.dml} (\code{dml.py}).
#'
#' DoubleML (R package): \url{https://docs.doubleml.org}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # DMLearner(...)
#' }
#' @export
DMLearner <- function(model_y = "ranger",
                      model_t = "ranger",
                      model_final = "lm",
                      fit_cate_intercept = TRUE,
                      control_name = 0,
                      n_fold = 5L,
                      ate_alpha = 0.05,
                      seed = NULL,
                      alpha_glmnet = 1,
                      kernel_dim = 20L,
                      kernel_bw = 1.0) {
  structure(list(
    model_y = model_y,
    model_t = model_t,
    model_final = model_final,
    fit_cate_intercept = fit_cate_intercept,
    control_name = control_name,
    n_fold = as.integer(n_fold),
    ate_alpha = ate_alpha,
    seed = seed,
    alpha_glmnet = alpha_glmnet,
    kernel_dim = as.integer(kernel_dim),
    kernel_bw = kernel_bw,
    models_y = NULL,
    models_t = NULL,
    model_cate = NULL,
    t_groups = NULL,
    X_names = NULL,
    res_y = NULL,
    res_t = NULL,
    X_fit = NULL,
    multi_treatment = FALSE,
    n_treat = NULL,
    treatment_names = NULL,
    scalar_treatment_mode = NULL
  ), class = "DMLearner")
}

#' LinearDML: DML with OLS / stats final stage (EconML \code{LinearDML})
#'
#' Parametric DML with linear final regression; EconML combines \code{StatsModelsCateEstimatorMixin}
#' with \code{DML} for inference-rich linear finals — here \code{model_final = "lm"}.
#'
#' @param model_y,model_t first-stage learners (default \code{"ranger"})
#' @param fit_cate_intercept logical (default \code{TRUE})
#' @param control_name,n_fold,ate_alpha,seed as in \code{\link{DMLearner}}
#' @return Object of class \code{c("LinearDML", "DMLearner")}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # LinearDML(...)
#' }
#' @export
LinearDML <- function(model_y = "ranger",
                      model_t = "ranger",
                      fit_cate_intercept = TRUE,
                      control_name = 0,
                      n_fold = 5L,
                      ate_alpha = 0.05,
                      seed = NULL) {
  obj <- DMLearner(model_y = model_y, model_t = model_t, model_final = "lm",
                   fit_cate_intercept = fit_cate_intercept, control_name = control_name,
                   n_fold = n_fold, ate_alpha = ate_alpha, seed = seed)
  class(obj) <- c("LinearDML", "DMLearner")
  obj
}

#' SparseLinearDML: DML with Lasso / elastic-net final (EconML \code{SparseLinearDML})
#'
#' High-dimensional second stage via \code{glmnet::cv.glmnet} (EconML \code{DebiasedLassoCateEstimatorMixin} path).
#'
#' @param model_y,model_t first-stage learners
#' @param fit_cate_intercept logical
#' @param alpha L1 penalty weight for final glmnet (default \code{1})
#' @param control_name,n_fold,ate_alpha,seed as in \code{\link{DMLearner}}
#' @return Object of class \code{c("SparseLinearDML", "DMLearner")}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # SparseLinearDML(...)
#' }
#' @export
SparseLinearDML <- function(model_y = "ranger",
                             model_t = "ranger",
                             fit_cate_intercept = TRUE,
                             alpha = 1,
                             control_name = 0,
                             n_fold = 5L,
                             ate_alpha = 0.05,
                             seed = NULL) {
  obj <- DMLearner(model_y = model_y, model_t = model_t, model_final = "glmnet",
                   fit_cate_intercept = fit_cate_intercept, control_name = control_name,
                   n_fold = n_fold, ate_alpha = ate_alpha, seed = seed, alpha_glmnet = alpha)
  class(obj) <- c("SparseLinearDML", "DMLearner")
  obj
}

#' KernelDML: DML with random Fourier features then linear CATE (EconML \code{KernelDML})
#'
#' Matches EconML \code{_RandomFeatures} + linear \code{model_final} on \eqn{\phi(X)}; joint design still
#' cross-multiplies with \eqn{\tilde{T}}.
#'
#' @param model_y,model_t first-stage learners
#' @param dim number of random features (EconML kernel feature dimension)
#' @param bw RBF-style bandwidth scaling
#' @param control_name,n_fold,ate_alpha,seed as in \code{\link{DMLearner}}
#' @return Object of class \code{c("KernelDML", "DMLearner")}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # KernelDML(...)
#' }
#' @export
KernelDML <- function(model_y = "ranger",
                      model_t = "ranger",
                      dim = 20L,
                      bw = 1.0,
                      control_name = 0,
                      n_fold = 5L,
                      ate_alpha = 0.05,
                      seed = NULL) {
  obj <- DMLearner(model_y = model_y, model_t = model_t, model_final = "kernel",
                   control_name = control_name, n_fold = n_fold, ate_alpha = ate_alpha,
                   seed = seed, kernel_dim = as.integer(dim), kernel_bw = bw)
  class(obj) <- c("KernelDML", "DMLearner")
  obj
}

#' NonParamDML: DML with flexible final regressor (EconML \code{NonParamDML})
#'
#' EconML uses the \code{use_weight_trick} path (\code{_FinalWrapper}) for a single continuous or binary
#' treatment; this R port fits \code{ranger}/\code{xgb} on features \eqn{\tilde{T} \odot X} to approximate
#' \eqn{\hat{\tau}(X)} from \eqn{\tilde{Y}} (see comments in \code{dml_fit_final}). \code{glmnet} final is
#' supported as a weighted flexible option.
#'
#' @param model_y,model_t first-stage learners
#' @param model_final \code{"ranger"}, \code{"xgb"}, or \code{"glmnet"}
#' @param control_name,n_fold,ate_alpha,seed as in \code{\link{DMLearner}}
#' @return Object of class \code{c("NonParamDML", "DMLearner")}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # NonParamDML(...)
#' }
#' @export
NonParamDML <- function(model_y = "ranger",
                        model_t = "ranger",
                        model_final = "ranger",
                        control_name = 0,
                        n_fold = 5L,
                        ate_alpha = 0.05,
                        seed = NULL) {
  if (!model_final %in% c("ranger", "xgb", "glmnet"))
    stop("model_final for NonParamDML must be 'ranger', 'xgb', or 'glmnet'")
  obj <- DMLearner(model_y = model_y, model_t = model_t, model_final = model_final,
                   control_name = control_name, n_fold = n_fold, ate_alpha = ate_alpha, seed = seed)
  class(obj) <- c("NonParamDML", "DMLearner")
  obj
}

#' CausalForestDML: first-stage DML + \code{grf} causal forest on residuals (EconML \code{CausalForestDML})
#'
#' Second stage: \code{grf::causal_forest} on covariates \eqn{X} with residual outcome and residual treatment
#' (R \pkg{grf} analogue of EconML’s causal-forest DML). Requires \code{grf}.
#'
#' @param model_y,model_t first-stage learners
#' @param control_name,n_fold,ate_alpha,seed as in \code{\link{DMLearner}}
#' @param num_trees,min_node_size passed to \code{grf::causal_forest}
#' @return Object of class \code{c("CausalForestDML", "DMLearner")}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # CausalForestDML(...)
#' }
#' @export
CausalForestDML <- function(model_y = "ranger",
                           model_t = "ranger",
                           control_name = 0,
                           n_fold = 5L,
                           ate_alpha = 0.05,
                           seed = NULL,
                           num_trees = 2000L,
                           min_node_size = 5L) {
  check_grf_dml()
  obj <- DMLearner(model_y = model_y, model_t = model_t, model_final = "causal_forest",
                   control_name = control_name, n_fold = n_fold, ate_alpha = ate_alpha, seed = seed)
  obj$num_trees <- as.integer(num_trees)
  obj$min_node_size <- as.integer(min_node_size)
  class(obj) <- c("CausalForestDML", "DMLearner")
  obj
}

# --------------- Fit DMLearner ---------------

#' Fit DMLearner — cross-fitting and final stage (EconML \code{DML.fit})
#'
#' Cross-validated nuisance fits per fold (analogous to EconML \code{_OrthoLearner} / \code{cv}), then
#' stacks out-of-fold residuals for the final CATE regression. Optional \code{p} fixes \eqn{E[T\mid X]}
#' for binary treatment (skips cross-fitted \eqn{\hat{f}}).
#'
#' @param X covariate matrix or data frame (\eqn{n \times d_x}); concatenate controls \eqn{W} manually if needed.
#' @param treatment vector, or \eqn{n \times d} matrix / data frame of continuous treatments (\eqn{d > 1}).
#' @param y outcome vector (\eqn{n}).
#' @param p optional propensity \eqn{E[T\mid X]} override (binary / discrete scalar \eqn{T} only).
#' @param \dots passed to first-stage learners (\code{ranger}, \code{xgboost}, etc.).
#' @return
#' Object returned by \code{fit.DMLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.DMLearner(...)
#' }
#' @export
fit.DMLearner <- function(obj, X, treatment, y, p = NULL, ...) {
  parsed <- dml_parse_treatment_y(X, treatment, y)
  X <- parsed$X
  y <- parsed$y
  n <- length(y)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  n_fold <- min(max(2L, obj$n_fold), n)
  if (!is.null(obj$seed)) set.seed(obj$seed)
  folds <- sample(rep(seq_len(n_fold), length.out = n))
  split_indices <- lapply(seq_len(n_fold), function(k) which(folds == k))

  if (isTRUE(parsed$multi_treatment)) {
    dml_assert_final_supports_multit(obj)
    if (!is.null(p))
      warning("Argument p (propensity) is ignored for multivariate continuous treatment.")
    Tmat <- parsed$Tmat
    d_t <- ncol(Tmat)
    obj$multi_treatment <- TRUE
    obj$n_treat <- d_t
    obj$treatment_names <- parsed$treatment_names
    obj$t_groups <- NULL
    obj$scalar_treatment_mode <- NULL
    pred_y <- numeric(n)
    pred_t <- matrix(0, n, d_t)
    obj$models_y <- vector("list", n_fold)
    obj$models_t <- vector("list", n_fold)
    for (k in seq_len(n_fold)) {
      train_idx <- setdiff(seq_len(n), split_indices[[k]])
      test_idx <- split_indices[[k]]
      X_tr <- X[train_idx, , drop = FALSE]
      X_te <- X[test_idx, , drop = FALSE]
      y_tr <- y[train_idx]
      obj$models_y[[k]] <- dml_fit_outcome(X_tr, y_tr, X_te, obj$model_y, ...)
      pred_y[test_idx] <- dml_pred_nuisance(obj$models_y[[k]], X_te, "outcome")
      obj$models_t[[k]] <- vector("list", d_t)
      for (j in seq_len(d_t)) {
        t_j_tr <- Tmat[train_idx, j]
        obj$models_t[[k]][[j]] <- dml_fit_outcome(X_tr, t_j_tr, X_te, obj$model_t, ...)
        pred_t[test_idx, j] <- dml_pred_nuisance(obj$models_t[[k]][[j]], X_te, "outcome")
      }
    }
    obj$res_y <- y - pred_y
    obj$res_t <- Tmat - pred_t
    obj$X_fit <- X
    obj$X_names <- colnames(X)
    obj$model_cate <- dml_fit_final(obj, X, obj$res_y, obj$res_t, ...)
    return(obj)
  }

  treatment <- parsed$treatment
  mode <- dml_scalar_treatment_mode(treatment)
  obj$scalar_treatment_mode <- mode

  if (identical(mode, "continuous")) {
    check_continuous_treatment(treatment)
    if (!is.null(p)) warning("Argument p (propensity) is ignored for continuous scalar treatment.")
    obj$t_groups <- NULL
    obj$multi_treatment <- FALSE
    obj$n_treat <- 1L
    obj$treatment_names <- NULL
    pred_y <- numeric(n)
    pred_t <- numeric(n)
    obj$models_y <- vector("list", n_fold)
    obj$models_t <- vector("list", n_fold)
    for (k in seq_len(n_fold)) {
      train_idx <- setdiff(seq_len(n), split_indices[[k]])
      test_idx <- split_indices[[k]]
      X_tr <- X[train_idx, , drop = FALSE]
      X_te <- X[test_idx, , drop = FALSE]
      y_tr <- y[train_idx]
      obj$models_y[[k]] <- dml_fit_outcome(X_tr, y_tr, X_te, obj$model_y, ...)
      pred_y[test_idx] <- dml_pred_nuisance(obj$models_y[[k]], X_te, "outcome")
      obj$models_t[[k]] <- dml_fit_outcome(X_tr, treatment[train_idx], X_te, obj$model_t, ...)
      pred_t[test_idx] <- dml_pred_nuisance(obj$models_t[[k]], X_te, "outcome")
    }
    obj$res_y <- y - pred_y
    obj$res_t <- treatment - pred_t
    obj$X_fit <- X
    obj$X_names <- colnames(X)
    obj$model_cate <- dml_fit_final(obj, X, obj$res_y, obj$res_t, ...)
    return(obj)
  }

  check_treatment_vector(treatment, obj$control_name)
  t_groups <- sort(unique(treatment[treatment != obj$control_name]))
  if (length(t_groups) > 1L) {
    message("DMLearner: multiple treatment groups detected; using first non-control as treatment (binary DML).")
    treatment_bin <- as.integer(treatment == t_groups[1L])
  } else {
    treatment_bin <- as.integer(treatment != obj$control_name)
  }
  obj$t_groups <- t_groups
  obj$multi_treatment <- FALSE
  obj$n_treat <- 1L
  obj$treatment_names <- NULL

  pred_y <- numeric(n)
  pred_t <- numeric(n)
  obj$models_y <- vector("list", n_fold)
  obj$models_t <- vector("list", n_fold)

  for (k in seq_len(n_fold)) {
    train_idx <- setdiff(seq_len(n), split_indices[[k]])
    test_idx <- split_indices[[k]]
    X_tr <- X[train_idx, , drop = FALSE]
    y_tr <- y[train_idx]
    t_tr <- treatment_bin[train_idx]
    X_te <- X[test_idx, , drop = FALSE]

    obj$models_y[[k]] <- dml_fit_outcome(X_tr, y_tr, X_te, obj$model_y, ...)
    obj$models_t[[k]] <- dml_fit_treatment(X_tr, t_tr, X_te, obj$model_t, ...)
    pred_y[test_idx] <- dml_pred_nuisance(obj$models_y[[k]], X_te, "outcome")
    pred_t[test_idx] <- dml_pred_nuisance(obj$models_t[[k]], X_te, "treatment")
  }

  if (!is.null(p)) {
    if (is.list(p)) pred_t <- as.numeric(p[[as.character(t_groups[1L])]])
    else pred_t <- as.numeric(p)
    pred_t <- pmax(1e-6, pmin(1 - 1e-6, pred_t))
  }

  obj$res_y <- y - pred_y
  obj$res_t <- treatment_bin - pred_t
  obj$X_fit <- X
  obj$X_names <- colnames(X)
  obj$model_cate <- dml_fit_final(obj, X, obj$res_y, obj$res_t, ...)
  obj
}

#' Final CATE stage: residual-on-residual regression (EconML \code{_FinalWrapper.fit} in \code{dml.py})
#' @noRd
dml_fit_final <- function(obj, X, res_y, res_t, ...) {
  n <- nrow(X)
  n_treat <- dml_ncol_res_t(res_t)
  if (n_treat > 1L && !obj$model_final %in% c("lm", "glmnet", "kernel"))
    dml_assert_final_supports_multit(obj)
  phi <- dml_phi_X(X, obj$fit_cate_intercept)
  L <- ncol(phi)
  D <- dml_joint_res_design(res_t, phi)
  colnames(D) <- make.names(rep(colnames(phi), n_treat), unique = TRUE)
  # For linear CATE: joint regression on cross(res_t, phi) (EconML DML cross_product)
  if (obj$model_final == "lm") {
    fit <- stats::lm(res_y ~ 0 + ., data = cbind(data.frame(res_y = res_y), as.data.frame(D)))
    return(list(type = "lm", model = fit, has_intercept = isTRUE(obj$fit_cate_intercept),
                n_treat = n_treat, phi_width = L))
  }
  if (obj$model_final == "glmnet") {
    use_intercept <- n_treat == 1L
    fit <- glmnet::cv.glmnet(
      D, res_y, nfolds = 5L, alpha = obj$alpha_glmnet,
      intercept = use_intercept
    )
    return(list(type = "glmnet", model = fit, has_intercept = isTRUE(obj$fit_cate_intercept),
                n_treat = n_treat, phi_width = L, glmnet_intercept = use_intercept))
  }
  if (obj$model_final == "kernel") {
    # Random Fourier features then same joint design as lm
    if (!is.null(obj$seed)) set.seed(obj$seed)
    d <- ncol(X)
    W <- matrix(stats::rnorm(d * obj$kernel_dim), d, obj$kernel_dim) / obj$kernel_bw
    phi_k <- cos(X %*% W) * sqrt(2 / obj$kernel_dim)
    phi <- dml_phi_X(phi_k, obj$fit_cate_intercept)
    L <- ncol(phi)
    D <- dml_joint_res_design(res_t, phi)
    colnames(D) <- make.names(seq_len(ncol(D)), unique = TRUE)
    fit <- stats::lm(res_y ~ 0 + ., data = cbind(data.frame(res_y = res_y), as.data.frame(D)))
    return(list(type = "kernel", model = fit, W = W, bw = obj$kernel_bw, dim = obj$kernel_dim,
                has_intercept = isTRUE(obj$fit_cate_intercept), n_treat = n_treat, phi_width = L))
  }
  if (obj$model_final %in% c("ranger", "xgb")) {
    # NonParamDML: regress res_y on (X) with weight/sample (res_t as modifier). CATE(x) = E[res_y/res_t | X] approx
    # Standard approach: regress res_y on (res_t * X) so that E[res_y | X] = theta(X) * E[res_t|X] with theta = CATE.
    # Here we fit res_y ~ X with weights res_t^2 so that theta(X) is the conditional expectation of res_y/res_t given X.
    # Simpler: fit a single model with outcome res_y and features (X, res_t) and predict at res_t=1? No.
    # Correct: final stage minimizes E[(res_y - theta(X)*res_t)^2]. So we need to fit theta(X) where outcome is res_y/res_t with weight res_t^2 (when res_t != 0). Or: fit res_y on (res_t * phi(X)) with a flexible model. So design is (res_t * X) and outcome res_y; then theta(x) = predict(model, newdata = x) but we need to predict at (1 * x) = x for CATE. So we store that we need to multiply new X by 1 when predicting. So model learns f(z) where z = res_t * x, and CATE(x) = f(x) (predict at z = x).
    if (n_treat > 1L)
      stop("NonParamDML (ranger/xgb) does not support multivariate treatment; use LinearDML, SparseLinearDML, or KernelDML.")
    df <- as.data.frame(as.vector(res_t) * X)
    df$res_y <- res_y
    if (obj$model_final == "ranger") {
      fit <- ranger::ranger(res_y ~ ., data = df, ...)
      return(list(type = "ranger", model = fit))
    }
    if (obj$model_final == "xgb") {
      check_xgboost_dr()
      fit <- do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(df[, -ncol(df)]), df$res_y, ..., default_nrounds = 50L))
      return(list(type = "xgb", model = fit))
    }
  }
  if (obj$model_final == "causal_forest") {
    if (n_treat > 1L)
      stop("CausalForestDML does not support multivariate treatment; use LinearDML, SparseLinearDML, or KernelDML.")
    check_grf_dml()
    cf <- grf::causal_forest(X, res_y, as.vector(res_t), num.trees = obj$num_trees, min.node.size = obj$min_node_size, ...)
    return(list(type = "causal_forest", model = cf))
  }
  stop("model_final not supported: ", obj$model_final)
}

#' Predict CATE from fitted DMLearner
#' @return
#' Object returned by \code{predict.DMLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.DMLearner(...)
#' }
#' @export
predict.DMLearner <- function(object, newdata, treatment = NULL, y = NULL,
                              return_components = FALSE, verbose = TRUE, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names
  n <- nrow(newdata)
  m <- object$model_cate
  if (is.null(m)) {
    warning("DMLearner not fitted.")
    return(rep(NA_real_, n))
  }

  if (m$type == "lm") {
    phi_new <- dml_phi_X(newdata, m$has_intercept)
    L <- if (!is.null(m$phi_width)) m$phi_width else ncol(phi_new)
    d <- if (!is.null(m$n_treat)) m$n_treat else 1L
    cf <- stats::coef(m$model)
    if (d == 1L) {
      te <- as.vector(as.matrix(phi_new) %*% cf[seq_len(L)])
    } else {
      te <- matrix(NA_real_, n, d)
      for (j in seq_len(d)) {
        idx <- (j - 1L) * L + seq_len(L)
        te[, j] <- as.vector(as.matrix(phi_new) %*% cf[idx])
      }
    }
  } else if (m$type == "glmnet") {
    phi_new <- dml_phi_X(newdata, m$has_intercept)
    L <- if (!is.null(m$phi_width)) m$phi_width else ncol(phi_new)
    d <- if (!is.null(m$n_treat)) m$n_treat else 1L
    if (d == 1L) {
      te <- as.vector(predict(m$model, newx = phi_new, s = "lambda.1se")[, 1L])
    } else {
      cm <- stats::coef(m$model, s = "lambda.1se")
      cf <- as.numeric(cm)
      if (length(cf) > L * d) cf <- cf[-1L]
      if (length(cf) < L * d)
        stop("glmnet coefficient length does not match multivariate CATE layout.")
      te <- matrix(NA_real_, n, d)
      for (j in seq_len(d)) {
        idx <- (j - 1L) * L + seq_len(L)
        te[, j] <- as.vector(as.matrix(phi_new) %*% cf[idx])
      }
    }
  } else if (m$type == "kernel") {
    phi_k_new <- cos(newdata %*% m$W) * sqrt(2 / m$dim)
    phi_new <- dml_phi_X(phi_k_new, m$has_intercept)
    L <- if (!is.null(m$phi_width)) m$phi_width else ncol(phi_new)
    d <- if (!is.null(m$n_treat)) m$n_treat else 1L
    cf <- stats::coef(m$model)
    if (d == 1L) {
      te <- as.vector(as.matrix(phi_new) %*% cf[seq_len(L)])
    } else {
      te <- matrix(NA_real_, n, d)
      for (j in seq_len(d)) {
        idx <- (j - 1L) * L + seq_len(L)
        te[, j] <- as.vector(as.matrix(phi_new) %*% cf[idx])
      }
    }
  } else if (m$type == "ranger") {
    # Model was fit on (res_t * X) -> res_y. So predict at (1 * x) = x gives CATE(x)
    df_new <- as.data.frame(newdata)
    te <- predict(m$model, data = df_new)$predictions
  } else if (m$type == "xgb") {
    te <- as.vector(predict(m$model, newdata = newdata))
  } else if (m$type == "causal_forest") {
    # grf::causal_forest is also class "causal_forest"; stats::predict would hit RCausalML's method
    if (inherits(m$model, "grf")) {
      grf_pred <- utils::getS3method("predict", "causal_forest", envir = asNamespace("grf"))
      pr <- grf_pred(m$model, newdata = newdata)
      te <- as.vector(pr[["predictions"]])
    } else {
      te <- as.vector(predict(m$model, newdata = newdata)$predictions)
    }
  } else {
    te <- rep(NA_real_, n)
  }

  if (is.matrix(te) && !is.null(object$treatment_names) &&
      ncol(te) == length(object$treatment_names))
    colnames(te) <- object$treatment_names

  if (return_components) {
    yhat_c <- matrix(NA_real_, n, length(object$models_y))
    for (j in seq_along(object$models_y)) {
      yhat_c[, j] <- dml_pred_nuisance(object$models_y[[j]], newdata, "outcome")
    }
    yhat_c <- rowMeans(yhat_c, na.rm = TRUE)
    list(te = te, yhat_c = yhat_c)
  } else {
    te
  }
}

#' Fit and predict (DMLearner)
#' @return
#' Object returned by \code{fit_predict.DMLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit_predict.DMLearner(...)
#' }
#' @export
fit_predict.DMLearner <- function(obj, X, treatment, y, p = NULL, return_components = FALSE, seed = NULL, ...) {
  if (!is.null(seed)) obj$seed <- seed
  obj <- fit(obj, X, treatment, y, p = p, ...)
  out <- predict(obj, X, return_components = return_components, verbose = FALSE)
  if (return_components) list(fit = obj, te = out$te, yhat_c = out$yhat_c) else list(fit = obj, te = out)
}

#' Estimate ATE for DMLearner
#' @return
#' Object returned by \code{estimate_ate.DMLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # estimate_ate.DMLearner(...)
#' }
#' @export
estimate_ate.DMLearner <- function(obj, X, treatment, y, p = NULL,
                                    return_ci = TRUE,
                                    bootstrap_ci = FALSE,
                                    n_bootstraps = 1000L,
                                    bootstrap_size = 10000L,
                                    n_cores = 1L,
                                    seed = NULL,
                                    pretrain = FALSE,
                                    ...) {
  parsed <- dml_parse_treatment_y(X, treatment, y)
  X <- parsed$X
  y <- parsed$y
  n <- length(y)
  tr_for_fit <- if (isTRUE(parsed$multi_treatment)) parsed$Tmat else parsed$treatment
  if (!isTRUE(parsed$multi_treatment) && !identical(obj$scalar_treatment_mode, "continuous")) {
    t_groups <- obj$t_groups
    if (is.null(t_groups)) {
      check_treatment_vector(tr_for_fit, obj$control_name)
      t_groups <- sort(unique(tr_for_fit[tr_for_fit != obj$control_name]))
    }
  }
  if (!pretrain) {
    if (!is.null(seed)) obj$seed <- seed
    obj <- fit(obj, X, treatment, y, p = p, ...)
  }
  te <- predict(obj, X, verbose = FALSE)
  multi_te <- is.matrix(te)
  if (multi_te) {
    ate <- colMeans(te, na.rm = TRUE)
    tn <- obj$treatment_names
    if (!is.null(tn) && length(tn) == length(ate)) names(ate) <- tn
  } else {
    ate <- mean(te, na.rm = TRUE)
  }
  if (!return_ci || !bootstrap_ci) {
    z <- stats::qnorm(1 - obj$ate_alpha / 2)
    if (multi_te) {
      se <- apply(te, 2L, function(z_) sqrt(stats::var(z_, na.rm = TRUE) / n))
      ate_lb <- ate - z * se
      ate_ub <- ate + z * se
      if (!is.null(names(ate))) {
        names(ate_lb) <- names(ate)
        names(ate_ub) <- names(ate)
      }
    } else {
      se <- sqrt(stats::var(te, na.rm = TRUE) / n)
      ate_lb <- ate - z * se
      ate_ub <- ate + z * se
    }
    return(if (return_ci) list(ate = ate, ate_lb = ate_lb, ate_ub = ate_ub) else list(ate = ate))
  }
  size <- min(as.integer(bootstrap_size), n)
  n_boot <- as.integer(n_bootstraps)
  boot_ate <- function(b) {
    idx <- sample(n, size = size, replace = TRUE)
    T_idx <- if (isTRUE(parsed$multi_treatment)) parsed$Tmat[idx, , drop = FALSE] else tr_for_fit[idx]
    obj_b <- fit(obj, X[idx, , drop = FALSE], T_idx, y[idx],
                 p = if (is.null(p)) NULL else if (is.list(p)) lapply(p, function(v) v[idx]) else p[idx],
                 ...)
    te_b <- predict(obj_b, X[idx, , drop = FALSE], verbose = FALSE)
    if (is.matrix(te_b)) colMeans(te_b, na.rm = TRUE) else mean(te_b, na.rm = TRUE)
  }
  res_boot <- parallel_lapply(seq_len(n_boot), boot_ate, n_cores = n_cores)
  ate_boot <- do.call(rbind, lapply(res_boot, function(r) matrix(as.numeric(r), nrow = 1L)))
  alpha <- obj$ate_alpha
  out <- list(ate = ate,
              ate_lb = apply(ate_boot, 2L, stats::quantile, probs = alpha / 2, names = FALSE),
              ate_ub = apply(ate_boot, 2L, stats::quantile, probs = 1 - alpha / 2, names = FALSE))
  if (multi_te && !is.null(obj$treatment_names) && length(obj$treatment_names) == length(out$ate)) {
    names(out$ate) <- obj$treatment_names
    names(out$ate_lb) <- obj$treatment_names
    names(out$ate_ub) <- obj$treatment_names
  }
  out
}

# --------------- coef / intercept for LinearDML, SparseLinearDML ---------------

#' Coefficients of the linear CATE model (LinearDML or SparseLinearDML)
#' @return
#' Object returned by \code{coef.LinearDML}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # coef.LinearDML(...)
#' }
#' @export
coef.LinearDML <- function(object, ...) {
  dml_coef_inner(object, type = "lm")
}

#' @rdname coef.LinearDML
#' @return
#' Object returned by \code{coef.SparseLinearDML}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # coef.SparseLinearDML(...)
#' }
#' @export
coef.SparseLinearDML <- function(object, ...) {
  dml_coef_inner(object, type = "glmnet")
}

#' @noRd
dml_coef_inner <- function(object, type) {
  m <- object$model_cate
  if (is.null(m) || !m$type %in% c("lm", "glmnet")) return(NULL)
  d <- if (!is.null(m$n_treat)) m$n_treat else 1L
  L <- if (!is.null(m$phi_width)) m$phi_width else NA_integer_
  if (type == "lm") {
    cf <- stats::coef(m$model)
  } else {
    cm <- coef(m$model, s = "lambda.1se")
    cf <- as.numeric(cm)
    if (is.null(m$glmnet_intercept) || isTRUE(m$glmnet_intercept)) cf <- cf[-1L]
  }
  if (d == 1L) {
    if (isTRUE(m$has_intercept) && length(cf) > 0L) cf <- cf[-1L]
    if (length(cf) == 0L) return(numeric(0))
    if (!is.null(object$X_names) && length(object$X_names) == length(cf))
      names(cf) <- object$X_names
    return(cf)
  }
  if (is.na(L) || length(cf) < L * d) return(NULL)
  nx <- if (isTRUE(m$has_intercept)) L - 1L else L
  out <- matrix(NA_real_, nx, d)
  tn <- object$treatment_names
  if (is.null(tn)) tn <- paste0("T", seq_len(d))
  for (j in seq_len(d)) {
    idx <- (j - 1L) * L + seq_len(L)
    block <- cf[idx]
    if (isTRUE(m$has_intercept)) block <- block[-1L]
    out[, j] <- block
  }
  if (!is.null(object$X_names) && nrow(out) == length(object$X_names))
    rownames(out) <- object$X_names
  colnames(out) <- tn
  out
}

#' Intercept of the linear CATE model (LinearDML or SparseLinearDML)
#' @return
#' Object returned by \code{intercept.LinearDML}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # intercept.LinearDML(...)
#' }
#' @export
intercept.LinearDML <- function(object, ...) {
  dml_intercept_inner(object, type = "lm")
}

#' @rdname intercept.LinearDML
#' @return
#' Object returned by \code{intercept.SparseLinearDML}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # intercept.SparseLinearDML(...)
#' }
#' @export
intercept.SparseLinearDML <- function(object, ...) {
  dml_intercept_inner(object, type = "glmnet")
}

#' @noRd
dml_intercept_inner <- function(object, type) {
  m <- object$model_cate
  if (is.null(m) || !m$type %in% c("lm", "glmnet")) return(NA_real_)
  if (!isTRUE(m$has_intercept)) {
    d <- if (!is.null(m$n_treat)) m$n_treat else 1L
    return(if (d == 1L) 0 else rep(0, d))
  }
  d <- if (!is.null(m$n_treat)) m$n_treat else 1L
  L <- if (!is.null(m$phi_width)) m$phi_width else NA_integer_
  if (type == "lm") {
    cf <- stats::coef(m$model)
  } else {
    cm <- coef(m$model, s = "lambda.1se")
    cf <- as.numeric(cm)
    if (is.null(m$glmnet_intercept) || isTRUE(m$glmnet_intercept)) cf <- cf[-1L]
  }
  if (d == 1L) return(as.numeric(cf[1L]))
  if (is.na(L)) return(rep(NA_real_, d))
  vapply(seq_len(d), function(j) as.numeric(cf[(j - 1L) * L + 1L]), NA_real_)
}

# --------------- Dynamic DML (panel) and DML with IV (native R) ---------------
# DynamicDMLearner: panel/sequential DML with cross-fitting (no Python).
#
# DML with IV (same nuisance backend as dml_fit_outcome / dml_fit_treatment):
#   OrthoIVLearner        — orthogonal IV moments; linear CATE in X
#   DMLIVLearner          — DMLIV pseudo-outcome / weighting; linear CATE in X
#   NonParamDMLIVLearner  — DMLIV first stage; nonlinear CATE (ranger or lm)
# Nuisance learners per slot: "lm", "ranger", "glmnet", "xgb" ("auto" -> "ranger").
# Binary T or Z: discrete_treatment / discrete_instrument; glmnet & xgb use logistic.
# CATE: predict(); nuisance quality: mlr3measures on cross-fitted preds vs Y,T,Z.

#' Resolve learner name for IV DML ("auto" -> "ranger")
#' @noRd
dml_iv_learner <- function(x) {
  if (is.null(x) || identical(x, "auto")) "ranger" else x
}

#' Build covariate matrix XW from X and W for nuisance models
#' @noRd
dml_iv_XW <- function(X, W, n) {
  if (is.null(X) && is.null(W)) return(matrix(1, n, 1))
  if (is.null(X)) return(as.matrix(W))
  if (is.null(W)) return(as.matrix(X))
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  if (inherits(W, "data.frame")) W <- as.matrix(W)
  if (is.vector(X)) X <- matrix(X, ncol = 1L)
  if (is.vector(W)) W <- matrix(W, ncol = 1L)
  cbind(X, W)
}

#' Fit E[Z|X,W] for IV DML (binary Z -> treatment model, else outcome model)
#' @noRd
dml_fit_z <- function(XW_tr, z_tr, XW_te, learner, discrete_instrument) {
  z_tr <- as.numeric(z_tr)
  if (discrete_instrument && all(z_tr %in% c(0, 1)))
    dml_fit_treatment(XW_tr, as.integer(z_tr), XW_te, learner)
  else
    dml_fit_outcome(XW_tr, z_tr, XW_te, learner)
}

#' Predict E[Z|X,W] from fitted model
#' @noRd
dml_pred_z <- function(m, XW_new, discrete_instrument) {
  dml_pred_nuisance(m, XW_new, if (discrete_instrument) "treatment" else "outcome")
}

# --------------- DynamicDMLearner (panel / sequential treatments, native R) ---------------

#' Dynamic Double Machine Learning (panel data, native R)
#'
#' Cross-fitted DML for sequential/dynamic treatment effects in panel data.
#' Builds E[Y|X,W], E[T|X,W] with cross-fitting, then fits CATE via residual-on-residual.
#' No Python/EconML dependency.
#'
#' @param model_y first-stage outcome model ("lm", "ranger", "glmnet", "xgb"); default "ranger"
#' @param model_t first-stage treatment model; default "ranger"
#' @param fit_cate_intercept logical (default TRUE)
#' @param cv integer; number of CV folds for cross-fitting (default 2)
#' @param random_state integer or NULL for random seed
#' @return Object of class \code{DynamicDMLearner}
#' @references Lewis & Syrgkanis, Double/Debiased ML for Dynamic Treatment Effects. arXiv:2002.07285.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # DynamicDMLearner(...)
#' }
#' @export
DynamicDMLearner <- function(model_y = "auto",
                             model_t = "auto",
                             fit_cate_intercept = TRUE,
                             cv = 2L,
                             random_state = NULL) {
  structure(list(
    model_y = model_y,
    model_t = model_t,
    fit_cate_intercept = fit_cate_intercept,
    cv = as.integer(cv),
    random_state = random_state,
    models_y = NULL,
    models_t = NULL,
    model_cate = NULL,
    X_names = NULL,
    res_y = NULL,
    res_t = NULL,
    X_fit = NULL
  ), class = "DynamicDMLearner")
}

#' Fit DynamicDMLearner (panel data, native R)
#' @param Y outcome vector (length n)
#' @param T treatment matrix or vector (n x d_t or length n)
#' @param X covariates for heterogeneity (n x d_x); optional
#' @param W time-varying controls (n x d_w); optional
#' @param groups group/unit id per row (length n); used only for ordering; can be NULL
#' @return
#' Object returned by \code{fit.DynamicDMLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.DynamicDMLearner(...)
#' }
#' @export
fit.DynamicDMLearner <- function(obj, Y, T, X = NULL, W = NULL, groups = NULL, ...) {
  Y <- as.numeric(Y)
  n <- length(Y)
  if (is.vector(T)) T <- matrix(T, ncol = 1L)
  T <- as.matrix(T)
  XW <- dml_iv_XW(X, W, n)
  if (is.null(colnames(XW))) colnames(XW) <- paste0("V", seq_len(ncol(XW)))
  n_fold <- min(max(2L, obj$cv), n)
  if (!is.null(obj$random_state)) set.seed(obj$random_state)
  folds <- sample(rep(seq_len(n_fold), length.out = n))
  split_indices <- lapply(seq_len(n_fold), function(k) which(folds == k))
  learner_y <- dml_iv_learner(obj$model_y)
  learner_t <- dml_iv_learner(obj$model_t)
  pred_y <- numeric(n)
  pred_t <- matrix(NA_real_, n, ncol(T))
  obj$models_y <- vector("list", n_fold)
  obj$models_t <- vector("list", n_fold)
  for (k in seq_len(n_fold)) {
    train_idx <- setdiff(seq_len(n), split_indices[[k]])
    test_idx <- split_indices[[k]]
    XW_tr <- XW[train_idx, , drop = FALSE]
    XW_te <- XW[test_idx, , drop = FALSE]
    obj$models_y[[k]] <- dml_fit_outcome(XW_tr, Y[train_idx], XW_te, learner_y, ...)
    pred_y[test_idx] <- dml_pred_nuisance(obj$models_y[[k]], XW_te, "outcome")
    obj$models_t[[k]] <- vector("list", ncol(T))
    for (j in seq_len(ncol(T))) {
      t_j_tr <- T[train_idx, j]
      if (length(unique(t_j_tr)) <= 2L)
        obj$models_t[[k]][[j]] <- dml_fit_treatment(XW_tr, as.integer(t_j_tr), XW_te, learner_t, ...)
      else
        obj$models_t[[k]][[j]] <- dml_fit_outcome(XW_tr, t_j_tr, XW_te, learner_t, ...)
      pred_t[test_idx, j] <- dml_pred_nuisance(obj$models_t[[k]][[j]], XW_te,
        if (length(unique(t_j_tr)) <= 2L) "treatment" else "outcome")
    }
  }
  res_y <- Y - pred_y
  res_t <- T - pred_t
  if (ncol(res_t) == 1L) res_t <- as.vector(res_t)
  obj$res_y <- res_y
  obj$res_t <- res_t
  obj$X_fit <- if (!is.null(X)) as.matrix(X) else matrix(1, n, 1)
  if (!is.null(X) && !is.null(colnames(X))) obj$X_names <- colnames(X)
  phi <- obj$X_fit
  if (isTRUE(obj$fit_cate_intercept)) phi <- cbind(intercept = 1, phi)
  if (is.vector(res_t)) {
    D <- res_t * phi
    obj$model_cate <- stats::lm(res_y ~ 0 + ., data = as.data.frame(D))
    obj$model_cate <- list(type = "lm", model = obj$model_cate, has_intercept = obj$fit_cate_intercept)
  } else {
    obj$model_cate <- list(type = "lm_multi", models = vector("list", ncol(res_t)), has_intercept = obj$fit_cate_intercept)
    for (j in seq_len(ncol(res_t))) {
      D <- res_t[, j] * phi
      obj$model_cate$models[[j]] <- stats::lm(res_y ~ 0 + ., data = as.data.frame(D))
    }
  }
  obj
}

#' Predict CATE (constant marginal effect) from fitted DynamicDMLearner
#' @param newdata covariate matrix X (m x d_x); can be NULL for intercept-only
#' @return
#' Object returned by \code{predict.DynamicDMLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.DynamicDMLearner(...)
#' }
#' @export
predict.DynamicDMLearner <- function(object, newdata = NULL, ...) {
  if (is.null(object$model_cate)) {
    warning("DynamicDMLearner not fitted.")
    return(NULL)
  }
  if (is.null(newdata)) newdata <- object$X_fit
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1L)
  phi <- newdata
  if (isTRUE(object$fit_cate_intercept)) phi <- cbind(intercept = 1, phi)
  m <- object$model_cate
  if (m$type == "lm") {
    te <- as.vector(as.matrix(phi) %*% stats::coef(m$model))
  } else {
    te <- as.vector(as.matrix(phi) %*% stats::coef(m$models[[1L]]))
  }
  te
}

#' Effect of treatment change (generic)
#' @param object fitted model (e.g. DynamicDMLearner)
#' @param ... passed to methods
#' @return
#' Object returned by \code{effect}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # effect(...)
#' }
#' @export
effect <- function(object, ...) UseMethod("effect")

#' Effect of treatment change (DynamicDMLearner)
#' @param T0 baseline treatment (vector or matrix)
#' @param T1 alternative treatment (same shape as T0)
#' @return
#' Object returned by \code{effect.DynamicDMLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # effect.DynamicDMLearner(...)
#' }
#' @export
effect.DynamicDMLearner <- function(object, newdata = NULL, T0 = 0, T1 = 1, ...) {
  te <- predict(object, newdata = newdata, ...)
  if (is.null(te)) return(NULL)
  T0 <- as.numeric(T0)
  T1 <- as.numeric(T1)
  sum(te * (T1 - T0))
}

# --------------- DML with IV: OrthoIV, DMLIV, NonParamDMLIV (native R) ---------------
# Aligned in spirit with Microsoft EconML (econml.iv.dml, econml.dml) and R DoubleML
# (double_ml_iivm.R: DoubleMLIIVM; double_ml_pliv.R: DoubleMLPLIV). This file uses
# string-based nuisances (lm / ranger / glmnet / xgb), not mlr3 learners.

#' Orthogonal Double ML with instrumental variables (OrthoIV)
#'
#' Cross-fitted nuisance regressions for IV double ML and a linear CATE
#' \eqn{\theta(X)^\top \phi(X)}. Solves the orthogonal moment (single instrument or
#' first column of \eqn{Z} used in the estimating equations)
#' \eqn{E[(Y - E[Y|X,W] - \theta(X)(T - E[T|X,W]))(Z - E[Z|X,W])] = 0} when
#' \code{projection = FALSE}; if \code{projection = TRUE}, the residual treatment
#' uses \eqn{T - E[T|X,W,Z]} instead of \eqn{T - E[T|X,W]}. No Python/EconML dependency.
#'
#' @section Nuisance machine learning (first stage):
#' Each of \code{model_y_xw}, \code{model_t_xw}, \code{model_t_xwz}, \code{model_z_xw}
#' is \code{"auto"} (maps to \code{"ranger"}) or one of \code{"lm"}, \code{"ranger"},
#' \code{"glmnet"}, \code{"xgb"}. Continuous \eqn{Y}, \eqn{T}, and \eqn{Z} use
#' regression-style fits. For binary \eqn{T} (\code{discrete_treatment = TRUE}) or
#' binary \eqn{Z} (\code{discrete_instrument = TRUE} with 0/1 labels), \code{glmnet}
#' and \code{xgb} use logistic (binomial) objectives; \code{ranger} fits a forest to
#' 0/1 labels; \code{lm} is a linear probability model. Together this covers linear,
#' random-forest, regularized logistic/elastic-net, and gradient-boosted nuisance models
#' as in EconML/DML IV pipelines.
#'
#' @section Treatment and instrument types:
#' \strong{Continuous} \eqn{T}: \code{discrete_treatment = FALSE} (default).
#' \strong{Binary} \eqn{T}: \code{discrete_treatment = TRUE}.
#' \strong{Multi-level} or multi-arm \eqn{T}: this constructor uses a scalar \eqn{T}
#' vector; encode categories as integers or analyze arms via \pkg{DoubleML} (e.g.
#' \code{DoubleMLPLR} with dummy \eqn{D}) or multiple runs. \strong{Continuous or binary}
#' instruments: set \code{discrete_instrument} when \eqn{Z} is binary 0/1.
#'
#' @section Prediction and evaluation:
#' The \code{predict} method returns CATE. To score nuisance predictions
#' (cross-fitted \eqn{\hat{E}[Y|X,W]}, \eqn{\hat{E}[T|\cdot]}, \eqn{\hat{E}[Z|X,W]} against
#' observed \eqn{Y}, \eqn{T}, \eqn{Z}), use \pkg{mlr3measures}, e.g. \code{rmse} and
#' \code{mae} for continuous targets and \code{acc} (or custom measures) for binary
#' nuisances.
#'
#' @param model_y_xw,model_t_xw,model_t_xwz,model_z_xw nuisance learner per equation; see above
#' @param projection logical; if \code{TRUE} use \eqn{T - E[T|X,W,Z]} in the moment
#' @param fit_cate_intercept logical (default \code{TRUE})
#' @param discrete_treatment,discrete_instrument logical (default \code{FALSE})
#' @param cv integer; cross-fitting folds (default \code{2})
#' @param random_state integer or \code{NULL}
#' @return Object of class \code{OrthoIVLearner}
#' @references Chernozhukov et al. (2018) Double/debiased ML; EconML IV DML classes;
#'   Bach et al. (2021) DoubleML \url{https://docs.doubleml.org}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # OrthoIVLearner(...)
#' }
#' @export
OrthoIVLearner <- function(model_y_xw = "auto",
                           model_t_xw = "auto",
                           model_t_xwz = "auto",
                           model_z_xw = "auto",
                           projection = FALSE,
                           fit_cate_intercept = TRUE,
                           discrete_treatment = FALSE,
                           discrete_instrument = FALSE,
                           cv = 2L,
                           random_state = NULL) {
  structure(list(
    model_y_xw = model_y_xw,
    model_t_xw = model_t_xw,
    model_t_xwz = model_t_xwz,
    model_z_xw = model_z_xw,
    projection = projection,
    fit_cate_intercept = fit_cate_intercept,
    discrete_treatment = discrete_treatment,
    discrete_instrument = discrete_instrument,
    cv = as.integer(cv),
    random_state = random_state,
    model_cate = NULL,
    X_names = NULL,
    X_fit = NULL
  ), class = "OrthoIVLearner")
}

#' Fit OrthoIVLearner (native R)
#' @param Y outcome
#' @param T treatment
#' @param Z instrument
#' @param X,W optional covariates and controls
#' @return
#' Object returned by \code{fit.OrthoIVLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.OrthoIVLearner(...)
#' }
#' @export
fit.OrthoIVLearner <- function(obj, Y, T, Z, X = NULL, W = NULL, ...) {
  Y <- as.numeric(Y)
  T <- as.numeric(T)
  n <- length(Y)
  if (is.vector(Z)) Z <- matrix(Z, ncol = 1L)
  Z <- as.matrix(Z)
  if (!is.null(X)) { if (inherits(X, "data.frame")) X <- as.matrix(X); if (is.vector(X)) X <- matrix(X, ncol = 1L) }
  if (!is.null(W)) { if (inherits(W, "data.frame")) W <- as.matrix(W); if (is.vector(W)) W <- matrix(W, ncol = 1L) }
  XW <- dml_iv_XW(X, W, n)
  X_effect <- if (is.null(X)) matrix(1, n, 1) else as.matrix(X)
  if (is.null(colnames(X_effect))) colnames(X_effect) <- paste0("X", seq_len(ncol(X_effect)))
  n_fold <- min(max(2L, obj$cv), n)
  if (!is.null(obj$random_state)) set.seed(obj$random_state)
  folds <- sample(rep(seq_len(n_fold), length.out = n))
  split_indices <- lapply(seq_len(n_fold), function(k) which(folds == k))
  ly <- dml_iv_learner(obj$model_y_xw)
  lt <- dml_iv_learner(obj$model_t_xw)
  lz <- dml_iv_learner(obj$model_z_xw)
  ltz <- dml_iv_learner(obj$model_t_xwz)
  obj$models_y <- vector("list", n_fold)
  obj$models_t <- vector("list", n_fold)
  obj$models_z <- vector("list", n_fold)
  obj$models_t_xwz <- vector("list", n_fold)
  pred_y <- numeric(n)
  pred_t <- numeric(n)
  pred_z <- matrix(NA_real_, n, ncol(Z))
  pred_t_xwz <- numeric(n)
  for (k in seq_len(n_fold)) {
    train_idx <- setdiff(seq_len(n), split_indices[[k]])
    test_idx <- split_indices[[k]]
    XW_tr <- XW[train_idx, , drop = FALSE]
    XW_te <- XW[test_idx, , drop = FALSE]
    obj$models_y[[k]] <- dml_fit_outcome(XW_tr, Y[train_idx], XW_te, ly, ...)
    pred_y[test_idx] <- dml_pred_nuisance(obj$models_y[[k]], XW_te, "outcome")
    obj$models_t[[k]] <- if (obj$discrete_treatment)
      dml_fit_treatment(XW_tr, as.integer(T[train_idx]), XW_te, lt, ...)
    else
      dml_fit_outcome(XW_tr, T[train_idx], XW_te, lt, ...)
    pred_t[test_idx] <- dml_pred_nuisance(obj$models_t[[k]], XW_te, if (obj$discrete_treatment) "treatment" else "outcome")
    obj$models_z[[k]] <- vector("list", ncol(Z))
    for (j in seq_len(ncol(Z))) {
      obj$models_z[[k]][[j]] <- dml_fit_z(XW_tr, Z[train_idx, j], XW_te, lz, obj$discrete_instrument)
      pred_z[test_idx, j] <- dml_pred_z(obj$models_z[[k]][[j]], XW_te, obj$discrete_instrument)
    }
    XWZ_tr <- cbind(XW[train_idx, , drop = FALSE], Z[train_idx, , drop = FALSE])
    XWZ_te <- cbind(XW_te, Z[test_idx, , drop = FALSE])
    obj$models_t_xwz[[k]] <- if (obj$discrete_treatment)
      dml_fit_treatment(XWZ_tr, as.integer(T[train_idx]), XWZ_te, ltz, ...)
    else
      dml_fit_outcome(XWZ_tr, T[train_idx], XWZ_te, ltz, ...)
    pred_t_xwz[test_idx] <- dml_pred_nuisance(obj$models_t_xwz[[k]], XWZ_te, if (obj$discrete_treatment) "treatment" else "outcome")
  }
  eta <- Y - pred_y
  nu <- T - pred_t
  if (obj$projection) nu <- T - pred_t_xwz
  zeta <- Z - pred_z
  if (ncol(zeta) == 1L) zeta <- as.vector(zeta)
  phi <- X_effect
  if (isTRUE(obj$fit_cate_intercept)) phi <- cbind(intercept = 1, phi)
  if (is.vector(zeta)) {
    A <- crossprod(zeta * nu * phi, phi)
    b <- as.vector(crossprod(zeta * eta, phi))
  } else {
    A <- crossprod(zeta[, 1L] * nu * phi, phi)
    b <- as.vector(crossprod(zeta[, 1L] * eta, phi))
  }
  beta <- tryCatch(solve(A, b), error = function(e) rep(NA_real_, length(b)))
  obj$model_cate <- list(type = "ortho_iv", coef = beta, has_intercept = obj$fit_cate_intercept)
  obj$X_fit <- X_effect
  obj$X_names <- colnames(X_effect)
  obj
}

#' Predict CATE from fitted OrthoIVLearner
#' @return
#' Object returned by \code{predict.OrthoIVLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.OrthoIVLearner(...)
#' }
#' @export
predict.OrthoIVLearner <- function(object, newdata = NULL, ...) {
  if (is.null(object$model_cate)) { warning("OrthoIVLearner not fitted."); return(NULL) }
  if (is.null(newdata)) newdata <- object$X_fit
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1L)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names
  phi <- newdata
  if (isTRUE(object$model_cate$has_intercept)) phi <- cbind(intercept = 1, phi)
  as.vector(as.matrix(phi) %*% object$model_cate$coef)
}

#' DML IV with linear CATE (DMLIV, native R)
#'
#' Cross-fitted first stages as in \code{OrthoIVLearner()}, then a DMLIV-style
#' pseudo-outcome: regress \eqn{(Y - \hat{E}[Y|X,W]) / (\hat{E}[T|X,W,Z] - \hat{E}[T|X,W])}
#' on \eqn{X} (linear final stage). Same nuisance ML options: \code{"lm"}, \code{"ranger"},
#' \code{"glmnet"}, \code{"xgb"} with \code{discrete_treatment} / \code{discrete_instrument}
#' selecting regression vs logistic nuisances for binary components. Applicable to
#' continuous or binary \eqn{T} (scalar); multi-arm structures align with dummy-coded
#' treatment in DoubleML PLR-style setups. Use \code{predict()} for CATE;
#' nuisance evaluation via \pkg{mlr3measures} on out-of-fold predictions. No Python.
#'
#' @param model_y_xw,model_t_xw,model_t_xwz first-stage specs (default \code{"auto"} -> \code{"ranger"})
#' @param fit_cate_intercept logical (default \code{TRUE})
#' @param discrete_treatment,discrete_instrument logical (default \code{FALSE})
#' @param cv integer (default \code{2}); \code{random_state} integer or \code{NULL}
#' @return Object of class \code{DMLIVLearner}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # DMLIVLearner(...)
#' }
#' @export
DMLIVLearner <- function(model_y_xw = "auto",
                         model_t_xw = "auto",
                         model_t_xwz = "auto",
                         fit_cate_intercept = TRUE,
                         discrete_treatment = FALSE,
                         discrete_instrument = FALSE,
                         cv = 2L,
                         random_state = NULL) {
  structure(list(
    model_y_xw = model_y_xw,
    model_t_xw = model_t_xw,
    model_t_xwz = model_t_xwz,
    fit_cate_intercept = fit_cate_intercept,
    discrete_treatment = discrete_treatment,
    discrete_instrument = discrete_instrument,
    cv = as.integer(cv),
    random_state = random_state,
    model_cate = NULL,
    X_names = NULL,
    X_fit = NULL
  ), class = "DMLIVLearner")
}

#' Fit DMLIVLearner (native R)
#' @return
#' Object returned by \code{fit.DMLIVLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.DMLIVLearner(...)
#' }
#' @export
fit.DMLIVLearner <- function(obj, Y, T, Z, X = NULL, W = NULL, ...) {
  Y <- as.numeric(Y)
  T <- as.numeric(T)
  n <- length(Y)
  if (is.vector(Z)) Z <- matrix(Z, ncol = 1L)
  Z <- as.matrix(Z)
  if (!is.null(X)) { if (inherits(X, "data.frame")) X <- as.matrix(X); if (is.vector(X)) X <- matrix(X, ncol = 1L) }
  if (!is.null(W)) { if (inherits(W, "data.frame")) W <- as.matrix(W); if (is.vector(W)) W <- matrix(W, ncol = 1L) }
  XW <- dml_iv_XW(X, W, n)
  X_effect <- if (is.null(X)) matrix(1, n, 1) else as.matrix(X)
  if (is.null(colnames(X_effect))) colnames(X_effect) <- paste0("X", seq_len(ncol(X_effect)))
  n_fold <- min(max(2L, obj$cv), n)
  if (!is.null(obj$random_state)) set.seed(obj$random_state)
  folds <- sample(rep(seq_len(n_fold), length.out = n))
  split_indices <- lapply(seq_len(n_fold), function(k) which(folds == k))
  ly <- dml_iv_learner(obj$model_y_xw)
  lt <- dml_iv_learner(obj$model_t_xw)
  ltz <- dml_iv_learner(obj$model_t_xwz)
  pred_y <- numeric(n)
  pred_t_xw <- numeric(n)
  pred_t_xwz <- numeric(n)
  obj$models_y <- vector("list", n_fold)
  obj$models_t <- vector("list", n_fold)
  obj$models_t_xwz <- vector("list", n_fold)
  for (k in seq_len(n_fold)) {
    train_idx <- setdiff(seq_len(n), split_indices[[k]])
    test_idx <- split_indices[[k]]
    XW_tr <- XW[train_idx, , drop = FALSE]
    XW_te <- XW[test_idx, , drop = FALSE]
    XWZ_tr <- cbind(XW_tr, Z[train_idx, , drop = FALSE])
    XWZ_te <- cbind(XW_te, Z[test_idx, , drop = FALSE])
    obj$models_y[[k]] <- dml_fit_outcome(XW_tr, Y[train_idx], XW_te, ly, ...)
    pred_y[test_idx] <- dml_pred_nuisance(obj$models_y[[k]], XW_te, "outcome")
    obj$models_t[[k]] <- if (obj$discrete_treatment)
      dml_fit_treatment(XW_tr, as.integer(T[train_idx]), XW_te, lt, ...)
    else
      dml_fit_outcome(XW_tr, T[train_idx], XW_te, lt, ...)
    pred_t_xw[test_idx] <- dml_pred_nuisance(obj$models_t[[k]], XW_te, if (obj$discrete_treatment) "treatment" else "outcome")
    obj$models_t_xwz[[k]] <- if (obj$discrete_treatment)
      dml_fit_treatment(XWZ_tr, as.integer(T[train_idx]), XWZ_te, ltz, ...)
    else
      dml_fit_outcome(XWZ_tr, T[train_idx], XWZ_te, ltz, ...)
    pred_t_xwz[test_idx] <- dml_pred_nuisance(obj$models_t_xwz[[k]], XWZ_te, if (obj$discrete_treatment) "treatment" else "outcome")
  }
  d <- pred_t_xwz - pred_t_xw
  sgn <- sign(d)
  sgn[sgn == 0] <- 1
  denom <- pmax(0.01, abs(d)) * sgn
  pseudo <- (Y - pred_y) / denom
  phi <- X_effect
  if (isTRUE(obj$fit_cate_intercept)) phi <- cbind(intercept = 1, phi)
  fit <- stats::lm(pseudo ~ 0 + ., data = as.data.frame(phi))
  obj$model_cate <- list(type = "lm", model = fit, has_intercept = obj$fit_cate_intercept)
  obj$X_fit <- X_effect
  obj$X_names <- colnames(X_effect)
  obj
}

#' Predict CATE from fitted DMLIVLearner
#' @return
#' Object returned by \code{predict.DMLIVLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.DMLIVLearner(...)
#' }
#' @export
predict.DMLIVLearner <- function(object, newdata = NULL, ...) {
  if (is.null(object$model_cate)) { warning("DMLIVLearner not fitted."); return(NULL) }
  if (is.null(newdata)) newdata <- object$X_fit
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1L)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names
  phi <- newdata
  if (isTRUE(object$model_cate$has_intercept)) phi <- cbind(intercept = 1, phi)
  as.vector(predict(object$model_cate$model, newdata = as.data.frame(phi)))
}

#' Nonparametric DML IV (NonParamDMLIV, native R)
#'
#' Same cross-fitted nuisances and DMLIV pseudo-outcome as \code{DMLIVLearner()},
#' but the second stage is nonlinear: random forest (\code{model_final = "ranger"},
#' default) or linear \code{lm}. First-stage \code{model_*} strings are unchanged
#' (\code{"lm"}, \code{"ranger"}, \code{"glmnet"}, \code{"xgb"}). Use for flexible
#' \eqn{\theta(X)} when the partially linear CATE in \code{DMLIVLearner} is too
#' restrictive. Binary/continuous \eqn{T} and \eqn{Z} as for DMLIV; \code{predict} for
#' CATE; \pkg{mlr3measures} for nuisance or pseudo-outcome prediction scores.
#'
#' @param model_y_xw,model_t_xw,model_t_xwz first-stage specs (default \code{"auto"})
#' @param model_final \code{"ranger"} or \code{"lm"} for final CATE (default \code{"ranger"})
#' @param discrete_treatment,discrete_instrument logical (default \code{FALSE})
#' @param cv integer (default \code{2}); \code{random_state} integer or \code{NULL}
#' @return Object of class \code{NonParamDMLIVLearner}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # NonParamDMLIVLearner(...)
#' }
#' @export
NonParamDMLIVLearner <- function(model_y_xw = "auto",
                                 model_t_xw = "auto",
                                 model_t_xwz = "auto",
                                 model_final = NULL,
                                 discrete_treatment = FALSE,
                                 discrete_instrument = FALSE,
                                 cv = 2L,
                                 random_state = NULL) {
  if (is.null(model_final)) model_final <- "ranger"
  structure(list(
    model_y_xw = model_y_xw,
    model_t_xw = model_t_xw,
    model_t_xwz = model_t_xwz,
    model_final = model_final,
    discrete_treatment = discrete_treatment,
    discrete_instrument = discrete_instrument,
    cv = as.integer(cv),
    random_state = random_state,
    model_cate = NULL,
    X_names = NULL,
    X_fit = NULL
  ), class = "NonParamDMLIVLearner")
}

#' Fit NonParamDMLIVLearner (native R)
#' @return
#' Object returned by \code{fit.NonParamDMLIVLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.NonParamDMLIVLearner(...)
#' }
#' @export
fit.NonParamDMLIVLearner <- function(obj, Y, T, Z, X = NULL, W = NULL, ...) {
  Y <- as.numeric(Y)
  T <- as.numeric(T)
  n <- length(Y)
  if (is.vector(Z)) Z <- matrix(Z, ncol = 1L)
  Z <- as.matrix(Z)
  if (!is.null(X)) { if (inherits(X, "data.frame")) X <- as.matrix(X); if (is.vector(X)) X <- matrix(X, ncol = 1L) }
  if (!is.null(W)) { if (inherits(W, "data.frame")) W <- as.matrix(W); if (is.vector(W)) W <- matrix(W, ncol = 1L) }
  XW <- dml_iv_XW(X, W, n)
  X_effect <- if (is.null(X)) matrix(1, n, 1) else as.matrix(X)
  if (is.null(colnames(X_effect))) colnames(X_effect) <- paste0("X", seq_len(ncol(X_effect)))
  n_fold <- min(max(2L, obj$cv), n)
  if (!is.null(obj$random_state)) set.seed(obj$random_state)
  folds <- sample(rep(seq_len(n_fold), length.out = n))
  split_indices <- lapply(seq_len(n_fold), function(k) which(folds == k))
  ly <- dml_iv_learner(obj$model_y_xw)
  lt <- dml_iv_learner(obj$model_t_xw)
  ltz <- dml_iv_learner(obj$model_t_xwz)
  pred_y <- numeric(n)
  pred_t_xw <- numeric(n)
  pred_t_xwz <- numeric(n)
  obj$models_y <- vector("list", n_fold)
  obj$models_t <- vector("list", n_fold)
  obj$models_t_xwz <- vector("list", n_fold)
  for (k in seq_len(n_fold)) {
    train_idx <- setdiff(seq_len(n), split_indices[[k]])
    test_idx <- split_indices[[k]]
    XW_tr <- XW[train_idx, , drop = FALSE]
    XW_te <- XW[test_idx, , drop = FALSE]
    XWZ_tr <- cbind(XW_tr, Z[train_idx, , drop = FALSE])
    XWZ_te <- cbind(XW_te, Z[test_idx, , drop = FALSE])
    obj$models_y[[k]] <- dml_fit_outcome(XW_tr, Y[train_idx], XW_te, ly, ...)
    pred_y[test_idx] <- dml_pred_nuisance(obj$models_y[[k]], XW_te, "outcome")
    obj$models_t[[k]] <- if (obj$discrete_treatment)
      dml_fit_treatment(XW_tr, as.integer(T[train_idx]), XW_te, lt, ...)
    else
      dml_fit_outcome(XW_tr, T[train_idx], XW_te, lt, ...)
    pred_t_xw[test_idx] <- dml_pred_nuisance(obj$models_t[[k]], XW_te, if (obj$discrete_treatment) "treatment" else "outcome")
    obj$models_t_xwz[[k]] <- if (obj$discrete_treatment)
      dml_fit_treatment(XWZ_tr, as.integer(T[train_idx]), XWZ_te, ltz, ...)
    else
      dml_fit_outcome(XWZ_tr, T[train_idx], XWZ_te, ltz, ...)
    pred_t_xwz[test_idx] <- dml_pred_nuisance(obj$models_t_xwz[[k]], XWZ_te, if (obj$discrete_treatment) "treatment" else "outcome")
  }
  d <- pred_t_xwz - pred_t_xw
  sgn <- sign(d)
  sgn[sgn == 0] <- 1
  denom <- pmax(0.01, abs(d)) * sgn
  pseudo <- (Y - pred_y) / denom
  df_final <- as.data.frame(X_effect)
  df_final$pseudo <- pseudo
  if (obj$model_final == "ranger") {
    obj$model_cate <- list(type = "ranger", model = ranger::ranger(pseudo ~ ., data = df_final, ...))
  } else {
    obj$model_cate <- list(type = "lm", model = stats::lm(pseudo ~ 0 + ., data = df_final), has_intercept = FALSE)
  }
  obj$X_fit <- X_effect
  obj$X_names <- colnames(X_effect)
  obj
}

#' Predict CATE from fitted NonParamDMLIVLearner
#' @return
#' Object returned by \code{predict.NonParamDMLIVLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.NonParamDMLIVLearner(...)
#' }
#' @export
predict.NonParamDMLIVLearner <- function(object, newdata = NULL, ...) {
  if (is.null(object$model_cate)) { warning("NonParamDMLIVLearner not fitted."); return(NULL) }
  if (is.null(newdata)) newdata <- object$X_fit
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1L)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names
  if (object$model_cate$type == "ranger")
    as.vector(predict(object$model_cate$model, data = as.data.frame(newdata))$predictions)
  else
    as.vector(predict(object$model_cate$model, newdata = as.data.frame(newdata)))
}

# --------------- Optional: DoubleML (R) backend ---------------

#' @keywords internal
#' @noRd
.doubleml_plr_build <- function(dml_data, ml_l, ml_m, ml_g = NULL,
                                n_folds = 5L, n_rep = 1L,
                                score = "partialling out",
                                apply_cross_fitting = TRUE,
                                dml_procedure = "dml2",
                                draw_sample_splitting = TRUE,
                                ...) {
  need_ml_g <- (is.character(score) && score == "IV-type") || is.function(score)
  if (need_ml_g && is.null(ml_g)) ml_g <- ml_l$clone()
  args <- list(
    data = dml_data,
    ml_l = ml_l,
    ml_m = ml_m,
    n_folds = n_folds,
    n_rep = n_rep,
    score = score,
    dml_procedure = dml_procedure,
    draw_sample_splitting = draw_sample_splitting,
    apply_cross_fitting = apply_cross_fitting
  )
  if (need_ml_g) args$ml_g <- ml_g
  dots <- list(...)
  for (nm in names(dots)) args[[nm]] <- dots[[nm]]
  do.call(DoubleML::DoubleMLPLR$new, args)
}

#' @keywords internal
#' @noRd
.doubleml_plr_resolve_newdata <- function(dml_data, newdata) {
  if (!is.null(newdata)) return(as.data.frame(newdata))
  xc <- dml_data$x_cols
  if (!length(xc)) return(NULL)
  as.data.frame(dml_data$data_model[, xc, with = FALSE])
}

#' @keywords internal
#' @noRd
.doubleml_plr_pack_result <- function(dml_plr, newdata, return_ml_object) {
  ate <- dml_plr$coef
  ate_se <- dml_plr$se
  cate <- NULL
  if (!is.null(newdata) && nrow(newdata) > 0L) {
    tryCatch(
      cate <- dml_plr$predict(newdata = newdata),
      error = function(e) NULL
    )
  }
  out <- list(ate = ate, ate_se = ate_se, cate = cate)
  if (isTRUE(return_ml_object)) out$dml_obj <- dml_plr
  out
}

#' Fit DoubleML PLR from a \code{DoubleMLData} object and mlr3 learners
#'
#' Convenience wrapper for \code{DoubleML::DoubleMLPLR$new(...)$fit()} when the data
#' backend is already built (e.g. \code{\link{doubleml_data_from_matrix}},
#' \code{\link{doubleml_data_from_data_frame}}). Use this with standard learners,
#' \code{mlr3pipelines} \code{GraphLearner}s, or ensemble pipelines (\code{gunion},
#' \code{regravg} / \code{classifavg}): nuisance estimation follows the same cross-fitted
#' resampling path as in \pkg{DoubleML}’s internal helpers (\code{R_dml/helper.R}:
#' \code{dml_cv_predict}, \code{initiate_task}, \code{initiate_learner}).
#'
#' @param dml_data A \code{DoubleMLData} object.
#' @param ml_l,ml_m,ml_g mlr3 learners for \eqn{l_0(X)}, \eqn{m_0(X)}, and (if needed) \eqn{g_0(X)}; \code{ml_g} is used for \code{score = "IV-type"} or a custom score function.
#' @param n_folds,n_rep,score,apply_cross_fitting,dml_procedure,draw_sample_splitting Passed to \code{DoubleMLPLR$new}.
#' @param newdata Optional \code{data.frame} of covariates for \code{predict}; default uses \code{dml_data$x_cols} from the fitted backend.
#' @param return_ml_object If \code{TRUE}, include \code{dml_obj} (fitted \code{DoubleMLPLR}).
#' @param \dots Further arguments passed to \code{DoubleMLPLR$new}.
#' @return List with \code{ate}, \code{ate_se}, optional \code{cate}, optional \code{dml_obj}; or \code{NULL} if \pkg{DoubleML} / \pkg{mlr3} unavailable.
#' @references DoubleML pipelines: \url{https://docs.doubleml.org/stable/examples/R_double_ml_pipeline.html}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_plr_fit_data(...)
#' }
#' @export
doubleml_plr_fit_data <- function(dml_data, ml_l, ml_m, ml_g = NULL,
                                  n_folds = 5L, n_rep = 1L,
                                  score = "partialling out",
                                  apply_cross_fitting = TRUE,
                                  dml_procedure = "dml2",
                                  draw_sample_splitting = TRUE,
                                  return_ml_object = FALSE,
                                  newdata = NULL,
                                  ...) {
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    message("Package 'DoubleML' not installed. Install with: install.packages('DoubleML')")
    return(NULL)
  }
  if (!requireNamespace("mlr3", quietly = TRUE)) {
    message("Package 'mlr3' required for doubleml_plr_fit_data.")
    return(NULL)
  }
  if (!inherits(dml_data, "DoubleMLData")) {
    stop("dml_data must inherit from DoubleMLData (use DoubleML::DoubleMLData$new or doubleml_data_from_*).", call. = FALSE)
  }
  dml_plr <- .doubleml_plr_build(
    dml_data = dml_data,
    ml_l = ml_l,
    ml_m = ml_m,
    ml_g = ml_g,
    n_folds = n_folds,
    n_rep = n_rep,
    score = score,
    apply_cross_fitting = apply_cross_fitting,
    dml_procedure = dml_procedure,
    draw_sample_splitting = draw_sample_splitting,
    ...
  )
  dml_plr$fit()
  nd <- .doubleml_plr_resolve_newdata(dml_data, newdata)
  .doubleml_plr_pack_result(dml_plr, nd, return_ml_object)
}

#' Tune then fit DoubleML PLR (mlr3tuning + \code{DoubleMLPLR$tune})
#'
#' Builds a \code{DoubleMLPLR} object, runs \code{$tune(param_set, tune_settings, tune_on_folds)},
#' then \code{$fit()}. Use after constructing \code{ml_l} / \code{ml_m} (including pipeline or
#' ensemble learners) and a \code{DoubleMLData} backend. Requires \pkg{mlr3tuning} (and typically
#' \pkg{paradox}) in addition to \pkg{DoubleML} and \pkg{mlr3}.
#'
#' @inheritParams doubleml_plr_fit_data
#' @param param_set Named list of \pkg{paradox} parameter sets per nuisance, e.g. \code{list(ml_l = ps(...), ml_m = ps(...))}; see \code{DoubleMLPLR$tune}.
#' @param tune_settings List with at least \code{terminator} and tuning algorithm (e.g. \code{algorithm}, \code{resolution}, \code{rsmp_tune}); see \code{?DoubleML::DoubleML}.
#' @param tune_on_folds Logical; passed to \code{$tune()}.
#' @return Same structure as \code{\link{doubleml_plr_fit_data}}; \code{NULL} if a required package is missing.
#' @references \url{https://docs.doubleml.org}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_plr_tune_data(...)
#' }
#' @export
doubleml_plr_tune_data <- function(dml_data, ml_l, ml_m, ml_g = NULL,
                                   param_set, tune_settings,
                                   tune_on_folds = FALSE,
                                   n_folds = 5L, n_rep = 1L,
                                   score = "partialling out",
                                   apply_cross_fitting = TRUE,
                                   dml_procedure = "dml2",
                                   draw_sample_splitting = TRUE,
                                   return_ml_object = FALSE,
                                   newdata = NULL,
                                   ...) {
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    message("Package 'DoubleML' not installed. Install with: install.packages('DoubleML')")
    return(NULL)
  }
  if (!requireNamespace("mlr3", quietly = TRUE)) {
    message("Package 'mlr3' required for doubleml_plr_tune_data.")
    return(NULL)
  }
  if (!requireNamespace("mlr3tuning", quietly = TRUE)) {
    message("Package 'mlr3tuning' required for doubleml_plr_tune_data.")
    return(NULL)
  }
  if (!inherits(dml_data, "DoubleMLData")) {
    stop("dml_data must inherit from DoubleMLData (use DoubleML::DoubleMLData$new or doubleml_data_from_*).", call. = FALSE)
  }
  dml_plr <- .doubleml_plr_build(
    dml_data = dml_data,
    ml_l = ml_l,
    ml_m = ml_m,
    ml_g = ml_g,
    n_folds = n_folds,
    n_rep = n_rep,
    score = score,
    apply_cross_fitting = apply_cross_fitting,
    dml_procedure = dml_procedure,
    draw_sample_splitting = draw_sample_splitting,
    ...
  )
  dml_plr$tune(
    param_set = param_set,
    tune_settings = tune_settings,
    tune_on_folds = tune_on_folds
  )
  dml_plr$fit()
  nd <- .doubleml_plr_resolve_newdata(dml_data, newdata)
  .doubleml_plr_pack_result(dml_plr, nd, return_ml_object)
}

#' Fit Partially Linear DML via DoubleML (R) when available
#'
#' If the \code{DoubleML} package is installed, fits a partially linear regression
#' model using \code{DoubleML::DoubleMLPLR} and returns ATE and optional CATE predictions.
#' Otherwise returns NULL. Useful for comparison with \code{LinearDML} or for using
#' DoubleML's inference (e.g. \code{confint}, \code{bootstrap}). Requires \code{mlr3} and
#' \code{mlr3learners} (dependencies of DoubleML). For a pre-built \code{DoubleMLData} object
#' and arbitrary learners (including pipelines), use \code{\link{doubleml_plr_fit_data}}.
#'
#' @param X covariate matrix or data.frame
#' @param treatment treatment vector (binary 0/1 uses \code{classif} learner by default; continuous \eqn{D} uses \code{regr})
#' @param y outcome vector
#' @param ml_l mlr3 learner for \eqn{E[Y|X]} (default: \code{lrn("regr.ranger")})
#' @param ml_m mlr3 learner for \eqn{E[D|X]} (default: \code{classif.ranger} if \code{treatment} is binary, else \code{regr.ranger})
#' @param ml_g mlr3 learner for \eqn{g_0(X) = E[Y - D\theta_0|X]}; required for \code{score = "IV-type"} or a custom score function (default: clone of \code{ml_l} when needed)
#' @param n_folds,n_rep cross-fitting folds and sample-splitting repetitions (passed to \code{DoubleMLPLR})
#' @param score \code{"partialling out"}, \code{"IV-type"}, or a custom score \code{function(y, d, l_hat, m_hat, g_hat, smpls)} returning \code{list(psi_a, psi_b)}
#' @param apply_cross_fitting logical; if \code{FALSE}, no cross-fitting (see DoubleML docs)
#' @param dml_procedure \code{"dml1"} or \code{"dml2"}
#' @param return_ml_object if \code{TRUE}, return the fitted \code{DoubleMLPLR} object as \code{dml_obj}
#' @return List with \code{ate}, \code{ate_se}, \code{cate} (if predict available), and optionally \code{dml_obj}. Or NULL if DoubleML not installed.
#' @references DoubleML: https://docs.doubleml.org ; Bach et al. (2021) arXiv:2103.09603
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_plr(...)
#' }
#' @export
doubleml_plr <- function(X, treatment, y,
                         ml_l = NULL,
                         ml_m = NULL,
                         ml_g = NULL,
                         n_folds = 5L,
                         n_rep = 1L,
                         score = "partialling out",
                         apply_cross_fitting = TRUE,
                         dml_procedure = "dml2",
                         return_ml_object = FALSE) {
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    message("Package 'DoubleML' not installed. Install with: install.packages('DoubleML')")
    return(NULL)
  }
  if (!requireNamespace("mlr3", quietly = TRUE)) {
    message("Package 'mlr3' required for doubleml_plr.")
    return(NULL)
  }
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  treatment <- as.numeric(treatment)
  y <- as.numeric(y)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  df <- data.frame(y = y, d = treatment, X)
  dml_data <- DoubleML::DoubleMLData$new(df, y_col = "y", d_cols = "d", x_cols = colnames(X))
  if (is.null(ml_l)) ml_l <- mlr3::lrn("regr.ranger")
  d_bin <- all(treatment %in% c(0, 1))
  if (is.null(ml_m)) {
    ml_m <- if (d_bin) mlr3::lrn("classif.ranger") else mlr3::lrn("regr.ranger")
  }
  need_ml_g <- (is.character(score) && score == "IV-type") || is.function(score)
  if (need_ml_g && is.null(ml_g)) ml_g <- ml_l$clone()
  doubleml_plr_fit_data(
    dml_data = dml_data,
    ml_l = ml_l,
    ml_m = ml_m,
    ml_g = if (need_ml_g) ml_g else NULL,
    n_folds = n_folds,
    n_rep = n_rep,
    score = score,
    apply_cross_fitting = apply_cross_fitting,
    dml_procedure = dml_procedure,
    return_ml_object = return_ml_object,
    newdata = as.data.frame(X)
  )
}

# --------------- DoubleML data backend wrappers (double_ml_data.R) ---------------

#' Build \code{DoubleMLData} or \code{DoubleMLClusterData} from a data frame
#'
#' Thin wrapper around \code{DoubleML::double_ml_data_from_data_frame()}.
#' Specify \code{cluster_cols} for a cluster backend; otherwise a standard
#' \code{DoubleMLData} is returned.
#'
#' @param df, x_cols, y_col, d_cols, z_cols, s_col, cluster_cols, use_other_treat_as_covariate
#'   As in \pkg{DoubleML} (see \code{?DoubleML::double_ml_data_from_data_frame}).
#' @return A \code{DoubleMLData} or \code{DoubleMLClusterData} object, or \code{NULL} if
#'   \pkg{DoubleML} is not installed.
#' @references DoubleML: \url{https://docs.doubleml.org}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_data_from_data_frame(...)
#' }
#' @export
doubleml_data_from_data_frame <- function(df, x_cols = NULL, y_col = NULL,
                                            d_cols = NULL, z_cols = NULL, s_col = NULL,
                                            cluster_cols = NULL,
                                            use_other_treat_as_covariate = TRUE) {
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    message("Package 'DoubleML' not installed. Install with: install.packages('DoubleML')")
    return(NULL)
  }
  DoubleML::double_ml_data_from_data_frame(
    df = df, x_cols = x_cols, y_col = y_col, d_cols = d_cols,
    z_cols = z_cols, s_col = s_col, cluster_cols = cluster_cols,
    use_other_treat_as_covariate = use_other_treat_as_covariate
  )
}

#' Build \code{DoubleMLData} from matrices (\code{X}, \code{y}, \code{d}, optional \code{z}, etc.)
#'
#' Wrapper around \code{DoubleML::double_ml_data_from_matrix()}.
#'
#' @param X,y,d,z,s,cluster_vars,data_class,use_other_treat_as_covariate As in \pkg{DoubleML}.
#' @return \code{DoubleMLData}, \code{DoubleMLClusterData}, or \code{data.table} per \code{data_class};
#'   \code{NULL} if \pkg{DoubleML} is not installed.
#' @references DoubleML: \url{https://docs.doubleml.org}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_data_from_matrix(...)
#' }
#' @export
doubleml_data_from_matrix <- function(X = NULL, y, d, z = NULL, s = NULL, cluster_vars = NULL,
                                      data_class = "DoubleMLData",
                                      use_other_treat_as_covariate = TRUE) {
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    message("Package 'DoubleML' not installed. Install with: install.packages('DoubleML')")
    return(NULL)
  }
  DoubleML::double_ml_data_from_matrix(
    X = X, y = y, d = d, z = z, s = s, cluster_vars = cluster_vars,
    data_class = data_class, use_other_treat_as_covariate = use_other_treat_as_covariate
  )
}

# --------------- DoubleML PLIV (double_ml_pliv.R) ---------------

#' Partially linear IV DML via \pkg{DoubleML} (\code{DoubleMLPLIV})
#'
#' Fits the partially linear IV model with cross-fitted nuisances
#' \eqn{l_0(X) = E[Y|X]}, \eqn{m_0(X) = E[Z|X]}, \eqn{r_0(X) = E[D|X]} (and optional
#' \eqn{g_0} for \code{score = "IV-type"}), matching \code{DoubleML::DoubleMLPLIV}.
#'
#' @param X covariate matrix or data.frame.
#' @param treatment endogenous treatment \eqn{D} (vector).
#' @param y outcome \eqn{Y}.
#' @param Z instrument(s): vector or matrix (\eqn{n \times n_z}).
#' @param ml_l, ml_m, ml_r, ml_g \code{mlr3} learners; \code{ml_g} required for
#'   \code{score = "IV-type"} (default: clone of \code{ml_l} when needed).
#' @param partialX, partialZ logical; see \code{DoubleML::DoubleMLPLIV}.
#' @param n_folds,n_rep,score,apply_cross_fitting,dml_procedure,return_ml_object
#'   As for \code{\link{doubleml_plr}} / \pkg{DoubleML}.
#' @return List with \code{ate}, \code{ate_se}, optional \code{cate}, and optionally \code{dml_obj}.
#'   \code{NULL} if \pkg{DoubleML} or \pkg{mlr3} is unavailable.
#' @references DoubleML PLIV: \url{https://docs.doubleml.org}; \code{double_ml_pliv.R} in \pkg{DoubleML}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_pliv(...)
#' }
#' @export
doubleml_pliv <- function(X, treatment, y, Z,
                          ml_l = NULL, ml_m = NULL, ml_r = NULL, ml_g = NULL,
                          partialX = TRUE, partialZ = FALSE,
                          n_folds = 5L, n_rep = 1L,
                          score = "partialling out",
                          apply_cross_fitting = TRUE,
                          dml_procedure = "dml2",
                          return_ml_object = FALSE) {
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    message("Package 'DoubleML' not installed. Install with: install.packages('DoubleML')")
    return(NULL)
  }
  if (!requireNamespace("mlr3", quietly = TRUE)) {
    message("Package 'mlr3' required for doubleml_pliv.")
    return(NULL)
  }
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  treatment <- as.numeric(treatment)
  y <- as.numeric(y)
  n <- length(y)
  if (is.null(Z)) stop("Z (instrument) must be provided for doubleml_pliv.", call. = FALSE)
  if (is.vector(Z)) Z <- matrix(Z, ncol = 1L)
  Z <- as.matrix(Z)
  if (nrow(Z) != n) stop("nrow(Z) must match length(y).", call. = FALSE)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  nz <- ncol(Z)
  z_cols <- if (nz == 1L) "z" else paste0("z", seq_len(nz))
  colnames(Z) <- z_cols
  df <- data.frame(y = y, d = treatment, X, Z, check.names = FALSE)
  x_cols <- colnames(X)
  dml_data <- DoubleML::DoubleMLData$new(
    df, y_col = "y", d_cols = "d", x_cols = x_cols, z_cols = z_cols
  )
  if (isTRUE(partialZ) && !isTRUE(partialX)) {
    if (is.null(ml_r)) ml_r <- mlr3::lrn("regr.ranger")
    dml_pliv <- DoubleML::DoubleMLPLIV$new(
      dml_data,
      ml_l = NULL, ml_m = NULL, ml_r = ml_r,
      partialX = FALSE, partialZ = TRUE,
      n_folds = n_folds, n_rep = n_rep, score = score,
      dml_procedure = dml_procedure, apply_cross_fitting = apply_cross_fitting
    )
  } else {
    if (is.null(ml_l)) ml_l <- mlr3::lrn("regr.ranger")
    if (is.null(ml_m)) ml_m <- ml_l$clone()
    if (is.null(ml_r)) ml_r <- ml_l$clone()
    need_ml_g <- (is.character(score) && score == "IV-type") || is.function(score)
    if (need_ml_g && is.null(ml_g)) ml_g <- ml_l$clone()
    if (need_ml_g) {
      dml_pliv <- DoubleML::DoubleMLPLIV$new(
        dml_data,
        ml_l = ml_l, ml_m = ml_m, ml_r = ml_r, ml_g = ml_g,
        partialX = partialX, partialZ = partialZ,
        n_folds = n_folds, n_rep = n_rep, score = score,
        dml_procedure = dml_procedure, apply_cross_fitting = apply_cross_fitting
      )
    } else {
      dml_pliv <- DoubleML::DoubleMLPLIV$new(
        dml_data,
        ml_l = ml_l, ml_m = ml_m, ml_r = ml_r,
        partialX = partialX, partialZ = partialZ,
        n_folds = n_folds, n_rep = n_rep, score = score,
        dml_procedure = dml_procedure, apply_cross_fitting = apply_cross_fitting
      )
    }
  }
  dml_pliv$fit()
  ate <- dml_pliv$coef
  ate_se <- dml_pliv$se
  cate <- NULL
  if (nrow(X) > 0) {
    tryCatch({
      cate <- dml_pliv$predict(newdata = as.data.frame(X))
    }, error = function(e) NULL)
  }
  out <- list(ate = ate, ate_se = ate_se, cate = cate)
  if (isTRUE(return_ml_object)) out$dml_obj <- dml_pliv
  out
}

# --------------- DoubleML-style datasets (datasets.R; DGPs without extra deps) ---------------

#' @noRd
.dml_mvrnorm <- function(n, mean, sigma) {
  MASS::mvrnorm(n = n, mu = mean, Sigma = sigma)
}

#' Simulated PLR data (Chernozhukov et al., 2018 Fig. 1 DGP)
#'
#' Reproduces the \code{make_plr_CCDDHNR2018} process from \pkg{DoubleML}
#' using \code{MASS::mvrnorm}. For \code{return_type = "DoubleMLData"}, \pkg{DoubleML}
#' must be installed.
#'
#' @param n_obs,dim_x,alpha,return_type As in \pkg{DoubleML}::\code{make_plr_CCDDHNR2018}.
#' @return \code{data.frame}, \code{list} with matrices (\code{X}, \code{y}, \code{d}), or \code{DoubleMLData}.
#' @references Chernozhukov et al. (2018), \emph{The Econometrics Journal}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_make_plr_CCDDHNR2018(...)
#' }
#' @export
doubleml_make_plr_CCDDHNR2018 <- function(n_obs = 500, dim_x = 20, alpha = 0.5,
                                          return_type = "data.frame") {
  return_type <- match.arg(return_type, c("matrix", "data.frame", "DoubleMLData"))
  n_obs <- as.integer(n_obs)
  dim_x <- as.integer(dim_x)
  if (n_obs < 1L || dim_x < 1L) stop("n_obs and dim_x must be positive.", call. = FALSE)
  cov_mat <- stats::toeplitz(0.7^(0:(dim_x - 1)))
  a_0 <- 1
  a_1 <- 0.25
  s_1 <- 1
  b_0 <- 1
  b_1 <- 0.25
  s_2 <- 1
  x <- .dml_mvrnorm(n_obs, rep(0, dim_x), cov_mat)
  d <- as.matrix(a_0 * x[, 1L] + a_1 * (exp(x[, 3L]) / (1 + exp(x[, 3L]))) + s_1 * stats::rnorm(n_obs))
  y <- as.matrix(alpha * d + b_0 * exp(x[, 1L]) / (1 + exp(x[, 1L])) + b_1 * x[, 3L] + s_2 * stats::rnorm(n_obs))
  colnames(x) <- paste0("X", seq_len(dim_x))
  colnames(y) <- "y"
  colnames(d) <- "d"
  if (return_type == "matrix") {
    return(list(X = x, y = y, d = d))
  }
  dat <- data.frame(x, y = c(y), d = c(d))
  if (return_type == "data.frame") return(dat)
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    stop("return_type = 'DoubleMLData' requires package 'DoubleML'.", call. = FALSE)
  }
  DoubleML::DoubleMLData$new(dat, y_col = "y", d_cols = "d")
}

#' Simulated PLIV data (Chernozhukov, Hansen, Spindler, 2015)
#'
#' Same DGP as \pkg{DoubleML}::\code{make_pliv_CHS2015}. Uses \code{MASS::mvrnorm}.
#'
#' @param n_obs,alpha,dim_x,dim_z,return_type As in \pkg{DoubleML}::\code{make_pliv_CHS2015}.
#' @return \code{list}, \code{data.frame}, or \code{DoubleMLData}.
#' @references Chernozhukov V, Hansen C, Spindler M (2015), AER P&P.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_make_pliv_CHS2015(...)
#' }
#' @export
doubleml_make_pliv_CHS2015 <- function(n_obs, alpha = 1, dim_x = 200, dim_z = 150,
                                       return_type = "data.frame") {
  return_type <- match.arg(return_type, c("matrix", "data.frame", "DoubleMLData"))
  n_obs <- as.integer(n_obs)
  dim_x <- as.integer(dim_x)
  dim_z <- as.integer(dim_z)
  if (dim_x < dim_z) stop("dim_x must be >= dim_z (CHS2015 DGP).", call. = FALSE)
  sigma_e_u <- matrix(c(1, 0.6, 0.6, 1), ncol = 2L)
  e_u <- .dml_mvrnorm(n_obs, rep(0, 2L), sigma_e_u)
  epsilon <- e_u[, 1L]
  u <- e_u[, 2L]
  sigma_x <- stats::toeplitz(0.5^(0:(dim_x - 1L)))
  x <- .dml_mvrnorm(n_obs, rep(0, dim_x), sigma_x)
  I_z <- diag(1, nrow = dim_z, ncol = dim_z)
  xi <- .dml_mvrnorm(n_obs, rep(0, dim_z), 0.25 * I_z)
  beta <- 1 / (1:dim_x)^2
  gamma <- beta
  delta <- 1 / (1:dim_z)^2
  zeros <- matrix(0, nrow = dim_z, ncol = dim_x - dim_z)
  Pi <- cbind(I_z, zeros)
  z <- x %*% t(Pi) + xi
  d <- as.vector(x %*% gamma + z %*% delta + u)
  y <- as.vector(alpha * d + x %*% beta + epsilon)
  colnames(x) <- paste0("X", seq_len(dim_x))
  colnames(z) <- paste0("Z", seq_len(dim_z))
  if (return_type == "matrix") {
    return(list(X = x, y = y, d = d, z = z))
  }
  dat <- data.frame(x, y = y, d = d, z)
  if (return_type == "data.frame") return(dat)
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    stop("return_type = 'DoubleMLData' requires package 'DoubleML'.", call. = FALSE)
  }
  DoubleML::DoubleMLData$new(
    dat,
    y_col = "y", d_cols = "d", x_cols = colnames(x), z_cols = colnames(z)
  )
}

#' 401(k) data (\pkg{DoubleML} \code{fetch_401k})
#'
#' @param ... Arguments passed to \code{DoubleML::fetch_401k}.
#' @return As in \pkg{DoubleML}; \code{NULL} if \pkg{DoubleML} is not installed.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_fetch_401k(...)
#' }
#' @export
doubleml_fetch_401k <- function(...) {
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    message("Package 'DoubleML' not installed. Install with: install.packages('DoubleML')")
    return(NULL)
  }
  DoubleML::fetch_401k(...)
}

#' Pennsylvania bonus experiment data (\pkg{DoubleML} \code{fetch_bonus})
#'
#' @param ... Arguments passed to \code{DoubleML::fetch_bonus}.
#' @return As in \pkg{DoubleML}; \code{NULL} if not installed.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_fetch_bonus(...)
#' }
#' @export
doubleml_fetch_bonus <- function(...) {
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    message("Package 'DoubleML' not installed. Install with: install.packages('DoubleML')")
    return(NULL)
  }
  DoubleML::fetch_bonus(...)
}

# --------------- DoubleML in DiD (change score on panel / pre-post) ---------------
# DiD outcome is delta_y = y1 - y0; treatment D can be binary (DoubleMLIRM, ATTE),
# multi-arm discrete (expanded dummies or user-supplied dummy matrix, DoubleMLPLR),
# or continuous (DoubleMLPLR). Aligns with DoubleML::DoubleMLIRM / DoubleMLPLR
# and the did package idea of group-time ATT; nuisance quality via mlr3measures.

#' @noRd
.dml_did_check_deps <- function() {
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    stop("Package 'DoubleML' is required. Install with install.packages('DoubleML').", call. = FALSE)
  }
  if (!requireNamespace("mlr3", quietly = TRUE)) {
    stop("Package 'mlr3' is required (DoubleML dependency).", call. = FALSE)
  }
  if (!requireNamespace("mlr3learners", quietly = TRUE)) {
    stop("Package 'mlr3learners' is required (DoubleML dependency).", call. = FALSE)
  }
  invisible(TRUE)
}

#' @noRd
.dml_did_prepare_D <- function(D) {
  if (inherits(D, "data.frame")) {
    if (ncol(D) > 1L) {
      D <- as.matrix(D)
    } else {
      D <- D[[1L]]
    }
  }
  if (is.matrix(D)) {
    if (ncol(D) == 1L) {
      D <- as.vector(D[, 1L])
    } else {
      Dm <- apply(D, 2L, as.numeric)
      if (is.null(colnames(Dm))) colnames(Dm) <- paste0("d", seq_len(ncol(Dm)))
      bin_cols <- vapply(seq_len(ncol(Dm)), function(j) {
        z <- Dm[, j]
        u <- unique(z[!is.na(z)])
        length(u) <= 2L && all(u %in% c(0, 1))
      }, logical(1L))
      return(list(
        D_matrix = Dm,
        engine = "PLR",
        binary_cols = bin_cols,
        was_expanded = FALSE,
        reference_level = NULL
      ))
    }
  }
  Dv <- as.numeric(D)
  ok <- !is.na(Dv)
  u <- sort(unique(Dv[ok]))
  if (length(u) < 2L) {
    stop("Treatment D must have at least two distinct non-NA values.", call. = FALSE)
  }
  if (length(u) == 2L) {
    D01 <- if (all(u %in% c(0, 1))) Dv else as.numeric(Dv == u[2L])
    Dm <- matrix(D01, ncol = 1L)
    colnames(Dm) <- "d"
    return(list(
      D_matrix = Dm,
      engine = "IRM",
      binary_cols = TRUE,
      was_expanded = FALSE,
      reference_level = NULL
    ))
  }
  multi_disc <- length(u) <= 30L && max(abs(Dv[ok] - round(Dv[ok]))) < 1e-8
  if (multi_disc) {
    fv <- factor(Dv, levels = u)
    Dm <- stats::model.matrix(~ 0 + fv)
    if (ncol(Dm) < 2L) {
      stop("Discrete multi-arm DiD requires at least two treatment levels.", call. = FALSE)
    }
    Dm <- Dm[, -1L, drop = FALSE]
    colnames(Dm) <- paste0("d", seq_len(ncol(Dm)))
    return(list(
      D_matrix = Dm,
      engine = "PLR",
      binary_cols = rep(TRUE, ncol(Dm)),
      was_expanded = TRUE,
      reference_level = u[1L]
    ))
  }
  Dm <- matrix(Dv, ncol = 1L)
  colnames(Dm) <- "d"
  list(
    D_matrix = Dm,
    engine = "PLR",
    binary_cols = FALSE,
    was_expanded = FALSE,
    reference_level = NULL
  )
}

#' @noRd
.dml_did_need_regr_m <- function(binary_cols) {
  !all(binary_cols)
}

#' @noRd
.dml_did_default_learners <- function(kind, engine, need_regr_m) {
  kind <- match.arg(kind, c("linear", "rf", "xgboost"))
  if (kind == "linear") {
    l_g <- mlr3::lrn("regr.lm")
    l_m_cl <- mlr3::lrn("classif.log_reg")
    l_m_rg <- mlr3::lrn("regr.lm")
  } else if (kind == "rf") {
    l_g <- mlr3::lrn("regr.ranger")
    l_m_cl <- mlr3::lrn("classif.ranger")
    l_m_rg <- mlr3::lrn("regr.ranger")
  } else {
    l_g <- mlr3::lrn("regr.xgboost", nrounds = 50L, verbosity = 0L)
    l_m_cl <- mlr3::lrn("classif.xgboost", nrounds = 50L, verbosity = 0L)
    l_m_rg <- mlr3::lrn("regr.xgboost", nrounds = 50L, verbosity = 0L)
  }
  if (identical(engine, "IRM")) {
    return(list(ml_g = l_g, ml_m = l_m_cl, ml_l = NULL))
  }
  list(
    ml_g = NULL,
    ml_m = if (need_regr_m) l_m_rg else l_m_cl,
    ml_l = l_g
  )
}

#' @noRd
.doubleml_did_core <- function(delta_y, D_matrix, covariates, engine, score,
                               ml_g, ml_m, ml_l,
                               n_folds, n_rep, store_predictions, ...) {
  if (inherits(covariates, "data.frame")) covariates <- as.matrix(covariates)
  dml_data <- DoubleML::double_ml_data_from_matrix(X = covariates, y = delta_y, d = D_matrix)
  if (identical(engine, "IRM")) {
    DoubleML::DoubleMLIRM$new(
      dml_data,
      ml_g = ml_g,
      ml_m = ml_m,
      score = score,
      n_folds = n_folds,
      n_rep = n_rep,
      ...
    )
  } else {
    DoubleML::DoubleMLPLR$new(
      dml_data,
      ml_l = ml_l,
      ml_m = ml_m,
      score = "partialling out",
      n_folds = n_folds,
      n_rep = n_rep,
      ...
    )
  }
}

#' @noRd
.doubleml_did_pack_result <- function(dml_obj, engine, prep) {
  coef <- as.vector(dml_obj$coef)
  nm <- names(dml_obj$coef)
  if (!is.null(nm)) names(coef) <- nm
  se <- as.vector(dml_obj$se)
  if (!is.null(nm)) names(se) <- nm
  psi <- dml_obj$psi
  out <- list(coef = coef, se = se, psi = psi, engine = engine, D_model = prep$D_matrix)
  if (identical(engine, "IRM")) {
    out$ATT <- coef[1L]
    out$att.inf.func <- as.vector(psi[, 1L, 1L])
  } else {
    out$inf_func <- psi[, 1L, , drop = FALSE]
  }
  if (isTRUE(prep$was_expanded)) {
    out$reference_level <- prep$reference_level
    out$note <- paste0(
      "Discrete treatment expanded to ", ncol(prep$D_matrix),
      " dummies vs reference level ", prep$reference_level, " (DoubleMLPLR)."
    )
  }
  if (identical(engine, "PLR")) {
    if (isTRUE(prep$was_expanded)) {
      out$mode <- "discrete_multi"
    } else if (ncol(prep$D_matrix) > 1L) {
      out$mode <- "multi_column"
    } else if (!prep$binary_cols[1L]) {
      out$mode <- "continuous"
    } else {
      out$mode <- "binary_plr"
    }
  } else {
    out$mode <- "binary"
  }
  out
}

#' @noRd
.doubleml_did_dispatch <- function(y1, y0, D, covariates, kind,
                                   ml_g = NULL, ml_m = NULL, ml_l = NULL,
                                   n_folds = 10L, n_rep = 1L, score = "ATTE",
                                   store_predictions = FALSE,
                                   return_ml_object = FALSE, ...) {
  .dml_did_check_deps()
  if (n_rep > 1L) {
    warning("n_rep > 1 is not fully supported in this wrapper; summaries use the fitted object as returned by DoubleML.", call. = FALSE)
  }
  delta_y <- as.numeric(y1) - as.numeric(y0)
  if (length(delta_y) != nrow(as.matrix(covariates))) {
    stop("length(y1), length(y0), and nrow(covariates) must match.", call. = FALSE)
  }
  prep <- .dml_did_prepare_D(D)
  engine <- prep$engine
  need_regr_m <- identical(engine, "PLR") && .dml_did_need_regr_m(prep$binary_cols)
  defs <- .dml_did_default_learners(kind, engine, need_regr_m)
  if (identical(engine, "IRM")) {
    if (is.null(ml_g)) ml_g <- defs$ml_g
    if (is.null(ml_m)) ml_m <- defs$ml_m
    ml_l <- NULL
    sc <- score
  } else {
    if (is.null(ml_l)) ml_l <- defs$ml_l
    if (is.null(ml_m)) ml_m <- defs$ml_m
    ml_g <- NULL
    sc <- "ATTE"
  }
  dml_obj <- .doubleml_did_core(
    delta_y, prep$D_matrix, covariates, engine,
    if (identical(engine, "IRM")) sc else "partialling out",
    ml_g, ml_m, ml_l, n_folds, n_rep, store_predictions, ...
  )
  dml_obj$fit(store_predictions = store_predictions)
  out <- .doubleml_did_pack_result(dml_obj, engine, prep)
  if (isTRUE(return_ml_object)) out$dml_obj <- dml_obj
  out
}

#' Evaluate cross-fitted nuisance predictions for DoubleML DiD
#'
#' Uses \code{mlr3measures::rmse} for outcome-type nuisances (\code{ml_g0}, \code{ml_g1},
#' \code{ml_l}) and \code{mlr3measures::acc} for binary propensity columns; for continuous
#' treatment columns uses RMSE between observed \eqn{D} and \eqn{\hat{m}(X)}.
#'
#' @param delta_y numeric; first-difference outcome \eqn{y_1 - y_0} (same length as rows of \code{D_model}).
#' @param D_model numeric matrix or vector; treatment used in \code{double_ml_data_from_matrix}
#'   (binary column, dummy columns for multi-arm, or continuous column).
#' @param dml_obj fitted \code{DoubleMLIRM} or \code{DoubleMLPLR} with \code{store_predictions = TRUE}.
#' @param custom_measures optional named list of functions \code{f(obs, pred)} overriding defaults
#'   for parameter names in \code{dml_obj$params_names()}.
#' @return Named list of scalar scores.
#' @references Callaway & Sant'Anna (2021) for DiD; DoubleML \url{https://docs.doubleml.org/}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_did_eval_preds(...)
#' }
#' @export
doubleml_did_eval_preds <- function(delta_y, D_model, dml_obj, custom_measures = NULL) {
  if (!requireNamespace("mlr3measures", quietly = TRUE)) {
    stop("Package 'mlr3measures' is required for doubleml_did_eval_preds.", call. = FALSE)
  }
  predictions <- dml_obj$predictions
  param_names <- dml_obj$params_names()
  measures_res <- list()
  custom_measures <- if (is.null(custom_measures)) list() else custom_measures

  if (inherits(dml_obj, "DoubleMLIRM")) {
    Dv <- if (is.matrix(D_model)) as.numeric(D_model[, 1L]) else as.numeric(D_model)
    is_atte <- is.character(dml_obj$score) && identical(dml_obj$score, "ATTE")
    for (pn in param_names) {
      cust <- custom_measures[[pn]]
      pr <- as.vector(predictions[[pn]][, 1L, 1L])
      if (pn == "ml_m") {
        truth <- factor(Dv, levels = c(0, 1))
        response <- factor(as.integer(pr > 0.5), levels = c(0, 1))
        measures_res[[pn]] <- if (is.function(cust)) cust(truth, response) else mlr3measures::acc(truth, response)
      } else if (pn == "ml_g0") {
        idx <- Dv == 0
        measures_res[[pn]] <- if (is.function(cust)) {
          cust(delta_y[idx], pr[idx])
        } else {
          mlr3measures::rmse(delta_y[idx], pr[idx])
        }
      } else if (pn == "ml_g1") {
        if (is_atte && all(is.na(pr))) next
        idx <- Dv == 1
        if (!any(idx)) next
        measures_res[[pn]] <- if (is.function(cust)) {
          cust(delta_y[idx], pr[idx])
        } else {
          mlr3measures::rmse(delta_y[idx], pr[idx])
        }
      }
    }
    return(measures_res)
  }

  if (!inherits(dml_obj, "DoubleMLPLR")) {
    stop("dml_obj must be a fitted DoubleMLIRM or DoubleMLPLR object.", call. = FALSE)
  }
  Dmat <- if (is.matrix(D_model)) D_model else matrix(D_model, ncol = 1L)
  n_treat <- dml_obj$data$n_treat
  dcols <- dml_obj$data$d_cols
  for (pn in param_names) {
    for (jt in seq_len(n_treat)) {
      pr <- as.vector(predictions[[pn]][, 1L, jt])
      key <- if (n_treat > 1L) paste0(pn, "[", dcols[jt], "]") else pn
      dj <- as.numeric(Dmat[, min(jt, ncol(Dmat))])
      cust <- custom_measures[[key]]
      if (is.null(cust)) cust <- custom_measures[[pn]]
      if (pn == "ml_l") {
        measures_res[[key]] <- if (is.function(cust)) cust(delta_y, pr) else mlr3measures::rmse(delta_y, pr)
      } else if (pn == "ml_m") {
        udj <- unique(dj[!is.na(dj)])
        bin_j <- length(udj) <= 2L && all(udj %in% c(0, 1))
        if (bin_j) {
          truth <- factor(dj, levels = c(0, 1))
          response <- factor(as.integer(pr > 0.5), levels = c(0, 1))
          measures_res[[key]] <- if (is.function(cust)) cust(truth, response) else mlr3measures::acc(truth, response)
        } else {
          measures_res[[key]] <- if (is.function(cust)) cust(dj, pr) else mlr3measures::rmse(dj, pr)
        }
      }
    }
  }
  measures_res
}

#' @rdname doubleml_did_linear
#' @order 1
#' @title DoubleML DiD wrappers (linear / ranger / xgboost nuisances)
#'
#' @description
#' Estimates a first-difference outcome \eqn{Y = Y_1 - Y_0} with DoubleML. Binary treatment
#' uses \code{DoubleML::DoubleMLIRM} with \code{score = "ATTE"} (default). Multi-arm discrete
#' (more than two levels, encoded as integers) is expanded to dummies versus the smallest level
#' and estimated with \code{DoubleML::DoubleMLPLR}. User-supplied dummy matrices or continuous
#' \eqn{D} use \code{DoubleMLPLR}. Defaults follow the usual DiD + DML mapping (see Callaway &
#' Sant'Anna 2021; DoubleML documentation).
#'
#' @param y1,y0 post and pre outcomes (same length).
#' @param D treatment: vector (binary, discrete multi-arm, or continuous), or matrix of
#'   treatment columns / dummies (\code{ncol > 1} uses PLR on all columns).
#' @param covariates matrix or \code{data.frame} of pre-treatment covariates.
#' @param ml_g,ml_m,ml_l \code{mlr3} learners; if \code{NULL}, sensible defaults for the wrapper.
#' @param n_folds,n_rep passed to DoubleML (see note on \code{n_rep}).
#' @param score \code{"ATTE"} or \code{"ATE"} for the IRM branch only.
#' @param print_eval if \code{TRUE}, print nuisance diagnostics from \code{\link{doubleml_did_eval_preds}}.
#' @param return_ml_object if \code{TRUE}, include \code{dml_obj} in the return list.
#' @param ... passed to \code{DoubleMLIRM$new} or \code{DoubleMLPLR$new}.
#' @return List with \code{coef}, \code{se}, \code{psi}, \code{mode}, \code{D_model}, and
#'   \code{ATT} / \code{att.inf.func} (binary IRM) or \code{inf_func} (PLR). Eval variants add
#'   \code{eval_predictions}.
#' @references Callaway & Sant'Anna (2021); Bach et al. (2021) \doi{10.48550/arXiv.2103.09603}; DoubleML \url{https://docs.doubleml.org/}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_did_linear(...)
#' }
#' @export
doubleml_did_linear <- function(y1, y0, D, covariates,
                                ml_g = NULL, ml_m = NULL, ml_l = NULL,
                                n_folds = 10L, n_rep = 1L, score = "ATTE",
                                return_ml_object = FALSE, ...) {
  .doubleml_did_dispatch(
    y1, y0, D, covariates, kind = "linear",
    ml_g = ml_g, ml_m = ml_m, ml_l = ml_l,
    n_folds = n_folds, n_rep = n_rep, score = score,
    store_predictions = FALSE, return_ml_object = return_ml_object, ...
  )
}

#' @rdname doubleml_did_linear
#' @order 2
#' @return
#' Object returned by \code{doubleml_did_rf}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_did_rf(...)
#' }
#' @export
doubleml_did_rf <- function(y1, y0, D, covariates,
                            ml_g = NULL, ml_m = NULL, ml_l = NULL,
                            n_folds = 10L, n_rep = 1L, score = "ATTE",
                            return_ml_object = FALSE, ...) {
  .doubleml_did_dispatch(
    y1, y0, D, covariates, kind = "rf",
    ml_g = ml_g, ml_m = ml_m, ml_l = ml_l,
    n_folds = n_folds, n_rep = n_rep, score = score,
    store_predictions = FALSE, return_ml_object = return_ml_object, ...
  )
}

#' @rdname doubleml_did_linear
#' @order 3
#' @return
#' Object returned by \code{doubleml_did_xgboost}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_did_xgboost(...)
#' }
#' @export
doubleml_did_xgboost <- function(y1, y0, D, covariates,
                                 ml_g = NULL, ml_m = NULL, ml_l = NULL,
                                 n_folds = 10L, n_rep = 1L, score = "ATTE",
                                 return_ml_object = FALSE, ...) {
  .doubleml_did_dispatch(
    y1, y0, D, covariates, kind = "xgboost",
    ml_g = ml_g, ml_m = ml_m, ml_l = ml_l,
    n_folds = n_folds, n_rep = n_rep, score = score,
    store_predictions = FALSE, return_ml_object = return_ml_object, ...
  )
}

#' @rdname doubleml_did_linear
#' @order 4
#' @return
#' Object returned by \code{doubleml_did_eval_linear}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_did_eval_linear(...)
#' }
#' @export
doubleml_did_eval_linear <- function(y1, y0, D, covariates,
                                     ml_g = NULL, ml_m = NULL, ml_l = NULL,
                                     n_folds = 10L, n_rep = 1L, score = "ATTE",
                                     print_eval = TRUE, return_ml_object = FALSE, ...) {
  out <- .doubleml_did_dispatch(
    y1, y0, D, covariates, kind = "linear",
    ml_g = ml_g, ml_m = ml_m, ml_l = ml_l,
    n_folds = n_folds, n_rep = n_rep, score = score,
    store_predictions = TRUE, return_ml_object = TRUE, ...
  )
  delta_y <- as.numeric(y1) - as.numeric(y0)
  ev <- doubleml_did_eval_preds(delta_y, out$D_model, out$dml_obj)
  if (isTRUE(print_eval)) print(ev)
  out$eval_predictions <- ev
  if (!isTRUE(return_ml_object)) out$dml_obj <- NULL
  out
}

#' @rdname doubleml_did_linear
#' @order 5
#' @return
#' Object returned by \code{doubleml_did_eval_rf}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_did_eval_rf(...)
#' }
#' @export
doubleml_did_eval_rf <- function(y1, y0, D, covariates,
                                   ml_g = NULL, ml_m = NULL, ml_l = NULL,
                                   n_folds = 10L, n_rep = 1L, score = "ATTE",
                                   print_eval = TRUE, return_ml_object = FALSE, ...) {
  out <- .doubleml_did_dispatch(
    y1, y0, D, covariates, kind = "rf",
    ml_g = ml_g, ml_m = ml_m, ml_l = ml_l,
    n_folds = n_folds, n_rep = n_rep, score = score,
    store_predictions = TRUE, return_ml_object = TRUE, ...
  )
  delta_y <- as.numeric(y1) - as.numeric(y0)
  ev <- doubleml_did_eval_preds(delta_y, out$D_model, out$dml_obj)
  if (isTRUE(print_eval)) print(ev)
  out$eval_predictions <- ev
  if (!isTRUE(return_ml_object)) out$dml_obj <- NULL
  out
}

#' @rdname doubleml_did_linear
#' @order 6
#' @return
#' Object returned by \code{doubleml_did_eval_xgboost}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # doubleml_did_eval_xgboost(...)
#' }
#' @export
doubleml_did_eval_xgboost <- function(y1, y0, D, covariates,
                                      ml_g = NULL, ml_m = NULL, ml_l = NULL,
                                      n_folds = 10L, n_rep = 1L, score = "ATTE",
                                      print_eval = TRUE, return_ml_object = FALSE, ...) {
  out <- .doubleml_did_dispatch(
    y1, y0, D, covariates, kind = "xgboost",
    ml_g = ml_g, ml_m = ml_m, ml_l = ml_l,
    n_folds = n_folds, n_rep = n_rep, score = score,
    store_predictions = TRUE, return_ml_object = TRUE, ...
  )
  delta_y <- as.numeric(y1) - as.numeric(y0)
  ev <- doubleml_did_eval_preds(delta_y, out$D_model, out$dml_obj)
  if (isTRUE(print_eval)) print(ev)
  out$eval_predictions <- ev
  if (!isTRUE(return_ml_object)) out$dml_obj <- NULL
  out
}
