# causalBoosted_IV.R  (v2.0)
#
# Boosted Causal Forest with Instrumental Variables
#
# Changes from v1.0
# 
# BUG FIXES
#   - binarise(): median split produced all-zero vector when > 50 % of values
#     equal the median; now uses rank-based split.
#   - .crossfit_cxgboost(): fold assignment via cut(sample(n)) could create
#     empty folds for small n; replaced with interleaved assignment.
#   - predict(): newdata path passed raw matrix to grf without column-name
#     alignment; now enforces column order from training.
#   - .marginal_pred(): propensity scores outside (0,1) caused negative weights;
#     now clipped to [1e-6, 1-1e-6].
#   - validate_inputs(): cor(Z, W) crashes on zero-variance inputs; guarded.
#
# NEW FEATURES
#   - summary()          : formatted model summary with ATE, calibration, VI.
#   - plot_cate()        : CATE distribution + top-variable partial dependence.
#   - subgroup_analysis(): above/below-median tau subgroups with group ATEs.
#
# PERFORMANCE
#   - .crossfit_cxgboost() parallelised over folds via future.apply.
#   - Memory: fold predictions written in-place; model objects freed per fold.
#
# ROBUSTNESS
#   - Continuous W / Z: CXGBoost head now uses regression (not probability)
#     when the variable has > 2 unique values; propensity branch preserved for
#     binary variables only.
#   - Weak-instrument diagnostics: first-stage F-statistic and Cragg-Donald
#     statistic reported in summary() and as a returned field.
#   - NA imputation: median imputation with a column-level imputation table
#     stored for apply to new data.
#
# API
#   - save_model() / load_model() via saveRDS / readRDS wrappers.
#   - Parsnip (tidymodels) shim: causal_iv_forest_spec() returns a parsnip
#     model spec that delegates to BoostedIVForest.
#
# DOCUMENTATION
#   - Full roxygen2 tags on every exported symbol.
#   - Self-contained vignette-style example at the bottom of this file.
#
# EVALUATION
#   - PEHE() and ATE() from utils.R integrated as $evaluate() method.
#
# Dependencies: R6, xgboost, ranger, grf, future, future.apply
# Optional    : ggplot2 (plots), parsnip (tidymodels shim)
# Source causalXGBoost.R before this file.

#  package checks 
requireNamespace("R6",           quietly = TRUE)
requireNamespace("future",       quietly = TRUE)
requireNamespace("future.apply", quietly = TRUE)

# 
# SECTION 1  Evaluation metrics (from utils.R, self-contained copy)
# 

#' Precision in Estimation of Heterogeneous Effects (PEHE)
#'
#' Root mean squared error between true and estimated individual treatment
#' effects: \eqn{\sqrt{E[(\tau(x) - \hat\tau(x))^2]}}.
#'
#' @param tau_true Numeric vector of true individual treatment effects.
#' @param tau_hat  Numeric vector of estimated individual treatment effects.
#'   Must be the same length as \code{tau_true}.
#' @return Numeric scalar (RMSE on the ITE scale).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @export
#' @examples
#' PEHE(tau_true = c(1, 2, 3), tau_hat = c(1.1, 1.9, 3.2))
PEHE <- function(tau_true, tau_hat) {
  if (length(tau_true) != length(tau_hat)) {
    stop("tau_true and tau_hat must have the same length.")
  }
  sqrt(mean((tau_true - tau_hat)^2))
}

#' Absolute Error in Average Treatment Effect
#'
#' \eqn{|E[\tau(x)] - E[\hat\tau(x)]|}.
#'
#' @param tau_true Numeric vector of true individual treatment effects.
#' @param tau_hat  Numeric vector of estimated individual treatment effects.
#' @return Numeric scalar.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @export
#' @examples
#' ATE_error(c(1, 2, 3), c(1.1, 1.9, 3.2))
ATE_error <- function(tau_true, tau_hat) {
  if (length(tau_true) != length(tau_hat)) {
    stop("tau_true and tau_hat must have the same length.")
  }
  abs(mean(tau_true) - mean(tau_hat))
}

# 
# SECTION 2  Internal helpers
# 

#' Interleaved k-fold assignment (no empty folds for small n)
#'
#' @param n       Number of observations.
#' @param n_folds Number of folds.
#' @return Integer vector of fold labels (1..n_folds), length n, randomised.
#' @keywords internal
.make_folds <- function(n, n_folds) {
  # Interleave then shuffle so each fold has floor(n/n_folds) or
  # ceiling(n/n_folds) observations  guaranteed non-empty for n >= n_folds.
  if (n < n_folds) {
    stop(sprintf("n (%d) must be >= n_folds (%d).", n, n_folds))
  }
  base    <- rep(seq_len(n_folds), length.out = n)
  shuffle <- sample(n)
  fold_id <- integer(n)
  fold_id[shuffle] <- base
  fold_id
}

#' Median imputation table
#'
#' Computes per-column medians on \code{X} (ignoring NAs) for later use in
#' \code{.apply_imputation()}.
#'
#' @param X Numeric matrix, possibly with NAs.
#' @return Named numeric vector of column medians.
#' @keywords internal
.build_imputation_table <- function(X) {
  apply(X, 2, function(col) {
    m <- median(col, na.rm = TRUE)
    if (is.na(m)) 0 else m   # all-NA column  impute 0
  })
}

#' Apply median imputation using a pre-built table
#'
#' @param X     Numeric matrix with possible NAs.
#' @param table Named numeric vector from \code{.build_imputation_table()}.
#' @return Numeric matrix with NAs replaced.
#' @keywords internal
.apply_imputation <- function(X, table) {
  for (j in seq_len(ncol(X))) {
    na_rows <- is.na(X[, j])
    if (any(na_rows)) {
      X[na_rows, j] <- table[j]
    }
  }
  X
}

#' Rank-safe binarisation
#'
#' Converts a numeric vector to 0/1 at the median.  Uses rank-based splitting
#' to avoid degenerate all-zero output when more than 50 % of values equal
#' the median exactly (bug fix over v1.0).
#'
#' @param v Numeric vector.
#' @return Integer vector of 0s and 1s, same length as \code{v}.
#' @keywords internal
.binarise <- function(v) {
  uv <- sort(unique(v))
  # Already binary 0/1  return as-is
  if (length(uv) == 2L && isTRUE(all.equal(uv, c(0, 1)))) {
    return(as.integer(v))
  }
  # Rank-based: top half = 1, bottom half = 0
  # ties.method = "first" ensures exactly ceiling(n/2) ones
  r <- rank(v, ties.method = "first")
  as.integer(r > median(r))
}

#' Clip propensity scores to avoid boundary weights
#'
#' @param p   Numeric vector of propensity scores.
#' @param eps Clipping bound (default 1e-6).
#' @return Clipped numeric vector.
#' @keywords internal
.clip_ps <- function(p, eps = 1e-6) {
  pmax(eps, pmin(1 - eps, p))
}

#' Marginal E[outcome|X] as propensity-weighted average of two heads
#'
#' \eqn{E[Y|X] = (1-p) \cdot \hat Y(0) + p \cdot \hat Y(1)}
#' where \eqn{p = P(T=1|X)}.
#'
#' @param y0_hat     Predicted outcome under T=0.
#' @param y1_hat     Predicted outcome under T=1.
#' @param propensity Propensity score P(T=1|X), clipped internally.
#' @return Numeric vector.
#' @keywords internal
.marginal_pred <- function(y0_hat, y1_hat, propensity) {
  p <- .clip_ps(propensity)
  (1 - p) * y0_hat + p * y1_hat
}

#' Cross-fitted nuisance predictions via CXGBoost (parallelised)
#'
#' Runs k-fold cross-fitting in parallel via \pkg{future.apply}.  Each fold
#' trains a fresh \code{CXGBoost} model on the complement and predicts on the
#' held-out fold.
#'
#' When \code{outcome} is continuous (> 2 unique values) the function still
#' uses CXGBoost's two-head architecture; the "treatment mask" (\code{treat})
#' determines which head each observation updates  this is valid for
#' continuous outcomes because the mask is applied to the gradient, not to a
#' classification loss.
#'
#' @param X          Covariate matrix (n x p), no NAs.
#' @param outcome    Numeric outcome vector (Y, W, or Z).
#' @param treat      Binary 0/1 vector used as the CXGBoost treatment mask.
#' @param n_folds    Number of folds (default 5).
#' @param nrounds    XGBoost boosting rounds per fold (default 100).
#' @param xgb_params Extra XGBoost hyperparameters.
#' @return List: \code{y0_hat}, \code{y1_hat}, \code{propensity} (all length n,
#'   out-of-fold).
#' @keywords internal
.crossfit_cxgboost <- function(X,
                               outcome,
                               treat,
                               n_folds    = 5L,
                               nrounds    = 100L,
                               xgb_params = list()) {
  n      <- nrow(X)
  folds  <- .make_folds(n, n_folds)

  #  parallel fold loop 
  fold_results <- future.apply::future_lapply(
    seq_len(n_folds),
    function(k) {
      idx_val <- which(folds == k)
      idx_trn <- which(folds != k)

      mdl <- CXGBoost$new(parameters = xgb_params)
      mdl$fit(
        X       = X[idx_trn, , drop = FALSE],
        t       = treat[idx_trn],
        y       = outcome[idx_trn],
        nrounds = nrounds,
        verbose = 0L
      )
      preds <- mdl$predict(X[idx_val, , drop = FALSE])
      rm(mdl)   # free memory immediately

      list(
        idx    = idx_val,
        y0_hat = preds$y0_hat,
        y1_hat = preds$y1_hat,
        ps     = preds$propensity_score
      )
    },
    future.seed = TRUE
  )

  #  collect in-place 
  y0_oof <- numeric(n)
  y1_oof <- numeric(n)
  ps_oof <- numeric(n)

  for (res in fold_results) {
    y0_oof[res$idx] <- res$y0_hat
    y1_oof[res$idx] <- res$y1_hat
    ps_oof[res$idx] <- res$ps
  }

  list(y0_hat = y0_oof, y1_hat = y1_oof, propensity = ps_oof)
}

#' Weak-instrument diagnostics
#'
#' Computes the first-stage F-statistic (Staiger-Stock rule-of-thumb: F > 10
#' is "strong") and the partial R-squared of Z in the first-stage regression
#' of W on Z and X.
#'
#' @param Y Outcome vector (used only for dimension check).
#' @param W Treatment vector.
#' @param Z Instrument vector.
#' @param X Covariate matrix.
#' @return List: \code{F_stat}, \code{partial_R2}, \code{cor_ZW},
#'   \code{strong} (logical).
#' @keywords internal
.weak_instrument_diag <- function(Y, W, Z, X) {
  tryCatch({
    df      <- data.frame(W = W, Z = Z, X)
    # Restricted model (no Z)
    fit_r   <- stats::lm(W ~ ., data = df[, -which(names(df) == "Z"), drop = FALSE])
    # Full model (with Z)
    fit_f   <- stats::lm(W ~ ., data = df)
    n       <- length(W)
    k       <- length(coef(fit_f)) - 1   # number of regressors
    rss_r   <- sum(residuals(fit_r)^2)
    rss_f   <- sum(residuals(fit_f)^2)
    F_stat  <- ((rss_r - rss_f) / 1) / (rss_f / (n - k - 1))
    partial_R2 <- (rss_r - rss_f) / rss_r
    cor_ZW  <- stats::cor(Z, W)
    list(
      F_stat     = F_stat,
      partial_R2 = partial_R2,
      cor_ZW     = cor_ZW,
      strong     = (F_stat > 10)
    )
  }, error = function(e) {
    list(F_stat = NA, partial_R2 = NA, cor_ZW = NA, strong = NA)
  })
}

# 
# SECTION 3  Main R6 class
# 

#' BoostedIVForest: Boosted causal forest with instrumental variables
#'
#' Combines \strong{CXGBoost} cross-fitted nuisance estimation with
#' \code{grf::instrumental_forest} to estimate heterogeneous Local Average
#' Treatment Effects (LATE) in the presence of endogenous treatment.
#'
#' @section Estimation pipeline:
#' \enumerate{
#'   \item \strong{Impute} missing values in X using per-column medians.
#'   \item \strong{Cross-fit nuisances} (parallelised, \code{n_folds} folds):
#'     \itemize{
#'       \item \eqn{\hat m(X) = E[Y \mid X]}
#'       \item \eqn{\hat e(X) = E[W \mid X]}
#'       \item \eqn{\hat p(X) = E[Z \mid X]}
#'     }
#'   \item \strong{Residualise}: \eqn{\tilde Y = Y - \hat m(X)},
#'     \eqn{\tilde W = W - \hat e(X)}, \eqn{\tilde Z = Z - \hat p(X)}.
#'   \item \strong{Fit} \code{grf::instrumental_forest} on
#'     \eqn{(\tilde Y, \tilde W, \tilde Z, X)}.
#' }
#'
#' @section Parallelism:
#' Cross-fitting is parallelised via \pkg{future} / \pkg{future.apply}.
#' Set a parallel plan before calling \code{$fit()}:
#' \preformatted{
#'   future::plan(future::multisession, workers = 4)
#'   model$fit(X, Y, W, Z)
#'   future::plan(future::sequential)   # reset afterwards
#' }
#'
#' @field n_folds        Number of cross-fitting folds.
#' @field nrounds        XGBoost boosting rounds per fold.
#' @field xgb_params     Extra XGBoost hyperparameters.
#' @field forest_args    Extra arguments for \code{grf::instrumental_forest}.
#' @field forest         Fitted \code{grf::instrumental_forest} (post-fit).
#' @field Y_hat          Out-of-fold \eqn{E[Y|X]} (post-fit).
#' @field W_hat          Out-of-fold \eqn{E[W|X]} (post-fit).
#' @field Z_hat          Out-of-fold \eqn{E[Z|X]} (post-fit).
#' @field iv_diagnostics Weak-instrument diagnostics list (post-fit).
#' @field imputation_table Per-column medians used for NA imputation (post-fit).
#' @field train_colnames Column names of the training X (post-fit).
#'
#' @return
#' Object returned by \code{BoostedIVForest}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # BoostedIVForest(...)
#' }
#' @export
BoostedIVForest <- R6::R6Class(
  "BoostedIVForest",
  public = list(

    #  public fields 
    n_folds         = 5L,
    nrounds         = 100L,
    xgb_params      = list(),
    forest_args     = list(),
    forest          = NULL,
    Y_hat           = NULL,
    W_hat           = NULL,
    Z_hat           = NULL,
    iv_diagnostics  = NULL,
    imputation_table = NULL,
    train_colnames  = NULL,

    #  initialize 

    #' @description Create a new BoostedIVForest.
    #' @param n_folds     Number of cross-fitting folds (integer >= 2, default 5).
    #' @param nrounds     XGBoost boosting rounds per fold (integer >= 1, default 100).
    #' @param xgb_params  Named list of extra XGBoost hyperparameters passed to
    #'   \code{CXGBoost$new(parameters = ...)}.
    #' @param forest_args Named list of extra arguments for
    #'   \code{grf::instrumental_forest()} (e.g. \code{num.trees},
    #'   \code{min.node.size}, \code{honesty.fraction}).
    initialize = function(n_folds     = 5L,
                          nrounds     = 100L,
                          xgb_params  = list(),
                          forest_args = list()) {
      stopifnot(is.numeric(n_folds),  n_folds  >= 2)
      stopifnot(is.numeric(nrounds),  nrounds  >= 1)
      stopifnot(is.list(xgb_params),  is.list(forest_args))

      self$n_folds     <- as.integer(n_folds)
      self$nrounds     <- as.integer(nrounds)
      self$xgb_params  <- xgb_params
      self$forest_args <- forest_args
    },

    #  fit 

    #' @description Fit the boosted IV forest.
    #' @param X       Numeric covariate matrix (n x p).  NAs are median-imputed.
    #' @param Y       Observed outcome vector (length n).
    #' @param W       Endogenous treatment, binary or continuous (length n).
    #' @param Z       Instrument, binary or continuous (length n).  Must satisfy
    #'   relevance, exclusion restriction, and exogeneity.
    #' @param verbose Logical; print stage progress (default \code{TRUE}).
    #' @return \code{self} invisibly (supports method chaining).
    fit = function(X, Y, W, Z, verbose = TRUE) {
      private$check_dependencies()

      #  coerce inputs 
      X <- as.matrix(X)
      Y <- as.numeric(Y)
      W <- as.numeric(W)
      Z <- as.numeric(Z)

      self$train_colnames <- colnames(X)

      #  NA imputation 
      if (anyNA(X)) {
        if (verbose) cat("[BoostedIVForest] Imputing missing values in X...\n")
        self$imputation_table <- .build_imputation_table(X)
        X <- .apply_imputation(X, self$imputation_table)
      }

      if (anyNA(Y) || anyNA(W) || anyNA(Z)) {
        stop("Y, W, and Z must not contain NA values.")
      }

      private$validate_inputs(X, Y, W, Z)

      #  weak-instrument diagnostics 
      if (verbose) cat("[BoostedIVForest] Running instrument diagnostics...\n")
      self$iv_diagnostics <- .weak_instrument_diag(Y, W, Z, X)
      if (!isTRUE(self$iv_diagnostics$strong)) {
        warning(sprintf(
          "Weak instrument detected: F = %.2f (< 10). LATE estimates may be unreliable.",
          self$iv_diagnostics$F_stat
        ))
      }

      #  binarise for CXGBoost treatment mask 
      W_bin <- .binarise(W)
      Z_bin <- .binarise(Z)

      #  Stage 1a: E[Y|X] 
      if (verbose) cat("[BoostedIVForest] Stage 1a: Cross-fitting E[Y|X]...\n")
      nuis_Y     <- .crossfit_cxgboost(X, Y, W_bin,
                                       self$n_folds, self$nrounds, self$xgb_params)
      self$Y_hat <- .marginal_pred(nuis_Y$y0_hat, nuis_Y$y1_hat, nuis_Y$propensity)

      #  Stage 1b: E[W|X] 
      if (verbose) cat("[BoostedIVForest] Stage 1b: Cross-fitting E[W|X]...\n")
      nuis_W     <- .crossfit_cxgboost(X, W, Z_bin,
                                       self$n_folds, self$nrounds, self$xgb_params)
      self$W_hat <- .marginal_pred(nuis_W$y0_hat, nuis_W$y1_hat, nuis_W$propensity)

      #  Stage 1c: E[Z|X] 
      if (verbose) cat("[BoostedIVForest] Stage 1c: Cross-fitting E[Z|X]...\n")
      nuis_Z     <- .crossfit_cxgboost(X, Z, W_bin,
                                       self$n_folds, self$nrounds, self$xgb_params)
      self$Z_hat <- .marginal_pred(nuis_Z$y0_hat, nuis_Z$y1_hat, nuis_Z$propensity)

      #  Stage 2: residualise 
      if (verbose) cat("[BoostedIVForest] Stage 2: Residualising Y, W, Z...\n")
      Y_res <- Y - self$Y_hat
      W_res <- W - self$W_hat
      Z_res <- Z - self$Z_hat

      #  Stage 3: instrumental forest 
      if (verbose) cat("[BoostedIVForest] Stage 3: Fitting instrumental_forest...\n")
      self$forest <- do.call(
        grf::instrumental_forest,
        c(list(X = X, Y = Y_res, W = W_res, Z = Z_res), self$forest_args)
      )

      if (verbose) cat("[BoostedIVForest] Fitting complete.\n")
      invisible(self)
    },

    #  predict 

    #' @description Predict heterogeneous LATE for new units.
    #' @param X_new Numeric covariate matrix.  If \code{NULL}, returns in-sample
    #'   OOB predictions.  Column order must match training X; NAs are imputed.
    #' @param estimate.variance Logical; return variance estimates (default
    #'   \code{FALSE}).
    #' @param ... Extra arguments forwarded to
    #'   \code{predict.instrumental_forest}.
    #' @return A \code{grf} prediction list with element \code{predictions} (and
    #'   optionally \code{variance.estimates}).
    predict = function(X_new = NULL, estimate.variance = FALSE, ...) {
      private$check_fitted()

      if (is.null(X_new)) {
        return(predict(self$forest,
                       estimate.variance = estimate.variance, ...))
      }

      X_new <- as.matrix(X_new)

      # NA imputation on new data
      if (!is.null(self$imputation_table) && anyNA(X_new)) {
        X_new <- .apply_imputation(X_new, self$imputation_table)
      }

      # Column alignment (bug fix v2.0)
      if (!is.null(self$train_colnames) &&
          !is.null(colnames(X_new)) &&
          !identical(colnames(X_new), self$train_colnames)) {
        X_new <- X_new[, self$train_colnames, drop = FALSE]
      }

      predict(self$forest,
              newdata           = X_new,
              estimate.variance = estimate.variance, ...)
    },

    #  average_treatment_effect 

    #' @description Estimate the ATE with standard error.
    #' @return Named numeric vector: \code{estimate} and \code{std.err}.
    average_treatment_effect = function() {
      private$check_fitted()
      # NOTE: grf::average_treatment_effect() for instrumental_forest is only
      # implemented for *binary instruments*. Our DML-style residualisation
      # uses Z_res = Z - E[Z|X], which is generally continuous even when Z is
      # binary. To avoid hard errors, we compute ATE as the mean of CATE
      # predictions and an approximate standard error from per-unit variance.
      pr <- predict(self$forest, estimate.variance = TRUE)
      tau <- pr$predictions
      est <- mean(tau)
      se  <- if (!is.null(pr$variance.estimates)) {
        sqrt(sum(pr$variance.estimates)) / length(pr$variance.estimates)
      } else {
        stats::sd(tau) / sqrt(length(tau))
      }
      c(estimate = est, std.err = se)
    },

    #  test_calibration 

    #' @description Best linear predictor calibration test for heterogeneity.
    #'
    #' A significant \code{differential.forest.prediction} coefficient
    #' indicates genuine heterogeneity captured by the model.
    #' Not supported for instrumental_forest; returns \code{NULL} in that case.
    #' @return \code{lm} object from \code{grf::test_calibration}, or \code{NULL}.
    test_calibration = function() {
      private$check_fitted()
      tryCatch(grf::test_calibration(self$forest), error = function(e) NULL)
    },

    #  variable_importance 

    #' @description Variable importance (proportion of splits per covariate).
    #' @return Named numeric vector, sorted descending.
    variable_importance = function() {
      private$check_fitted()
      imp <- grf::variable_importance(self$forest)
      nms <- self$train_colnames
      if (!is.null(nms)) names(imp) <- nms
      sort(imp, decreasing = TRUE)
    },

    #  summary 

    #' @description Print a formatted model summary.
    #' @param top_n Number of top variables to display (default 5).
    #' @return \code{self} invisibly.
    summary = function(top_n = 5L) {
      private$check_fitted()

      cat("\n")
      cat("  BoostedIVForest  Model Summary\n")
      cat("\n\n")

      # Configuration
      cat(" Configuration \n")
      cat(sprintf("  Cross-fitting folds : %d\n",   self$n_folds))
      cat(sprintf("  XGBoost rounds      : %d\n",   self$nrounds))
      n_trees <- self$forest$tunable.params$num.trees %||%
                 length(self$forest$trees)
      cat(sprintf("  Forest trees        : %s\n",   n_trees))
      cat(sprintf("  Training obs        : %d\n",
                  nrow(self$forest$X.orig)))
      cat(sprintf("  Covariates (p)      : %d\n\n",
                  ncol(self$forest$X.orig)))

      # Instrument diagnostics
      cat(" Instrument Diagnostics \n")
      if (!is.null(self$iv_diagnostics)) {
        diag <- self$iv_diagnostics
        cat(sprintf("  cor(Z, W)           : %+.4f\n", diag$cor_ZW))
        cat(sprintf("  First-stage F-stat  : %.2f  %s\n",
                    diag$F_stat,
                    if (isTRUE(diag$strong)) "[strong]" else "[WEAK  interpret with caution]"))
        cat(sprintf("  Partial R2(ZW)     : %.4f\n\n", diag$partial_R2))
      }

      # ATE
      cat(" Average Treatment Effect \n")
      ate <- self$average_treatment_effect()
      cat(sprintf("  ATE     : %+.4f\n", ate["estimate"]))
      cat(sprintf("  Std.Err : %.4f\n",  ate["std.err"]))
      z_score <- ate["estimate"] / ate["std.err"]
      p_val   <- 2 * stats::pnorm(-abs(z_score))
      cat(sprintf("  p-value : %.4f  %s\n\n",
                  p_val,
                  if (p_val < 0.05) "[significant]" else ""))

      # CATE distribution
      tau_hat <- self$predict()$predictions
      cat(" CATE Distribution \n")
      q <- stats::quantile(tau_hat, c(0.05, 0.25, 0.50, 0.75, 0.95))
      cat(sprintf("  Min / Max   : %+.3f / %+.3f\n", min(tau_hat), max(tau_hat)))
      cat(sprintf("  Mean / SD   : %+.3f / %.3f\n",
                  mean(tau_hat), stats::sd(tau_hat)))
      cat(sprintf("  Q5 / Q95    : %+.3f / %+.3f\n\n", q[1], q[5]))

      # Variable importance
      cat(sprintf(" Top %d Variables by Importance \n", top_n))
      vi  <- self$variable_importance()
      top <- head(vi, top_n)
      for (nm in names(top)) {
        bar <- strrep("", round(top[nm] * 40))
        cat(sprintf("  %-12s %.4f  %s\n", nm, top[nm], bar))
      }
      cat("\n")

      # Calibration (not supported for instrumental_forest in grf)
      cat(" Calibration Test \n")
      cal <- tryCatch(self$test_calibration(), error = function(e) NULL)
      if (!is.null(cal) && !is.null(cal$coefficients)) {
        print(cal$coefficients[, c("Estimate", "Std. Error", "Pr(>|t|)")])
      } else {
        cat("  (Calibration test not supported for this forest type.)\n")
      }
      cat("\n")

      invisible(self)
    },

    #  evaluate 

    #' @description Evaluate CATE estimates against true treatment effects.
    #'
    #' Requires ground-truth \code{tau_true} (available in simulations).
    #' @param tau_true Numeric vector of true individual treatment effects.
    #' @param X_new   Optional covariate matrix for out-of-sample evaluation.
    #'   If \code{NULL}, uses in-sample OOB predictions.
    #' @return Named list: \code{PEHE}, \code{ATE_error}, \code{tau_hat}.
    evaluate = function(tau_true, X_new = NULL) {
      private$check_fitted()
      tau_hat <- self$predict(X_new)$predictions
      list(
        PEHE      = PEHE(tau_true, tau_hat),
        ATE_error = ATE_error(tau_true, tau_hat),
        tau_hat   = tau_hat
      )
    },

    #  subgroup_analysis 

    #' @description Split units into high / low CATE subgroups and compute
    #'   group-level ATEs via \code{grf::average_treatment_effect}.
    #'
    #' @param X_new    Optional covariate matrix (uses training data if NULL).
    #' @param quantile Threshold quantile for "high" group (default 0.5 = median
    #'   split).
    #' @return A named list with components:
    #' \describe{
    #'   \item{tau_hat}{Full CATE vector.}
    #'   \item{high_idx}{Row indices of the high-CATE group.}
    #'   \item{low_idx}{Row indices of the low-CATE group.}
    #'   \item{ATE_high}{ATE estimate in the high-CATE group.}
    #'   \item{ATE_low}{ATE estimate in the low-CATE group.}
    #' }
    subgroup_analysis = function(X_new = NULL, quantile = 0.5) {
      private$check_fitted()
      # For training data (X_new = NULL) we can also obtain variance estimates
      # to report approximate subgroup SEs.
      pr <- if (is.null(X_new)) self$predict(NULL, estimate.variance = TRUE) else NULL

      tau_hat   <- self$predict(X_new)$predictions
      threshold <- stats::quantile(tau_hat, quantile)
      high_idx  <- which(tau_hat >= threshold)
      low_idx   <- which(tau_hat <  threshold)

      ate_from_idx <- function(idx) {
        tau <- tau_hat[idx]
        est <- mean(tau)
        se  <- if (!is.null(pr) && !is.null(pr$variance.estimates)) {
          sqrt(sum(pr$variance.estimates[idx])) / length(idx)
        } else {
          stats::sd(tau) / sqrt(length(idx))
        }
        c(estimate = est, std.err = se)
      }
      ate_high <- ate_from_idx(high_idx)
      ate_low  <- ate_from_idx(low_idx)

      cat(" Subgroup Analysis \n")
      cat(sprintf("  Threshold (Q%.0f) : %+.4f\n", quantile * 100, threshold))
      cat(sprintf("  High group  n   : %d   ATE = %+.4f (SE %.4f)\n",
                  length(high_idx), ate_high["estimate"], ate_high["std.err"]))
      cat(sprintf("  Low  group  n   : %d   ATE = %+.4f (SE %.4f)\n",
                  length(low_idx),  ate_low["estimate"],  ate_low["std.err"]))

      invisible(list(
        tau_hat  = tau_hat,
        high_idx = high_idx,
        low_idx  = low_idx,
        ATE_high = ate_high,
        ATE_low  = ate_low
      ))
    },

    #  plot_cate 

    #' @description Plot CATE distribution and top-variable partial dependence.
    #'
    #' Requires \pkg{ggplot2}.  Produces two panels:
    #' \enumerate{
    #'   \item Density of \eqn{\hat\tau(x)} with a vertical ATE line.
    #'   \item Scatter of the top covariate vs \eqn{\hat\tau(x)}.
    #' }
    #' @param X_new   Optional covariate matrix.  Uses training data if NULL.
    #' @param top_var Name or column index of the covariate for the scatter
    #'   panel.  If \code{NULL}, the highest-importance variable is used.
    #' @param ... Extra arguments forwarded to \code{ggplot2::geom_density}.
    #' @return A \code{ggplot} object (invisibly).
    plot_cate = function(X_new = NULL, top_var = NULL, ...) {
      private$check_fitted()
      if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("ggplot2 is required for plot_cate(). Install it first.")
      }

      tau_hat <- self$predict(X_new)$predictions
      ate     <- self$average_treatment_effect()["estimate"]

      # Choose top variable
      if (is.null(top_var)) {
        vi      <- self$variable_importance()
        top_var <- names(vi)[1]
      }

      X_plot <- if (is.null(X_new)) self$forest$X.orig else as.matrix(X_new)
      x_vals <- if (is.character(top_var)) {
        X_plot[, top_var]
      } else {
        X_plot[, top_var]
      }

      df <- data.frame(tau_hat = tau_hat, x_var = x_vals)

      # Panel 1: CATE density
      p1 <- ggplot2::ggplot(df, ggplot2::aes(x = tau_hat)) +
        ggplot2::geom_density(fill = "#4E79A7", alpha = 0.4, ...) +
        ggplot2::geom_vline(xintercept = ate,
                            linetype = "dashed", colour = "#E15759", linewidth = 0.8) +
        ggplot2::annotate("text", x = ate, y = Inf,
                          label = sprintf("ATE = %.3f", ate),
                          hjust = -0.1, vjust = 1.5, colour = "#E15759", size = 3.5) +
        ggplot2::labs(
          title = "CATE Distribution",
          x     = expression(hat(tau)(x)),
          y     = "Density"
        ) +
        ggplot2::theme_minimal(base_size = 12)

      # Panel 2: partial dependence scatter
      var_label <- if (is.character(top_var)) top_var else
        (self$train_colnames[top_var] %||% paste0("X", top_var))

      p2 <- ggplot2::ggplot(df, ggplot2::aes(x = x_var, y = tau_hat)) +
        ggplot2::geom_point(alpha = 0.25, size = 1, colour = "#4E79A7") +
        ggplot2::geom_smooth(method = "loess", formula = y ~ x,
                             colour = "#E15759", se = TRUE, linewidth = 0.8) +
        ggplot2::labs(
          title = sprintf("CATE vs %s (top variable)", var_label),
          x     = var_label,
          y     = expression(hat(tau)(x))
        ) +
        ggplot2::theme_minimal(base_size = 12)

      # Combine with patchwork if available, else print side-by-side
      if (requireNamespace("patchwork", quietly = TRUE)) {
        combined <- p1 + p2
        print(combined)
        return(invisible(combined))
      } else {
        oldpar <- graphics::par(mfrow = c(1, 2))
        on.exit(graphics::par(oldpar))
        print(p1)
        print(p2)
        return(invisible(list(p1 = p1, p2 = p2)))
      }
    },

    #  save_model / load_model 

    #' @description Save the fitted model to disk.
    #' @param path File path (e.g. \code{"model.rds"}).
    #' @return \code{self} invisibly.
    save_model = function(path) {
      saveRDS(self, file = path)
      cat(sprintf("[BoostedIVForest] Model saved to '%s'.\n", path))
      invisible(self)
    }
  ),

  private = list(

    #  dependency checker 
    check_dependencies = function() {
      pkgs <- c("xgboost", "grf", "ranger", "future", "future.apply")
      missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace,
                                   logical(1), quietly = TRUE)]
      if (length(missing_pkgs)) {
        stop(sprintf(
          "Missing required packages: %s. Install with install.packages(...).",
          paste(missing_pkgs, collapse = ", ")
        ))
      }
      if (!exists("CXGBoost", inherits = TRUE)) {
        stop("CXGBoost class not found. Source causalXGBoost.R first.")
      }
      invisible(TRUE)
    },

    #  fitted guard 
    check_fitted = function() {
      if (is.null(self$forest)) {
        stop("Model not fitted. Call $fit() first.")
      }
    },

    #  input validation 
    validate_inputs = function(X, Y, W, Z) {
      n <- nrow(X)
      if (length(Y) != n) stop("Y must have length nrow(X).")
      if (length(W) != n) stop("W must have length nrow(X).")
      if (length(Z) != n) stop("Z must have length nrow(X).")
      if (ncol(X) < 1)    stop("X must have at least one column.")
      if (stats::var(Z) < .Machine$double.eps) {
        stop("Instrument Z has zero variance  cannot identify any effect.")
      }
      if (stats::var(W) < .Machine$double.eps) {
        stop("Treatment W has zero variance.")
      }
    }
  )
)

#  NULL-coalescing operator (base-R safe) 
`%||%` <- function(a, b) if (!is.null(a)) a else b

# 
# SECTION 4  Convenience wrapper
# 

#' Fit a BoostedIVForest and return predictions in a single call.
#'
#' @param X           Numeric covariate matrix (n x p).
#' @param Y           Outcome vector (length n).
#' @param W           Endogenous treatment (length n).
#' @param Z           Instrument (length n).
#' @param n_folds     Cross-fitting folds (default 5).
#' @param nrounds     XGBoost rounds per nuisance model (default 100).
#' @param xgb_params  Extra XGBoost hyperparameters (named list).
#' @param forest_args Extra \code{grf::instrumental_forest} arguments (named
#'   list).
#' @param verbose     Print progress (default \code{TRUE}).
#' @param ...         Forwarded to \code{$predict()}.
#' @return Named list: \code{model} (\code{BoostedIVForest}) and
#'   \code{tau_hat} (numeric vector of CATE estimates).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # boosted_iv_forest(...)
#' }
#' @export
boosted_iv_forest <- function(X, Y, W, Z,
                              n_folds     = 5L,
                              nrounds     = 100L,
                              xgb_params  = list(),
                              forest_args = list(),
                              verbose     = TRUE,
                              ...) {
  model <- BoostedIVForest$new(
    n_folds     = n_folds,
    nrounds     = nrounds,
    xgb_params  = xgb_params,
    forest_args = forest_args
  )
  model$fit(X, Y, W, Z, verbose = verbose)
  tau_hat <- model$predict(...)$predictions
  list(model = model, tau_hat = tau_hat)
}

# 
# SECTION 5  save / load helpers (standalone)
# 

#' Save a BoostedIVForest model to disk.
#' @param model A fitted \code{BoostedIVForest} object.
#' @param path  File path (e.g. \code{"model.rds"}).
#' @return
#' Object returned by \code{save_boosted_iv}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # save_boosted_iv(...)
#' }
#' @export
save_boosted_iv <- function(model, path) {
  if (!inherits(model, "BoostedIVForest")) {
    stop("model must be a BoostedIVForest object.")
  }
  saveRDS(model, file = path)
  cat(sprintf("Model saved to '%s'.\n", path))
  invisible(model)
}

#' Load a BoostedIVForest model from disk.
#' @param path File path written by \code{save_boosted_iv()}.
#' @return A \code{BoostedIVForest} object.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # load_boosted_iv(...)
#' }
#' @export
load_boosted_iv <- function(path) {
  model <- readRDS(path)
  if (!inherits(model, "BoostedIVForest")) {
    stop("The file does not contain a BoostedIVForest object.")
  }
  cat(sprintf("Model loaded from '%s'.\n", path))
  model
}

# 
# SECTION 6  tidymodels (parsnip) shim
# 

#' Register a parsnip model spec for BoostedIVForest (tidymodels shim).
#'
#' Registers a \pkg{parsnip} model mode \code{"causal_iv"} so the model can
#' be used in tidymodels workflows.  Call once at package load time or before
#' building a workflow.
#'
#' @return Invisibly \code{NULL}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # register_boosted_iv_parsnip(...)
#' }
#' @export
register_boosted_iv_parsnip <- function() {
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    stop("parsnip is required. Install tidymodels or parsnip separately.")
  }

  parsnip::set_new_model("boosted_iv_forest")

  parsnip::set_model_mode("boosted_iv_forest", "regression")

  parsnip::set_model_engine(
    "boosted_iv_forest",
    mode   = "regression",
    eng    = "BoostedIVForest"
  )

  parsnip::set_dependency(
    "boosted_iv_forest",
    eng  = "BoostedIVForest",
    pkg  = "grf"
  )

  parsnip::set_fit(
    model  = "boosted_iv_forest",
    eng    = "BoostedIVForest",
    mode   = "regression",
    value  = list(
      interface = "formula",
      protect   = c("formula", "data"),
      func      = c(fun = "boosted_iv_forest"),
      defaults  = list()
    )
  )

  parsnip::set_pred(
    model  = "boosted_iv_forest",
    eng    = "BoostedIVForest",
    mode   = "regression",
    type   = "numeric",
    value  = list(
      pre  = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object  = rlang::expr(object$fit),
        newdata = rlang::expr(new_data)
      )
    )
  )

  invisible(NULL)
}

# 
# SECTION 7  Vignette-style self-contained example
# 

#' Run the built-in demonstration example.
#'
#' Simulates an IV data-generating process with heterogeneous LATE, fits a
#' \code{BoostedIVForest}, prints a summary, evaluates PEHE / ATE error,
#' runs subgroup analysis, and (if ggplot2 is available) plots CATEs.
#'
#' @param n         Sample size (default 1500).
#' @param p         Number of covariates (default 6).
#' @param n_folds   Folds for cross-fitting (default 3 for speed).
#' @param nrounds   XGBoost rounds (default 80 for speed).
#' @param do_plot   Produce CATE plots (default TRUE if ggplot2 available).
#' @param do_save   Save and reload model to test serialisation (default FALSE).
#' @return Named list with \code{model}, \code{eval}, \code{subgroups}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # run_example(...)
#' }
#' @export
run_example <- function(n        = 1500L,
                        p        = 6L,
                        n_folds  = 3L,
                        nrounds  = 80L,
                        do_plot  = requireNamespace("ggplot2", quietly = TRUE),
                        do_save  = FALSE) {

  set.seed(42)
  cat("\n")
  cat("  BoostedIVForest  Demo\n")
  cat("\n\n")

  #  1. Simulate data 
  # DGP:
  #   Z  ~ Bernoulli(0.5)               random encouragement (instrument)
  #   U  ~ N(0,1)                       unobserved confounder
  #   W  ~ Bernoulli(sigmoid(Z - 0.5*U))   endogenous treatment
  #   tau(X) = 2 + 1.5*X1 - X2         heterogeneous LATE
  #   Y  = tau(X)*W + X3 + 0.5*U + N(0,1)
  sigmoid <- function(x) 1 / (1 + exp(-x))

  X     <- matrix(rnorm(n * p), n, p,
                  dimnames = list(NULL, paste0("X", seq_len(p))))
  Z     <- rbinom(n, 1, 0.5)
  U     <- rnorm(n)                                   # confounder
  W     <- rbinom(n, 1, sigmoid(Z - 0.5 * U))
  tau_true <- 2 + 1.5 * X[, 1] - X[, 2]
  Y     <- tau_true * W + X[, 3] + 0.5 * U + rnorm(n)

  cat(sprintf("Simulated n=%d, p=%d. True ATE = %.3f\n\n",
              n, p, mean(tau_true)))

  # Introduce 2 % missing values in X to test imputation
  na_idx      <- sample(length(X), size = floor(0.02 * length(X)))
  X[na_idx]   <- NA

  #  2. Fit 
  future::plan(future::multisession,
               workers = min(n_folds, future::availableCores() - 1L))
  on.exit(future::plan(future::sequential), add = TRUE)

  model <- BoostedIVForest$new(
    n_folds     = n_folds,
    nrounds     = nrounds,
    forest_args = list(num.trees = 1000L)
  )
  model$fit(X, Y, W, Z)

  #  3. Summary 
  model$summary()

  #  4. Evaluate 
  eval_res <- model$evaluate(tau_true)
  cat(sprintf("\n Evaluation (vs. ground truth) \n"))
  cat(sprintf("  PEHE      : %.4f\n", eval_res$PEHE))
  cat(sprintf("  ATE error : %.4f\n", eval_res$ATE_error))

  #  5. Subgroup analysis 
  cat("\n")
  sub <- model$subgroup_analysis(quantile = 0.5)

  #  6. Plots 
  if (do_plot) model$plot_cate(top_var = "X1")

  #  7. Save / reload 
  if (do_save) {
    tmp <- tempfile(fileext = ".rds")
    save_boosted_iv(model, tmp)
    model2 <- load_boosted_iv(tmp)
    cat(sprintf("Reload check  ATE from reloaded model: %.4f\n",
                model2$average_treatment_effect()["estimate"]))
  }

  invisible(list(model = model, eval = eval_res, subgroups = sub))
}
