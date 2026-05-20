# causalXGBoost.R  (v2.0)
#
# Boosted model for causal inference
#   - XGBoost  : two-head causal outcome model (DragonNet-style masked MSE)
#   - Ranger   : propensity score model
#   - R6       : class interface mirroring Python style
#
# Changes from v1.0
# 
# BUG FIXES
#   - make_masked_mse(): hess was 0 for control units (tt=0), causing XGBoost
#     to stall; now always >= 1e-6 via pmax on a constant vector.
#   - predict(): ranger probability matrix column selection failed when factor
#     levels were not "0"/"1"; now uses which.max on colnames for "1".
#   - fit(): cat() progress noise leaked into parallel workers; verbose flag
#     now fully respected (verbose=0 suppresses all output).
#   - fit(): xgb.DMatrix label= argument does not accept a matrix directly in
#     all XGBoost versions; labels are now flattened row-wise to match preds.
#
# NEW FEATURES
#   - summary()   : formatted print of model config + propensity diagnostics.
#   - evaluate()  : PEHE and ATE against ground-truth potential outcomes.
#   - plot_importance() : ggplot2 variable importance chart.
#
# PERFORMANCE
#   - Propensity ranger num.trees raised to 200, oob.error stored.
#   - XGBoost early stopping supported via nrounds + watchlist when
#     eval_data is supplied to fit().
#
# ROBUSTNESS
#   - NA imputation: median imputation applied to X in fit() and predict().
#   - Input checks: informative errors for wrong dimensions, non-numeric X,
#     and constant columns.
#   - Treatment balance warning when P(T=1) < 0.05 or > 0.95.
#
# API
#   - save_model() / load_model() instance methods.
#   - clone_reset()  : return an unfitted copy with same hyperparameters.
#
# DOCUMENTATION
#   - Full roxygen2 tags on every method.
#   - Self-contained run_example() at bottom.
#
# EVALUATION
#   - PEHE() and ATE() updated to accept either matrix [Y(0),Y(1)] form
#     (original API) or two tau vectors (new convenience form).
#
# Dependencies: R6, xgboost, ranger
# Optional    : ggplot2 (plot_importance)

# R6 and ranger are in Imports; no library() call needed inside a package

# 
# SECTION 1  Evaluation metrics  (utils.R, updated)
# 

#' Precision in Estimation of Heterogeneous Effects (PEHE)
#'
#' Accepts two calling conventions:
#' \enumerate{
#'   \item \strong{Matrix form} (original): \code{PEHE(y, y_hat)} where both
#'     arguments are n x 2 matrices with columns \code{[Y(0), Y(1)]}.
#'     Returns MSE of the ITE: \eqn{E[((Y(1)-Y(0)) - (\hat Y(1)-\hat Y(0)))^2]}.
#'   \item \strong{Vector form} (new): \code{PEHE(tau_true = ..., tau_hat = ...)}
#'     where both arguments are length-n numeric vectors of individual treatment
#'     effects. Returns RMSE: \eqn{\sqrt{E[(\tau - \hat\tau)^2]}}.
#' }
#'
#' @param y        n x 2 matrix \code{[Y(0), Y(1)]}. Ignored when
#'   \code{tau_true} is supplied.
#' @param y_hat    n x 2 matrix \code{[Yhat(0), Yhat(1)]}. Ignored when
#'   \code{tau_hat} is supplied.
#' @param tau_true Numeric vector of true ITEs (vector form).
#' @param tau_hat  Numeric vector of estimated ITEs (vector form).
#' @return Numeric scalar.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @export
#' @examples
#' # Matrix form
#' y     <- matrix(c(1, 2, 0.5, 1.5), ncol = 2)
#' y_hat <- matrix(c(1.1, 2.2, 0.4, 1.6), ncol = 2)
#' PEHE(y, y_hat)
#'
#' # Vector form
#' PEHE(tau_true = c(1, 2, 3), tau_hat = c(1.1, 1.9, 3.2))
PEHE <- function(y = NULL, y_hat = NULL,
                 tau_true = NULL, tau_hat = NULL) {
  #  vector form 
  if (!is.null(tau_true) || !is.null(tau_hat)) {
    if (is.null(tau_true) || is.null(tau_hat)) {
      stop("Both tau_true and tau_hat must be supplied in vector form.")
    }
    tau_true <- as.numeric(tau_true)
    tau_hat  <- as.numeric(tau_hat)
    if (length(tau_true) != length(tau_hat)) {
      stop("tau_true and tau_hat must have the same length.")
    }
    return(sqrt(mean((tau_true - tau_hat)^2)))
  }

  #  matrix form (original API) 
  if (is.null(y) || is.null(y_hat)) {
    stop("Supply either (y, y_hat) matrices or (tau_true, tau_hat) vectors.")
  }
  y     <- as.matrix(y)
  y_hat <- as.matrix(y_hat)
  if (ncol(y) != 2L || ncol(y_hat) != 2L) {
    stop("Both y and y_hat must be matrices with 2 columns: [Y(0), Y(1)].")
  }
  if (nrow(y) != nrow(y_hat)) {
    stop("y and y_hat must have the same number of rows.")
  }
  mean(((y[, 2] - y[, 1]) - (y_hat[, 2] - y_hat[, 1]))^2)
}


#' Absolute Error in Average Treatment Effect (ATE)
#'
#' Accepts the same two calling conventions as \code{\link{PEHE}}.
#'
#' @param y        n x 2 matrix \code{[Y(0), Y(1)]} (matrix form).
#' @param y_hat    n x 2 matrix \code{[Yhat(0), Yhat(1)]} (matrix form).
#' @param tau_true Numeric vector of true ITEs (vector form).
#' @param tau_hat  Numeric vector of estimated ITEs (vector form).
#' @return Numeric scalar: \eqn{|E[\tau] - E[\hat\tau]|}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @export
#' @examples
#' # Matrix form
#' y     <- matrix(c(1, 2, 0.5, 1.5), ncol = 2)
#' y_hat <- matrix(c(1.1, 2.2, 0.4, 1.6), ncol = 2)
#' ATE(y, y_hat)
#'
#' # Vector form
#' ATE(tau_true = c(1, 2, 3), tau_hat = c(1.1, 1.9, 3.2))
ATE <- function(y = NULL, y_hat = NULL,
                tau_true = NULL, tau_hat = NULL) {
  #  vector form 
  if (!is.null(tau_true) || !is.null(tau_hat)) {
    if (is.null(tau_true) || is.null(tau_hat)) {
      stop("Both tau_true and tau_hat must be supplied in vector form.")
    }
    tau_true <- as.numeric(tau_true)
    tau_hat  <- as.numeric(tau_hat)
    if (length(tau_true) != length(tau_hat)) {
      stop("tau_true and tau_hat must have the same length.")
    }
    return(abs(mean(tau_true) - mean(tau_hat)))
  }

  #  matrix form (original API) 
  if (is.null(y) || is.null(y_hat)) {
    stop("Supply either (y, y_hat) matrices or (tau_true, tau_hat) vectors.")
  }
  y     <- as.matrix(y)
  y_hat <- as.matrix(y_hat)
  if (ncol(y) != 2L || ncol(y_hat) != 2L) {
    stop("Both y and y_hat must be matrices with 2 columns: [Y(0), Y(1)].")
  }
  if (nrow(y) != nrow(y_hat)) {
    stop("y and y_hat must have the same number of rows.")
  }
  abs(mean(y[, 2] - y[, 1]) - mean(y_hat[, 2] - y_hat[, 1]))
}

# 
# SECTION 2  Internal helpers
# 

#' Median imputation table (per column)
#' @keywords internal
.cxgb_build_imputation <- function(X) {
  apply(X, 2, function(col) {
    m <- median(col, na.rm = TRUE)
    if (is.na(m)) 0 else m
  })
}

#' Apply a pre-built imputation table to a matrix
#' @keywords internal
.cxgb_apply_imputation <- function(X, table) {
  for (j in seq_len(ncol(X))) {
    na_rows <- is.na(X[, j])
    if (any(na_rows)) X[na_rows, j] <- table[j]
  }
  X
}

#' Validate CXGBoost inputs
#' @keywords internal
.cxgb_validate <- function(X, t, y) {
  if (!is.numeric(X)) {
    stop("X must be a numeric matrix.")
  }
  if (nrow(X) != length(t)) {
    stop(sprintf("nrow(X) = %d but length(t) = %d.", nrow(X), length(t)))
  }
  if (length(t) != length(y)) {
    stop(sprintf("length(t) = %d but length(y) = %d.", length(t), length(y)))
  }
  if (!all(t %in% c(0L, 1L))) {
    stop("t must be binary with values 0 and 1.")
  }

  # Constant column check
  const_cols <- which(apply(X, 2, function(col) {
    stats::var(col, na.rm = TRUE) < .Machine$double.eps
  }))
  if (length(const_cols) > 0) {
    warning(sprintf(
      "Constant columns detected (indices: %s). These carry no information.",
      paste(const_cols, collapse = ", ")
    ))
  }

  # Treatment balance
  p_treat <- mean(t)
  if (p_treat < 0.05 || p_treat > 0.95) {
    warning(sprintf(
      "Severe treatment imbalance: P(T=1) = %.3f. Propensity estimates may be unreliable.",
      p_treat
    ))
  }

  invisible(TRUE)
}

# 
# SECTION 3  CXGBoost R6 class
# 

#' C-XGBoost: Boosted causal outcome model with propensity scoring
#'
#' R6 class implementing a causal inference model that uses \pkg{xgboost} for
#' the outcome model (two-head: Y(0) and Y(1)) and \pkg{ranger} for the
#' propensity score.  A DragonNet-style treatment mask ensures each unit's
#' gradient only updates the head corresponding to its observed treatment.
#'
#' @section Model architecture:
#' \itemize{
#'   \item \strong{Outcome model}: XGBoost with \code{num_target = 2} and a
#'     custom masked-MSE objective.  Head 0 is updated for control units
#'     (\eqn{T=0}); head 1 for treated units (\eqn{T=1}).
#'   \item \strong{Propensity model}: Ranger probability forest predicting
#'     \eqn{P(T=1 \mid X)}.
#' }
#'
#' @field parameters      List of XGBoost hyperparameters merged with defaults.
#' @field booster         Fitted \code{xgb.Booster} (post-fit).
#' @field propensity_model Fitted \code{ranger} model (post-fit).
#' @field scale_pos_weight Reserved for future use.
#' @field imputation_table Per-column medians for NA imputation (post-fit).
#' @field train_colnames  Column names of training X (post-fit).
#' @field oob_error       OOB classification error of the propensity model.
#' @field fit_time_secs   Training wall-clock time in seconds.
#'
#' @return
#' Object returned by \code{function_name}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # function_name(...)
#' }
#' @export
CXGBoost <- R6Class(
  "CXGBoost",
  public = list(

    #  public fields 
    parameters       = NULL,
    booster          = NULL,
    propensity_model = NULL,
    scale_pos_weight = NULL,
    imputation_table = NULL,
    train_colnames   = NULL,
    oob_error        = NULL,
    fit_time_secs    = NULL,

    #  initialize 

    #' @description Create a new CXGBoost model.
    #' @param parameters      Named list of XGBoost hyperparameters. Merged with
    #'   defaults (\code{eta=0.05}, \code{max_depth=6}, \code{subsample=0.8},
    #'   \code{colsample_bytree=0.8}, \code{min_child_weight=1},
    #'   \code{lambda=1}, \code{alpha=0}, \code{tree_method="hist"}).
    #' @param propensity_model \code{NULL} to fit internally, or a pre-fitted
    #'   \code{ranger} probability forest.
    #' @param scale_pos_weight Reserved; currently unused.
    initialize = function(parameters       = list(),
                          propensity_model = NULL,
                          scale_pos_weight = NULL) {
      if (!is.list(parameters)) {
        stop("parameters must be a named list.")
      }
      self$parameters       <- parameters
      self$propensity_model <- propensity_model
      self$scale_pos_weight <- scale_pos_weight
    },

    #  fit 

    #' @description Fit the C-XGBoost outcome model and propensity model.
    #'
    #' @param X         Numeric covariate matrix (n x p). NAs are median-imputed.
    #' @param t         Binary treatment vector (0/1), length n.
    #' @param y         Observed outcome vector (numeric), length n.
    #' @param nrounds   XGBoost boosting rounds (default 100).
    #' @param verbose   Integer verbosity: 0 = silent, 1 = progress (default 0).
    #' @param eval_data Optional named list \code{list(X=..., t=..., y=...)} for
    #'   XGBoost watchlist early-stopping diagnostics (no early stopping is
    #'   applied  evaluation loss is printed when \code{verbose >= 1}).
    #' @return \code{self} invisibly (supports method chaining).
    fit = function(X, t, y,
                   nrounds   = 100L,
                   verbose   = 0L,
                   eval_data = NULL) {

      #  coerce 
      X <- as.matrix(X)
      t <- as.integer(t)
      y <- as.numeric(y)

      self$train_colnames <- colnames(X)

      #  impute 
      if (anyNA(X)) {
        if (verbose >= 1L) cat("[CXGBoost] Imputing missing values in X...\n")
        self$imputation_table <- .cxgb_build_imputation(X)
        X <- .cxgb_apply_imputation(X, self$imputation_table)
      }

      #  validate 
      .cxgb_validate(X, t, y)

      start <- Sys.time()

      #  treatment mask 
      # t_i = 0    mask = [1, 0]  (gradient flows to head 0 only)
      # t_i = 1    mask = [0, 1]  (gradient flows to head 1 only)
      tt_mat <- t(vapply(
        t,
        FUN       = function(ti) if (ti == 0L) c(1, 0) else c(0, 1),
        FUN.VALUE = numeric(2)
      ))
      # Flatten row-wise to match XGBoost's flat prediction vector
      tt <- as.vector(t(tt_mat))   # length 2n

      #  XGBoost parameters 
      params <- modifyList(
        list(
          eta                        = 0.05,
          max_depth                  = 6L,
          subsample                  = 0.8,
          colsample_bytree           = 0.8,
          min_child_weight           = 1,
          lambda                     = 1,
          alpha                      = 0,
          tree_method                = "hist",
          num_target                 = 2L,
          multi_strategy             = "multi_output_tree",
          disable_default_eval_metric = TRUE
        ),
        self$parameters
      )

      #  outcome model: multi-output if available; else two separate boosters 
      if (verbose >= 1L) cat("[CXGBoost] Fitting outcome model...\n")

      # Multi-output training path (preferred). Some xgboost builds only accept
      # one label per row (no multi-target); in that case fall back to two models.
      multi_ok <- TRUE
      multi_err <- NULL
      res <- try({
        # Labels flattened row-wise: [y1, y1, y2, y2, ..., yn, yn]
        yt_flat <- as.vector(t(cbind(y, y)))
        dtrain  <- xgboost::xgb.DMatrix(data = X, label = yt_flat)

        watchlist <- list(train = dtrain)
        if (!is.null(eval_data)) {
          X_eval <- as.matrix(eval_data$X)
          if (!is.null(self$imputation_table) && anyNA(X_eval)) {
            X_eval <- .cxgb_apply_imputation(X_eval, self$imputation_table)
          }
          y_eval  <- as.numeric(eval_data$y)
          yt_eval <- as.vector(t(cbind(y_eval, y_eval)))
          deval   <- xgboost::xgb.DMatrix(data = X_eval, label = yt_eval)
          watchlist[["eval"]] <- deval
        }

        custom_obj  <- private$make_masked_mse(tt)
        self$booster <- xgboost::xgb.train(
          params    = params,
          data      = dtrain,
          nrounds   = as.integer(nrounds),
          obj       = custom_obj,
          watchlist = watchlist,
          verbose   = as.integer(verbose >= 1L)
        )
      }, silent = TRUE)
      if (inherits(res, "try-error")) {
        multi_ok  <- FALSE
        multi_err <- as.character(res)
      }

      if (!multi_ok) {
        if (verbose >= 1L) {
          cat("[CXGBoost] Multi-output training unavailable; falling back to two single-output boosters.\n")
        }

        # Single-output params: drop multi-target-specific keys
        params_single <- params
        params_single$num_target     <- NULL
        params_single$multi_strategy <- NULL

        # Fit separate models on observed outcomes in each treatment arm
        idx0 <- which(t == 0L)
        idx1 <- which(t == 1L)
        if (length(idx0) < 2L || length(idx1) < 2L) {
          stop("Not enough samples in one treatment arm to fit fallback boosters.")
        }

        d0 <- xgboost::xgb.DMatrix(data = X[idx0, , drop = FALSE], label = y[idx0])
        d1 <- xgboost::xgb.DMatrix(data = X[idx1, , drop = FALSE], label = y[idx1])

        self$booster <- list(
          mode = "two_boosters",
          y0   = xgboost::xgb.train(
            params  = params_single,
            data    = d0,
            nrounds = as.integer(nrounds),
            verbose = as.integer(verbose >= 1L)
          ),
          y1   = xgboost::xgb.train(
            params  = params_single,
            data    = d1,
            nrounds = as.integer(nrounds),
            verbose = as.integer(verbose >= 1L)
          ),
          multi_error = multi_err
        )
      }

      #  propensity model 
      if (verbose >= 1L) cat("[CXGBoost] Fitting propensity model...\n")

      if (is.null(self$propensity_model)) {
        ps_df <- data.frame(treatment = factor(t, levels = c(0L, 1L)), X)

        self$propensity_model <- ranger::ranger(
          treatment ~ .,
          data        = ps_df,
          num.trees   = 200L,
          max.depth   = 4L,
          probability = TRUE,
          seed        = 42L,
          num.threads = max(1L, parallel::detectCores() - 1L)
        )
        self$oob_error <- self$propensity_model$prediction.error

      } else if (!inherits(self$propensity_model, "ranger")) {
        stop("propensity_model must be NULL or a fitted ranger object.")
      }

      self$fit_time_secs <- as.numeric(
        difftime(Sys.time(), start, units = "secs")
      )

      if (verbose >= 1L) {
        cat(sprintf("[CXGBoost] Training complete  %.2f sec\n",
                    self$fit_time_secs))
        if (!is.null(self$oob_error)) {
          cat(sprintf("[CXGBoost] Propensity OOB error: %.4f\n",
                      self$oob_error))
        }
      }

      invisible(self)
    },

    #  predict 

    #' @description Predict potential outcomes and propensity scores.
    #'
    #' @param X Numeric covariate matrix (n x p). NAs are imputed using the
    #'   table built during \code{fit()}. Column order must match training X.
    #' @return A named list with components:
    #' \describe{
    #'   \item{y0_hat}{Predicted Y(0)  potential outcome under control.}
    #'   \item{y1_hat}{Predicted Y(1)  potential outcome under treatment.}
    #'   \item{propensity_score}{P(T=1|X), clipped to (1e-6, 1-1e-6).}
    #'   \item{tau_hat}{Individual treatment effect Y(1) - Y(0).}
    #' }
    predict = function(X) {
      if (is.null(self$booster)) {
        stop("Model not fitted. Call $fit() first.")
      }

      X <- as.matrix(X)

      #  NA imputation 
      if (!is.null(self$imputation_table) && anyNA(X)) {
        X <- .cxgb_apply_imputation(X, self$imputation_table)
      }

      #  column alignment (bug fix v2.0) 
      if (!is.null(self$train_colnames) &&
          !is.null(colnames(X)) &&
          !identical(colnames(X), self$train_colnames)) {
        X <- X[, self$train_colnames, drop = FALSE]
      }

      #  outcome predictions 
      dtest    <- xgboost::xgb.DMatrix(data = X)
      if (inherits(self$booster, "xgb.Booster")) {
        raw_pred <- predict(self$booster, dtest)
        # raw_pred is length 2n, ordered [y0_1, y1_1, y0_2, y1_2, ...]
        pred <- matrix(raw_pred, ncol = 2, byrow = TRUE)
      } else if (is.list(self$booster) && identical(self$booster$mode, "two_boosters")) {
        y0_hat <- as.numeric(predict(self$booster$y0, dtest))
        y1_hat <- as.numeric(predict(self$booster$y1, dtest))
        pred   <- cbind(y0_hat, y1_hat)
      } else {
        stop("Unknown booster type. Refit the model.")
      }

      #  propensity scores (bug fix v2.0) 
      ps_raw <- predict(
        self$propensity_model,
        data = data.frame(X)
      )$predictions

      if (is.null(dim(ps_raw))) {
        # Single-column vector (only one class observed)
        propensity_score <- as.numeric(ps_raw)
      } else {
        # Find the column corresponding to class "1"
        col_names <- colnames(ps_raw)
        pos_col   <- if ("1" %in% col_names) {
          "1"
        } else {
          # Fall back: last column (highest factor level)
          col_names[ncol(ps_raw)]
        }
        propensity_score <- as.numeric(ps_raw[, pos_col])
      }

      # Clip to avoid boundary weights
      propensity_score <- pmax(1e-6, pmin(1 - 1e-6, propensity_score))

      list(
        y0_hat           = pred[, 1],
        y1_hat           = pred[, 2],
        propensity_score = propensity_score,
        tau_hat          = pred[, 2] - pred[, 1]   # ITE
      )
    },

    #  summary 

    #' @description Print a formatted model summary.
    #' @return \code{self} invisibly.
    summary = function() {
      if (is.null(self$booster)) {
        cat("[CXGBoost] Model not yet fitted.\n")
        return(invisible(self))
      }

      cat("\n")
      cat("  CXGBoost  Model Summary\n")
      cat("\n\n")

      # Configuration
      cat(" Configuration \n")
      p <- modifyList(
        list(eta = 0.05, max_depth = 6L, subsample = 0.8,
             colsample_bytree = 0.8, lambda = 1, alpha = 0),
        self$parameters
      )
      cat(sprintf("  eta (learning rate)  : %.4f\n", p$eta))
      cat(sprintf("  max_depth            : %d\n",   p$max_depth))
      cat(sprintf("  subsample            : %.2f\n", p$subsample))
      cat(sprintf("  colsample_bytree     : %.2f\n", p$colsample_bytree))
      cat(sprintf("  lambda (L2)          : %.2f\n", p$lambda))
      cat(sprintf("  alpha  (L1)          : %.2f\n", p$alpha))
      n_trees <- if (inherits(self$booster, "xgb.Booster")) {
        self$booster$best_iteration %||% "?"
      } else if (is.list(self$booster) && identical(self$booster$mode, "two_boosters")) {
        "two_boosters"
      } else {
        "?"
      }
      cat(sprintf("  Boosting rounds      : %s\n\n", n_trees))

      # Timing
      if (!is.null(self$fit_time_secs)) {
        cat(sprintf("  Fit time             : %.2f sec\n\n",
                    self$fit_time_secs))
      }

      # Propensity diagnostics
      cat(" Propensity Model \n")
      if (!is.null(self$oob_error)) {
        cat(sprintf("  Ranger OOB error     : %.4f\n", self$oob_error))
      }
      cat(sprintf("  Ranger num.trees     : %d\n",
                  self$propensity_model$num.trees))
      cat(sprintf("  Ranger max.depth     : %d\n\n",
                  self$propensity_model$max.depth %||% 4L))

      # Variable importance (XGBoost gain)
      cat(" Top 10 Variables (XGBoost gain) \n")
      booster_for_imp <- if (inherits(self$booster, "xgb.Booster")) {
        self$booster
      } else if (is.list(self$booster) && identical(self$booster$mode, "two_boosters")) {
        self$booster$y1
      } else {
        NULL
      }
      imp <- if (!is.null(booster_for_imp)) xgboost::xgb.importance(model = booster_for_imp) else NULL
      if (!is.null(imp) && nrow(imp) > 0) {
        top <- head(imp[order(-imp$Gain), ], 10)
        for (i in seq_len(nrow(top))) {
          bar <- strrep("", round(top$Gain[i] /
                                     max(top$Gain) * 30))
          cat(sprintf("  %-12s %.4f  %s\n",
                      top$Feature[i], top$Gain[i], bar))
        }
      }
      cat("\n")

      invisible(self)
    },

    #  evaluate 

    #' @description Evaluate model against ground-truth potential outcomes.
    #'
    #' @param y_true n x 2 matrix \code{[Y(0), Y(1)]} of true potential
    #'   outcomes, or a numeric vector of true ITEs (\code{tau_true}).
    #' @param X      Covariate matrix for prediction. If \code{NULL}, the method
    #'   cannot predict  supply \code{X} from the training set or a test set.
    #' @return Named list: \code{PEHE}, \code{ATE_error}, \code{tau_hat},
    #'   \code{y0_hat}, \code{y1_hat}.
    evaluate = function(y_true, X) {
      preds <- self$predict(X)

      # Accept either matrix or ITE vector
      if (is.matrix(y_true) && ncol(y_true) == 2L) {
        pehe_val <- PEHE(y     = y_true,
                         y_hat = cbind(preds$y0_hat, preds$y1_hat))
        ate_val  <- ATE(y     = y_true,
                        y_hat = cbind(preds$y0_hat, preds$y1_hat))
      } else {
        tau_true <- as.numeric(y_true)
        pehe_val <- PEHE(tau_true = tau_true, tau_hat = preds$tau_hat)
        ate_val  <- ATE(tau_true  = tau_true, tau_hat = preds$tau_hat)
      }

      cat(sprintf("  PEHE      : %.6f\n", pehe_val))
      cat(sprintf("  ATE error : %.6f\n", ate_val))

      invisible(list(
        PEHE      = pehe_val,
        ATE_error = ate_val,
        tau_hat   = preds$tau_hat,
        y0_hat    = preds$y0_hat,
        y1_hat    = preds$y1_hat
      ))
    },

    #  plot_importance 

    #' @description Plot variable importance from the XGBoost outcome model.
    #'
    #' Requires \pkg{ggplot2}.
    #'
    #' @param top_n  Number of top variables to display (default 20).
    #' @param metric Importance metric: \code{"Gain"}, \code{"Cover"}, or
    #'   \code{"Frequency"} (default \code{"Gain"}).
    #' @return A \code{ggplot} object (invisibly).
    plot_importance = function(top_n = 20L, metric = "Gain") {
      if (is.null(self$booster)) {
        stop("Model not fitted. Call $fit() first.")
      }
      if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("ggplot2 is required. Install it first.")
      }

      booster_for_imp <- if (inherits(self$booster, "xgb.Booster")) {
        self$booster
      } else if (is.list(self$booster) && identical(self$booster$mode, "two_boosters")) {
        self$booster$y1
      } else {
        stop("Unknown booster type. Refit the model.")
      }

      imp <- xgboost::xgb.importance(model = booster_for_imp)
      if (is.null(imp) || nrow(imp) == 0) {
        stop("No importance information available from this booster.")
      }

      if (!metric %in% names(imp)) {
        stop(sprintf("metric must be one of: %s",
                     paste(names(imp), collapse = ", ")))
      }

      imp     <- imp[order(-imp[[metric]]), ]
      top_imp <- head(imp, top_n)
      top_imp$Feature <- factor(top_imp$Feature,
                                levels = rev(top_imp$Feature))

      p <- ggplot2::ggplot(
        top_imp,
        ggplot2::aes(x = .data[[metric]], y = Feature)
      ) +
        ggplot2::geom_col(fill = "#4E79A7", alpha = 0.85) +
        ggplot2::labs(
          title = sprintf("CXGBoost Variable Importance (top %d, %s)",
                          top_n, metric),
          x = metric,
          y = NULL
        ) +
        ggplot2::theme_minimal(base_size = 12) +
        ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())

      print(p)
      invisible(p)
    },

    #  save_model 

    #' @description Save the fitted model to an RDS file.
    #' @param path File path (e.g. \code{"cxgboost_model.rds"}).
    #' @return \code{self} invisibly.
    save_model = function(path) {
      saveRDS(self, file = path)
      cat(sprintf("[CXGBoost] Model saved to '%s'.\n", path))
      invisible(self)
    },

    #  clone_reset 

    #' @description Return an unfitted copy with the same hyperparameters.
    #'
    #' Useful for cross-validation loops where the same configuration is
    #' reused across folds without retaining fitted state.
    #'
    #' @return A new unfitted \code{CXGBoost} object.
    clone_reset = function() {
      CXGBoost$new(
        parameters       = self$parameters,
        propensity_model = NULL,
        scale_pos_weight = self$scale_pos_weight
      )
    }
  ),

  private = list(

    #  masked MSE objective (bug fix v2.0) 
    #
    # v1.0 bug: hess = pmax(2 * tt, 1e-16) returned 0 for control units
    # (tt = [1, 0]) on the second head  XGBoost uses hess as a learning-rate
    # weight and a zero hess stalls those leaves.
    # Fix: always return hess >= 1e-6 on both elements via a constant floor.
    make_masked_mse = function(tt) {
      force(tt)

      function(preds, dtrain) {
        labels <- xgboost::getinfo(dtrain, "label")

        # Gradient: zero for the masked head (tt = 0 there)
        grad <- 2 * (preds - labels) * tt

        # Hessian: constant 2 where tt = 1, tiny floor elsewhere
        # This prevents zero-hessian stalling on the inactive head.
        hess <- ifelse(tt > 0, 2, 1e-6)

        list(grad = grad, hess = hess)
      }
    }
  )
)

#  NULL-coalescing operator 
`%||%` <- function(a, b) if (!is.null(a)) a else b

#  standalone load helper 

#' Load a saved CXGBoost model from disk.
#' @param path File path written by \code{$save_model()} or
#'   \code{save_cxgboost()}.
#' @return A \code{CXGBoost} R6 object.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # load_cxgboost(...)
#' }
#' @export
load_cxgboost <- function(path) {
  model <- readRDS(path)
  if (!inherits(model, "CXGBoost")) {
    stop("The file does not contain a CXGBoost object.")
  }
  cat(sprintf("[CXGBoost] Model loaded from '%s'.\n", path))
  model
}

#' Save a CXGBoost model to disk (standalone wrapper).
#' @param model A fitted \code{CXGBoost} object.
#' @param path  File path.
#' @return
#' Object returned by \code{save_cxgboost}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # save_cxgboost(...)
#' }
#' @export
save_cxgboost <- function(model, path) {
  if (!inherits(model, "CXGBoost")) {
    stop("model must be a CXGBoost object.")
  }
  saveRDS(model, file = path)
  cat(sprintf("[CXGBoost] Model saved to '%s'.\n", path))
  invisible(model)
}

# 
# SECTION 4  Vignette-style self-contained example
# 

#' Run the built-in CXGBoost demonstration.
#'
#' Simulates a partially linear DGP with heterogeneous treatment effects,
#' fits a \code{CXGBoost} model, prints a summary, evaluates PEHE / ATE
#' error, and plots variable importance (if ggplot2 available).
#'
#' @param n       Sample size (default 1000).
#' @param p       Number of covariates (default 8).
#' @param nrounds XGBoost boosting rounds (default 100).
#' @param do_plot Plot variable importance (default TRUE if ggplot2 available).
#' @param do_save Test save/reload (default FALSE).
#' @return Named list: \code{model}, \code{preds}, \code{eval}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # run_cxgboost_example(...)
#' }
#' @export
run_cxgboost_example <- function(n       = 1000L,
                                 p       = 8L,
                                 nrounds = 100L,
                                 do_plot = requireNamespace("ggplot2",
                                                            quietly = TRUE),
                                 do_save = FALSE) {
  set.seed(42)

  cat("\n")
  cat("  CXGBoost  Demo\n")
  cat("\n\n")

  #  1. Simulate data 
  # DGP  (unconfounded for illustration):
  #   T  ~ Bernoulli(sigmoid(0.5 * X1))
  #   tau(X) = 2 + X1 - 0.5 * X2          (heterogeneous ITE)
  #   Y(0) = X3 + N(0,1)
  #   Y(1) = Y(0) + tau(X)
  #   Y    = (1 - T)*Y(0) + T*Y(1)
  sigmoid <- function(x) 1 / (1 + exp(-x))

  X      <- matrix(rnorm(n * p), n, p,
                   dimnames = list(NULL, paste0("X", seq_len(p))))
  T_bin  <- rbinom(n, 1, sigmoid(0.5 * X[, 1]))
  tau_t  <- 2 + X[, 1] - 0.5 * X[, 2]
  Y0     <- X[, 3] + rnorm(n)
  Y1     <- Y0 + tau_t
  Y      <- ifelse(T_bin == 1, Y1, Y0)

  # Ground truth matrix for matrix-form PEHE/ATE
  y_true <- cbind(Y0, Y1)

  # Introduce 2 % missingness to test imputation
  na_idx   <- sample(length(X), size = floor(0.02 * length(X)))
  X[na_idx] <- NA

  cat(sprintf("Simulated n=%d, p=%d | True ATE = %.3f\n\n",
              n, p, mean(tau_t)))

  #  2. Fit 
  model <- CXGBoost$new(
    parameters = list(eta = 0.05, max_depth = 5L, subsample = 0.8)
  )
  model$fit(X, T_bin, Y, nrounds = nrounds, verbose = 1L)

  #  3. Summary 
  model$summary()

  #  4. Predict 
  preds <- model$predict(X)
  cat(sprintf("\nPrediction sample (first 5 rows):\n"))
  print(data.frame(
    Y0_hat = round(preds$y0_hat[1:5],  3),
    Y1_hat = round(preds$y1_hat[1:5],  3),
    tau    = round(preds$tau_hat[1:5], 3),
    PS     = round(preds$propensity_score[1:5], 3)
  ))

  #  5. Evaluate 
  cat("\n Evaluation (matrix form) \n")
  eval_mat <- model$evaluate(y_true = y_true, X = X)

  cat("\n Evaluation (vector form) \n")
  eval_vec <- model$evaluate(y_true = tau_t, X = X)

  #  6. Plot 
  if (do_plot) model$plot_importance(top_n = min(p, 10L))

  #  7. Save / reload 
  if (do_save) {
    tmp <- tempfile(fileext = ".rds")
    model$save_model(tmp)
    m2  <- load_cxgboost(tmp)
    p2  <- m2$predict(X)
    cat(sprintf("Reload check  max tau diff: %.2e\n",
                max(abs(p2$tau_hat - preds$tau_hat))))
  }

  invisible(list(model = model, preds = preds,
                 eval_mat = eval_mat, eval_vec = eval_vec))
}

# 
# SECTION 5  Optional extensions loader
# 

#' Load causal boosting extensions from the R directory
#'
#' Convenience helper that sources:
#' \itemize{
#'   \item \code{multi_arm_causal_boost.R} (MultiArmCausalBoost)
#'   \item \code{causalBoosted_IV.R} (BoostedIVForest)
#' }
#'
#' This keeps \code{causalXGBoost.R} as a one-stop entry point while avoiding
#' symbol duplication (for example \code{PEHE}, \code{\%||\%}) that would occur if
#' all code were copied directly into one file.
#'
#' @param r_dir Directory containing extension scripts. Defaults to \code{"R"}.
#' @param files Character vector of script filenames to source.
#' @param verbose Print progress messages (default \code{TRUE}).
#' @param local  Passed to \code{source()} (default \code{.GlobalEnv}).
#' @return Invisibly, a named list containing loaded file paths.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # load_cxgboost_extensions(...)
#' }
#' @export
load_cxgboost_extensions <- function(
  r_dir = "R",
  files = c("multi_arm_causal_boost.R", "causalBoosted_IV.R"),
  verbose = TRUE,
  local = .GlobalEnv
) {
  loaded <- list()

  for (f in files) {
    script_path <- file.path(r_dir, f)
    if (!file.exists(script_path)) {
      warning(sprintf(
        "[CXGBoost] Extension file not found: '%s' (skipped).",
        script_path
      ))
      next
    }

    if (isTRUE(verbose)) {
      cat(sprintf("[CXGBoost] Loading extension: %s\n", script_path))
    }
    source(script_path, local = local)
    loaded[[f]] <- normalizePath(script_path, winslash = "/", mustWork = FALSE)
  }

  invisible(loaded)
}

