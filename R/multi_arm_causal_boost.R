# multi_arm_causal_boost.R
#
# Boosted multi-arm causal model
#   - XGBoost: one outcome model per treatment arm
#   - Ranger : multiclass propensity score model
#   - R6     : class interface
#
# Design:
#   This is a practical multi-arm extension of the binary CXGBoost starter.
#   It fits separate boosted outcome models E[Y | X, W = k] for each arm k,
#   then predicts all potential outcomes mu_k(X). Treatment contrasts are
#   computed relative to a chosen baseline arm.
#
# Notes:
#   - Supports factor-valued treatment W with >= 2 levels.
#   - Supports univariate or multivariate Y.
#   - For multivariate Y, one xgboost model is fit per (arm, outcome).
#   - Propensity scores are estimated with ranger probability forest.
#   - This is closer to a T-learner than an R-learner / DR-learner.
#
# Dependencies: R6, xgboost, ranger
# Optional    : ggplot2

# R6 and ranger are in Imports; no library() call needed inside a package

# 
# Utilities
# 

`%||%` <- function(a, b) if (!is.null(a)) a else b

.mcb_build_imputation <- function(X) {
  apply(X, 2, function(col) {
    m <- median(col, na.rm = TRUE)
    if (is.na(m)) 0 else m
  })
}

.mcb_apply_imputation <- function(X, table) {
  for (j in seq_len(ncol(X))) {
    idx <- is.na(X[, j])
    if (any(idx)) X[idx, j] <- table[j]
  }
  X
}

.mcb_validate_inputs <- function(X, Y, W) {
  X <- as.matrix(X)

  if (!is.numeric(X)) {
    stop("X must be a numeric matrix.")
  }

  if (!is.factor(W)) {
    stop("W must be a factor.")
  }

  if (nlevels(W) < 2L) {
    stop("W must contain at least two treatment arms.")
  }

  if (anyNA(W)) {
    stop("W contains NA values.")
  }

  if (is.vector(Y)) {
    Y <- matrix(as.numeric(Y), ncol = 1)
  } else {
    Y <- as.matrix(Y)
  }

  if (!is.numeric(Y)) {
    stop("Y must be numeric vector or numeric matrix.")
  }

  if (nrow(X) != nrow(Y)) {
    stop(sprintf("nrow(X) = %d but nrow(Y) = %d.", nrow(X), nrow(Y)))
  }

  if (nrow(X) != length(W)) {
    stop(sprintf("nrow(X) = %d but length(W) = %d.", nrow(X), length(W)))
  }

  arm_counts <- table(W)
  if (any(arm_counts < 2L)) {
    stop(sprintf(
      "Each treatment arm needs at least 2 observations. Counts: %s",
      paste(names(arm_counts), arm_counts, sep = "=", collapse = ", ")
    ))
  }

  const_cols <- which(apply(X, 2, function(col) {
    stats::var(col, na.rm = TRUE) < .Machine$double.eps
  }))
  if (length(const_cols) > 0) {
    warning(sprintf(
      "Constant columns detected in X (indices: %s).",
      paste(const_cols, collapse = ", ")
    ))
  }

  list(X = X, Y = Y, W = W)
}

.mcb_clip_probs <- function(p, eps = 1e-6) {
  pmax(eps, pmin(1 - eps, p))
}

.mcb_make_outcome_names <- function(Y) {
  cn <- colnames(Y)
  if (is.null(cn)) {
    paste0("Y.", seq_len(ncol(Y)))
  } else {
    make.names(cn, unique = TRUE)
  }
}

.mcb_make_contrast_names <- function(treatment_levels, baseline) {
  others <- setdiff(treatment_levels, baseline)
  paste0(others, " - ", baseline)
}

# PEHE generalized to multi-arm contrasts
# tau_true and tau_hat are arrays: [n, num_contrasts, num_outcomes]
multi_arm_PEHE <- function(tau_true, tau_hat, rmse = TRUE) {
  tau_true <- as.array(tau_true)
  tau_hat  <- as.array(tau_hat)

  if (!identical(dim(tau_true), dim(tau_hat))) {
    stop("tau_true and tau_hat must have identical dimensions.")
  }

  err <- mean((tau_true - tau_hat)^2)
  if (rmse) sqrt(err) else err
}

# ATE error generalized to multi-arm contrasts
multi_arm_ATE <- function(tau_true, tau_hat) {
  tau_true <- as.array(tau_true)
  tau_hat  <- as.array(tau_hat)

  if (!identical(dim(tau_true), dim(tau_hat))) {
    stop("tau_true and tau_hat must have identical dimensions.")
  }

  # average over rows, preserve contrast/outcome dims
  d <- dim(tau_true)
  n <- d[1]
  true_mean <- apply(tau_true, c(2, 3), mean)
  hat_mean  <- apply(tau_hat,  c(2, 3), mean)
  abs(hat_mean - true_mean)
}

# 
# MultiArmCausalBoost
# 

MultiArmCausalBoost <- R6Class(
  "MultiArmCausalBoost",
  public = list(

    parameters = NULL,
    propensity_model = NULL,
    outcome_models = NULL,
    imputation_table = NULL,
    train_colnames = NULL,
    treatment_levels = NULL,
    baseline = NULL,
    outcome_names = NULL,
    num_outcomes = NULL,
    oob_error = NULL,
    fit_time_secs = NULL,
    arm_counts = NULL,

    initialize = function(parameters = list(),
                          propensity_model = NULL,
                          baseline = NULL) {
      if (!is.list(parameters)) {
        stop("parameters must be a named list.")
      }
      self$parameters <- parameters
      self$propensity_model <- propensity_model
      self$baseline <- baseline
    },

    fit = function(X, Y, W,
                   nrounds = 200L,
                   verbose = 0L,
                   eval_data = NULL) {

      validated <- .mcb_validate_inputs(X, Y, W)
      X <- validated$X
      Y <- validated$Y
      W <- droplevels(validated$W)

      self$train_colnames   <- colnames(X)
      self$treatment_levels <- levels(W)
      self$arm_counts       <- table(W)
      self$num_outcomes     <- ncol(Y)
      self$outcome_names    <- .mcb_make_outcome_names(Y)

      if (is.null(self$baseline)) {
        self$baseline <- self$treatment_levels[1]
      }
      if (!self$baseline %in% self$treatment_levels) {
        stop("baseline must be one of the levels(W).")
      }

      if (anyNA(X)) {
        if (verbose >= 1L) cat("[MultiArmCausalBoost] Imputing missing values in X...\n")
        self$imputation_table <- .mcb_build_imputation(X)
        X <- .mcb_apply_imputation(X, self$imputation_table)
      }

      start <- Sys.time()

      params <- modifyList(
        list(
          eta = 0.05,
          max_depth = 6L,
          subsample = 0.8,
          colsample_bytree = 0.8,
          min_child_weight = 1,
          lambda = 1,
          alpha = 0,
          objective = "reg:squarederror",
          tree_method = "hist"
        ),
        self$parameters
      )

      if (verbose >= 1L) cat("[MultiArmCausalBoost] Fitting outcome models...\n")

      self$outcome_models <- list()

      for (arm in self$treatment_levels) {
        idx_arm <- which(W == arm)
        X_arm <- X[idx_arm, , drop = FALSE]

        arm_models <- vector("list", self$num_outcomes)
        names(arm_models) <- self$outcome_names

        for (m in seq_len(self$num_outcomes)) {
          y_arm <- Y[idx_arm, m]

          dtrain <- xgboost::xgb.DMatrix(data = X_arm, label = y_arm)

          evals <- list(train = dtrain)

          if (!is.null(eval_data)) {
            X_eval <- as.matrix(eval_data$X)
            if (!is.null(self$imputation_table) && anyNA(X_eval)) {
              X_eval <- .mcb_apply_imputation(X_eval, self$imputation_table)
            }

            W_eval <- eval_data$W
            Y_eval <- eval_data$Y
            if (!is.factor(W_eval)) W_eval <- factor(W_eval, levels = self$treatment_levels)
            if (is.vector(Y_eval)) Y_eval <- matrix(Y_eval, ncol = 1)
            Y_eval <- as.matrix(Y_eval)

            idx_eval_arm <- which(W_eval == arm)
            if (length(idx_eval_arm) > 0L) {
              deval <- xgboost::xgb.DMatrix(
                data = X_eval[idx_eval_arm, , drop = FALSE],
                label = Y_eval[idx_eval_arm, m]
              )
              evals[["eval"]] <- deval
            }
          }

          arm_models[[m]] <- xgboost::xgb.train(
            params = params,
            data = dtrain,
            nrounds = as.integer(nrounds),
            evals = evals,
            verbose = as.integer(verbose >= 2L)
          )
        }

        self$outcome_models[[arm]] <- arm_models
      }

      if (verbose >= 1L) cat("[MultiArmCausalBoost] Fitting propensity model...\n")

      if (is.null(self$propensity_model)) {
        ps_df <- data.frame(W = W, X)

        self$propensity_model <- ranger::ranger(
          W ~ .,
          data = ps_df,
          num.trees = 200L,
          max.depth = 6L,
          probability = TRUE,
          seed = 42L,
          # Quarto renders / CI can be resource-constrained; cap threads.
          num.threads = min(4L, max(1L, parallel::detectCores() - 1L))
        )
        self$oob_error <- self$propensity_model$prediction.error
      } else if (!inherits(self$propensity_model, "ranger")) {
        stop("propensity_model must be NULL or a fitted ranger object.")
      }

      self$fit_time_secs <- as.numeric(difftime(Sys.time(), start, units = "secs"))

      if (verbose >= 1L) {
        cat(sprintf("[MultiArmCausalBoost] Training complete  %.2f sec\n", self$fit_time_secs))
        if (!is.null(self$oob_error)) {
          cat(sprintf("[MultiArmCausalBoost] Propensity OOB error: %.4f\n", self$oob_error))
        }
      }

      invisible(self)
    },

    predict = function(X, baseline = NULL, drop = FALSE) {
      if (is.null(self$outcome_models)) {
        stop("Model not fitted. Call $fit() first.")
      }

      X <- as.matrix(X)

      if (!is.null(self$imputation_table) && anyNA(X)) {
        X <- .mcb_apply_imputation(X, self$imputation_table)
      }

      if (!is.null(self$train_colnames) &&
          !is.null(colnames(X)) &&
          !identical(colnames(X), self$train_colnames)) {
        X <- X[, self$train_colnames, drop = FALSE]
      }

      baseline <- baseline %||% self$baseline
      if (!baseline %in% self$treatment_levels) {
        stop("baseline must be one of the fitted treatment levels.")
      }

      n <- nrow(X)
      K <- length(self$treatment_levels)
      M <- self$num_outcomes

      mu_hat <- array(
        NA_real_,
        dim = c(n, K, M),
        dimnames = list(NULL, self$treatment_levels, self$outcome_names)
      )

      dtest <- xgboost::xgb.DMatrix(data = X)

      for (arm in self$treatment_levels) {
        for (m in seq_len(M)) {
          mu_hat[, arm, m] <- as.numeric(
            predict(self$outcome_models[[arm]][[m]], dtest)
          )
        }
      }

      ps_raw <- predict(self$propensity_model, data = data.frame(X))$predictions

      if (is.null(dim(ps_raw))) {
        ps_mat <- matrix(ps_raw, ncol = 1)
        colnames(ps_mat) <- self$treatment_levels[1]
      } else {
        ps_mat <- ps_raw
      }

      # align propensity columns to training treatment levels
      if (is.null(colnames(ps_mat))) {
        if (ncol(ps_mat) != length(self$treatment_levels)) {
          stop("Propensity prediction columns could not be aligned.")
        }
        colnames(ps_mat) <- self$treatment_levels
      } else {
        ps_mat <- ps_mat[, self$treatment_levels, drop = FALSE]
      }

      ps_mat <- apply(ps_mat, 2, .mcb_clip_probs)
      ps_mat <- matrix(ps_mat, nrow = n, ncol = length(self$treatment_levels),
                       dimnames = list(NULL, self$treatment_levels))

      contrast_levels <- setdiff(self$treatment_levels, baseline)
      tau_hat <- array(
        NA_real_,
        dim = c(n, length(contrast_levels), M),
        dimnames = list(NULL, paste0(contrast_levels, " - ", baseline), self$outcome_names)
      )

      for (j in seq_along(contrast_levels)) {
        arm <- contrast_levels[j]
        tau_hat[, j, ] <- mu_hat[, arm, , drop = FALSE] - mu_hat[, baseline, , drop = FALSE]
      }

      out <- list(
        mu_hat = mu_hat,
        propensity_score = ps_mat,
        tau_hat = tau_hat,
        baseline = baseline
      )

      if (drop) {
        out$mu_hat <- drop(out$mu_hat)
        out$tau_hat <- drop(out$tau_hat)
      }

      out
    },

    summary = function() {
      if (is.null(self$outcome_models)) {
        cat("[MultiArmCausalBoost] Model not yet fitted.\n")
        return(invisible(self))
      }

      cat("\n")
      cat("  MultiArmCausalBoost  Model Summary\n")
      cat("\n\n")

      cat(" Treatments \n")
      cat(sprintf("  Baseline arm         : %s\n", self$baseline))
      cat(sprintf("  Treatment levels     : %s\n", paste(self$treatment_levels, collapse = ", ")))
      cat(sprintf("  Arm counts           : %s\n\n",
                  paste(names(self$arm_counts), as.integer(self$arm_counts),
                        sep = "=", collapse = ", ")))

      cat(" Outcomes \n")
      cat(sprintf("  Number of outcomes   : %d\n", self$num_outcomes))
      cat(sprintf("  Outcome names        : %s\n\n", paste(self$outcome_names, collapse = ", ")))

      cat(" XGBoost parameters \n")
      p <- modifyList(
        list(eta = 0.05, max_depth = 6L, subsample = 0.8,
             colsample_bytree = 0.8, min_child_weight = 1,
             lambda = 1, alpha = 0),
        self$parameters
      )
      cat(sprintf("  eta                  : %.4f\n", p$eta))
      cat(sprintf("  max_depth            : %d\n", p$max_depth))
      cat(sprintf("  subsample            : %.2f\n", p$subsample))
      cat(sprintf("  colsample_bytree     : %.2f\n", p$colsample_bytree))
      cat(sprintf("  lambda               : %.2f\n", p$lambda))
      cat(sprintf("  alpha                : %.2f\n\n", p$alpha))

      cat(" Propensity model \n")
      if (!is.null(self$oob_error)) {
        cat(sprintf("  Ranger OOB error     : %.4f\n", self$oob_error))
      }
      if (!is.null(self$propensity_model)) {
        cat(sprintf("  Ranger num.trees     : %d\n", self$propensity_model$num.trees))
        cat(sprintf("  Ranger max.depth     : %s\n\n",
                    as.character(self$propensity_model$max.depth %||% NA)))
      }

      if (!is.null(self$fit_time_secs)) {
        cat(sprintf("  Fit time             : %.2f sec\n", self$fit_time_secs))
      }

      cat("\n")
      invisible(self)
    },

    evaluate = function(X, mu_true = NULL, tau_true = NULL, baseline = NULL) {
      preds <- self$predict(X, baseline = baseline, drop = FALSE)

      out <- list(
        tau_hat = preds$tau_hat,
        mu_hat = preds$mu_hat,
        propensity_score = preds$propensity_score,
        baseline = preds$baseline
      )

      if (!is.null(mu_true)) {
        mu_true <- as.array(mu_true)
        if (!identical(dim(mu_true), dim(preds$mu_hat))) {
          stop("mu_true must have same dimensions as predicted mu_hat: [n, arms, outcomes].")
        }
        out$mu_mse <- mean((mu_true - preds$mu_hat)^2)
        cat(sprintf("  Potential outcome MSE : %.6f\n", out$mu_mse))
      }

      if (!is.null(tau_true)) {
        tau_true <- as.array(tau_true)
        if (!identical(dim(tau_true), dim(preds$tau_hat))) {
          stop("tau_true must have same dimensions as predicted tau_hat: [n, contrasts, outcomes].")
        }
        out$PEHE <- multi_arm_PEHE(tau_true, preds$tau_hat, rmse = TRUE)
        out$ATE_error <- multi_arm_ATE(tau_true, preds$tau_hat)
        cat(sprintf("  PEHE (RMSE)           : %.6f\n", out$PEHE))
        cat("  ATE absolute error    :\n")
        print(out$ATE_error)
      }

      invisible(out)
    },

    plot_importance = function(arm = NULL, outcome = 1L, top_n = 20L, metric = "Gain") {
      if (is.null(self$outcome_models)) {
        stop("Model not fitted. Call $fit() first.")
      }
      if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("ggplot2 is required. Install it first.")
      }

      arm <- arm %||% self$treatment_levels[1]
      if (!arm %in% self$treatment_levels) {
        stop("arm must be one of the fitted treatment levels.")
      }
      if (outcome < 1L || outcome > self$num_outcomes) {
        stop("Invalid outcome index.")
      }

      booster <- self$outcome_models[[arm]][[outcome]]
      imp <- xgboost::xgb.importance(model = booster)

      if (is.null(imp) || nrow(imp) == 0) {
        stop("No importance information available.")
      }
      if (!metric %in% names(imp)) {
        stop(sprintf("metric must be one of: %s", paste(names(imp), collapse = ", ")))
      }

      imp <- imp[order(-imp[[metric]]), , drop = FALSE]
      top_imp <- head(imp, top_n)
      top_imp$Feature <- factor(top_imp$Feature, levels = rev(top_imp$Feature))

      p <- ggplot2::ggplot(
        top_imp,
        ggplot2::aes(x = .data[[metric]], y = Feature)
      ) +
        ggplot2::geom_col(fill = "#4E79A7", alpha = 0.85) +
        ggplot2::labs(
          title = sprintf("Importance: arm=%s, outcome=%s", arm, self$outcome_names[outcome]),
          x = metric,
          y = NULL
        ) +
        ggplot2::theme_minimal(base_size = 12) +
        ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())

      print(p)
      invisible(p)
    },

    save_model = function(path) {
      saveRDS(self, file = path)
      cat(sprintf("[MultiArmCausalBoost] Model saved to '%s'.\n", path))
      invisible(self)
    },

    clone_reset = function() {
      MultiArmCausalBoost$new(
        parameters = self$parameters,
        propensity_model = NULL,
        baseline = self$baseline
      )
    }
  )
)

# 
# Standalone helpers
# 

#' Load a saved MultiArmCausalBoost model
#' @param path RDS file path
#' @return A MultiArmCausalBoost object
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # load_multi_arm_causal_boost(...)
#' }
#' @export
load_multi_arm_causal_boost <- function(path) {
  model <- readRDS(path)
  if (!inherits(model, "MultiArmCausalBoost")) {
    stop("The file does not contain a MultiArmCausalBoost object.")
  }
  cat(sprintf("[MultiArmCausalBoost] Model loaded from '%s'.\n", path))
  model
}

#' Save a MultiArmCausalBoost model
#' @param model A fitted MultiArmCausalBoost object
#' @param path File path
#' @return
#' Object returned by \code{save_multi_arm_causal_boost}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # save_multi_arm_causal_boost(...)
#' }
#' @export
save_multi_arm_causal_boost <- function(model, path) {
  if (!inherits(model, "MultiArmCausalBoost")) {
    stop("model must be a MultiArmCausalBoost object.")
  }
  saveRDS(model, file = path)
  cat(sprintf("[MultiArmCausalBoost] Model saved to '%s'.\n", path))
  invisible(model)
}

# 
# Example
# 

#' Run a built-in multi-arm causal boost example
#'
#' @param n Sample size
#' @param p Number of features
#' @param nrounds Boosting rounds
#' @return Named list with model, predictions, evaluation
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # run_multi_arm_causal_boost_example(...)
#' }
#' @export
run_multi_arm_causal_boost_example <- function(n = 1200L,
                                               p = 8L,
                                               nrounds = 150L) {
  set.seed(42)

  X <- matrix(rnorm(n * p), n, p,
              dimnames = list(NULL, paste0("X", seq_len(p))))

  logits <- cbind(
    A = 0.2 + 0.3 * X[, 1],
    B = -0.1 + 0.5 * X[, 2],
    C = 0.1 - 0.4 * X[, 1] + 0.2 * X[, 3]
  )
  exp_logits <- exp(logits)
  probs <- exp_logits / rowSums(exp_logits)

  W <- apply(probs, 1, function(pr) sample(colnames(probs), 1, prob = pr))
  W <- factor(W, levels = c("A", "B", "C"))

  mu_A <- X[, 1] + 0.5 * X[, 3]
  mu_B <- mu_A + 1 + X[, 2]
  mu_C <- mu_A - 1.5 * X[, 2] + 0.5 * X[, 4]

  Y <- ifelse(W == "A", mu_A,
              ifelse(W == "B", mu_B, mu_C)) + rnorm(n)

  mu_true <- array(NA_real_, dim = c(n, 3, 1),
                   dimnames = list(NULL, c("A", "B", "C"), "Y.1"))
  mu_true[, "A", 1] <- mu_A
  mu_true[, "B", 1] <- mu_B
  mu_true[, "C", 1] <- mu_C

  tau_true <- array(NA_real_, dim = c(n, 2, 1),
                    dimnames = list(NULL, c("B - A", "C - A"), "Y.1"))
  tau_true[, "B - A", 1] <- mu_B - mu_A
  tau_true[, "C - A", 1] <- mu_C - mu_A

  model <- MultiArmCausalBoost$new(
    parameters = list(eta = 0.05, max_depth = 5L),
    baseline = "A"
  )

  model$fit(X, Y, W, nrounds = nrounds, verbose = 1L)
  model$summary()

  preds <- model$predict(X, drop = FALSE)
  eval <- model$evaluate(X, mu_true = mu_true, tau_true = tau_true, baseline = "A")

  invisible(list(model = model, preds = preds, eval = eval))
}
