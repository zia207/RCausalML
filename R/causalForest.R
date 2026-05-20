# CausalForest.R
# A from-scratch implementation of Causal Forest (Wager & Athey, 2018)
# Estimates heterogeneous treatment effects via honest causal trees

# ============================================================
# UTILITIES
# ============================================================

#' Compute propensity scores via logistic regression or provided values
.estimate_propensity <- function(X, W) {
  fit <- glm(W ~ ., data = data.frame(W = W, X), family = binomial())
  predict(fit, type = "response")
}

#' Compute outcome residuals (Y - E[Y|X])
.estimate_outcome <- function(X, Y) {
  fit <- lm(Y ~ ., data = data.frame(Y = Y, X))
  residuals(fit)
}

#' Sample split: return indices for two non-overlapping halves
.honest_split <- function(n, fraction = 0.5) {
  idx <- sample(seq_len(n))
  split_at <- floor(n * fraction)
  list(
    train = idx[seq_len(split_at)],
    est   = idx[(split_at + 1):n]
  )
}

#' Compute sample weights (uniform if NULL)
.resolve_weights <- function(weights, n) {
  if (is.null(weights)) rep(1.0 / n, n) else weights / sum(weights)
}

#' Variance of a weighted vector
.weighted_var <- function(x, w) {
  mu <- sum(w * x) / sum(w)
  sum(w * (x - mu)^2) / sum(w)
}

# ============================================================
# CAUSAL TREE (single tree)
# ============================================================

#' Build one honest causal tree
#'
#' @param X          Covariate matrix (numeric)
#' @param Y_res      Outcome residuals (Y - Y.hat)
#' @param W_res      Treatment residuals (W - W.hat)
#' @param sample_wts Per-observation weights
#' @param train_idx  Indices used for splitting
#' @param est_idx    Indices used for leaf estimation
#' @param mtry       Features to consider at each split
#' @param min_node   Minimum leaf size
#' @param alpha      Regularization: min fraction of data in each child
#' @param imbalance_penalty Penalizes unbalanced splits
#' @return A list representing the tree (recursive node structure)
.build_causal_tree <- function(X, Y_res, W_res, sample_wts,
                                train_idx, est_idx,
                                mtry, min_node, alpha,
                                imbalance_penalty) {

  n_train <- length(train_idx)
  n_est   <- length(est_idx)

  # --- Leaf node: estimate tau via local Wald-style estimate ---
  .make_leaf <- function(est_idx) {
    w  <- sample_wts[est_idx]
    yr <- Y_res[est_idx]
    wr <- W_res[est_idx]
    denom <- sum(w * wr^2)
    tau   <- if (abs(denom) < 1e-10) 0 else sum(w * yr * wr) / denom
    var_tau <- if (length(est_idx) < 2) Inf else {
      eps <- yr - tau * wr
      sum(w^2 * eps^2) / (denom^2 + 1e-10)
    }
    list(
      is_leaf  = TRUE,
      tau      = tau,
      var_tau  = var_tau,
      n_est    = n_est,
      est_idx  = est_idx
    )
  }

  # Stop early?
  if (n_train < 2 * min_node || n_est < 2) {
    return(.make_leaf(est_idx))
  }

  # --- Select random feature subset ---
  p        <- ncol(X)
  features <- sample(seq_len(p), min(mtry, p))

  best_gain  <- -Inf
  best_feat  <- NULL
  best_thresh <- NULL

  # --- Score a candidate split ---
  .split_score <- function(left_tr, right_tr) {
    if (length(left_tr) < min_node || length(right_tr) < min_node) return(-Inf)

    # Fraction-based balance penalty
    f_left  <- length(left_tr) / n_train
    f_right <- length(right_tr) / n_train
    if (f_left < alpha || f_right < alpha) return(-Inf)

    score_side <- function(idx) {
      w  <- sample_wts[idx]
      wr <- W_res[idx]
      yr <- Y_res[idx]
      denom <- sum(w * wr^2)
      if (denom < 1e-10) return(0)
      (sum(w * yr * wr))^2 / denom
    }

    gain <- score_side(left_tr) + score_side(right_tr)

    # Imbalance penalty discourages very unequal splits
    if (imbalance_penalty > 0) {
      imbalance <- abs(f_left - f_right)
      gain <- gain - imbalance_penalty * imbalance
    }

    gain
  }

  # --- Search over features and thresholds ---
  for (feat in features) {
    vals    <- X[train_idx, feat]
    uniq    <- sort(unique(vals))
    if (length(uniq) < 2) next

    # Candidate thresholds: midpoints between sorted unique values
    thresholds <- (uniq[-length(uniq)] + uniq[-1]) / 2

    for (thresh in thresholds) {
      left_tr  <- train_idx[vals <= thresh]
      right_tr <- train_idx[vals >  thresh]
      gain     <- .split_score(left_tr, right_tr)

      if (gain > best_gain) {
        best_gain   <- gain
        best_feat   <- feat
        best_thresh <- thresh
      }
    }
  }

  # No valid split found
  if (is.null(best_feat)) return(.make_leaf(est_idx))

  # --- Partition estimation set by best split ---
  est_vals   <- X[est_idx, best_feat]
  left_est   <- est_idx[est_vals <= best_thresh]
  right_est  <- est_idx[est_vals >  best_thresh]

  train_vals  <- X[train_idx, best_feat]
  left_train  <- train_idx[train_vals <= best_thresh]
  right_train <- train_idx[train_vals >  best_thresh]

  if (length(left_est) == 0 || length(right_est) == 0) {
    return(.make_leaf(est_idx))
  }

  # --- Recurse ---
  list(
    is_leaf    = FALSE,
    feature    = best_feat,
    threshold  = best_thresh,
    left       = .build_causal_tree(X, Y_res, W_res, sample_wts,
                                     left_train, left_est,
                                     mtry, min_node, alpha,
                                     imbalance_penalty),
    right      = .build_causal_tree(X, Y_res, W_res, sample_wts,
                                     right_train, right_est,
                                     mtry, min_node, alpha,
                                     imbalance_penalty)
  )
}

#' Route a single observation through a tree, return leaf node
.predict_tree <- function(tree, x_row) {
  node <- tree
  while (!node$is_leaf) {
    if (x_row[node$feature] <= node$threshold) {
      node <- node$left
    } else {
      node <- node$right
    }
  }
  node
}

# ============================================================
# MAIN: causal_forest()
# ============================================================

#' Fit a Causal Forest
#'
#' @param X                    Numeric covariate matrix (n x p)
#' @param Y                    Numeric outcome vector (length n)
#' @param W                    Binary or continuous treatment vector (length n)
#' @param Y.hat                Pre-estimated E[Y|X]; estimated if NULL
#' @param W.hat                Pre-estimated E[W|X]; estimated if NULL
#' @param num.trees            Number of trees
#' @param sample.weights       Observation weights; uniform if NULL
#' @param clusters             Cluster IDs for cluster-robust sampling (optional)
#' @param equalize.cluster.weights  Balance sampling across clusters
#' @param sample.fraction      Fraction of data per tree
#' @param mtry                 Features tried per split
#' @param min.node.size        Minimum observations in a leaf
#' @param honesty              Use honest splitting
#' @param honesty.fraction     Fraction of subsample used for estimation
#' @param honesty.prune.leaves Remove leaves with no estimation samples
#' @param alpha                Minimum split fraction (regularization)
#' @param imbalance.penalty    Penalty on unbalanced splits
#' @param stabilize.splits     Use W residuals in split criterion
#' @param ci.group.size        Trees grouped for variance estimation
#' @param tune.parameters      "none" or "all"
#' @param compute.oob.predictions  Compute out-of-bag CATE predictions
#' @param num.threads          Parallel threads (uses parallel package if > 1)
#' @param seed                 Random seed for reproducibility
#' @return A causal_forest object
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # causal_forest(...)
#' }
#' @export
causal_forest <- function(X, Y, W,
                           Y.hat                     = NULL,
                           W.hat                     = NULL,
                           num.trees                 = 2000,
                           sample.weights            = NULL,
                           clusters                  = NULL,
                           equalize.cluster.weights  = FALSE,
                           sample.fraction           = 0.5,
                           mtry                      = min(ceiling(sqrt(ncol(X)) + 20), ncol(X)),
                           min.node.size             = 5,
                           honesty                   = TRUE,
                           honesty.fraction          = 0.5,
                           honesty.prune.leaves      = TRUE,
                           alpha                     = 0.05,
                           imbalance.penalty         = 0,
                           stabilize.splits          = TRUE,
                           ci.group.size             = 2,
                           tune.parameters           = "none",
                           compute.oob.predictions   = TRUE,
                           num.threads               = 1,
                           seed                      = NULL) {

  # --- Input validation ---
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  W <- as.numeric(W)
  n <- nrow(X)
  p <- ncol(X)

  stopifnot(length(Y) == n, length(W) == n)
  stopifnot(sample.fraction > 0, sample.fraction <= 1)
  stopifnot(honesty.fraction > 0, honesty.fraction < 1)
  stopifnot(min.node.size >= 1)
  stopifnot(alpha >= 0, alpha < 0.5)

  if (!is.null(seed)) set.seed(seed)

  # --- Nuisance estimation (DR / Robinson decomposition) ---
  message("Estimating nuisance functions...")

  if (is.null(Y.hat)) {
    Y.hat <- fitted(lm(Y ~ ., data = data.frame(Y, X)))
  }
  if (is.null(W.hat)) {
    W.hat <- .estimate_propensity(X, W)
  }

  Y_res <- Y - Y.hat   # outcome residuals
  W_res <- W - W.hat   # treatment residuals

  # --- Sample weights ---
  sw <- .resolve_weights(sample.weights, n)

  # --- Optional: tune hyperparameters ---
  if (tune.parameters != "none") {
    message("Parameter tuning not fully implemented; using supplied parameters.")
  }

  # --- Cluster-aware subsampling helper ---
  .draw_subsample <- function() {
    sub_n <- max(1, floor(n * sample.fraction))

    if (!is.null(clusters)) {
      cl_ids   <- unique(clusters)
      if (equalize.cluster.weights) {
        chosen_cl <- sample(cl_ids, size = ceiling(length(cl_ids) * sample.fraction),
                            replace = FALSE)
      } else {
        chosen_cl <- sample(cl_ids, size = ceiling(length(cl_ids) * sample.fraction),
                            replace = FALSE)
      }
      idx <- which(clusters %in% chosen_cl)
    } else {
      idx <- sample(seq_len(n), size = sub_n, replace = FALSE)
    }

    if (honesty) {
      split    <- .honest_split(length(idx), honesty.fraction)
      train_idx <- idx[split$train]
      est_idx   <- idx[split$est]
    } else {
      train_idx <- idx
      est_idx   <- idx
    }

    list(train = train_idx, est = est_idx, all = idx)
  }

  # --- Grow forest ---
  message(sprintf("Growing %d causal trees...", num.trees))

  grow_one_tree <- function(b) {
    samp      <- .draw_subsample()
    tree      <- .build_causal_tree(
      X             = X,
      Y_res         = Y_res,
      W_res         = W_res,
      sample_wts    = sw,
      train_idx     = samp$train,
      est_idx       = samp$est,
      mtry          = mtry,
      min_node      = min.node.size,
      alpha         = alpha,
      imbalance_penalty = imbalance.penalty
    )
    list(tree = tree, oob_idx = setdiff(seq_len(n), samp$all))
  }

  if (num.threads > 1 && requireNamespace("parallel", quietly = TRUE)) {
    cl_par <- parallel::makeCluster(num.threads)
    on.exit(parallel::stopCluster(cl_par), add = TRUE)
    parallel::clusterExport(cl_par,
      c(".build_causal_tree", ".honest_split", ".resolve_weights",
        ".weighted_var", ".draw_subsample",
        "X", "Y_res", "W_res", "sw", "n",
        "sample.fraction", "honesty", "honesty.fraction",
        "mtry", "min.node.size", "alpha", "imbalance.penalty",
        "clusters", "equalize.cluster.weights"),
      envir = environment())
    trees_list <- parallel::parLapply(cl_par, seq_len(num.trees), grow_one_tree)
  } else {
    trees_list <- lapply(seq_len(num.trees), grow_one_tree)
  }

  trees    <- lapply(trees_list, `[[`, "tree")
  oob_sets <- lapply(trees_list, `[[`, "oob_idx")

  # --- OOB predictions ---
  oob_predictions <- rep(NA_real_, n)
  oob_counts      <- rep(0L, n)

  if (compute.oob.predictions) {
    message("Computing OOB predictions...")
    for (i in seq_along(trees)) {
      oob_i <- oob_sets[[i]]
      if (length(oob_i) == 0) next
      for (j in oob_i) {
        leaf <- .predict_tree(trees[[i]], X[j, ])
        if (!is.null(leaf$tau) && is.finite(leaf$tau)) {
          oob_predictions[j] <- sum(c(oob_predictions[j], leaf$tau), na.rm = TRUE)
          oob_counts[j]      <- oob_counts[j] + 1L
        }
      }
    }
    nonzero <- oob_counts > 0
    oob_predictions[nonzero] <- oob_predictions[nonzero] / oob_counts[nonzero]
  }

  # --- Return forest object ---
  structure(
    list(
      trees              = trees,
      oob_sets           = oob_sets,
      oob_predictions    = oob_predictions,
      X                  = X,
      Y                  = Y,
      W                  = W,
      Y.hat              = Y.hat,
      W.hat              = W.hat,
      Y_res              = Y_res,
      W_res              = W_res,
      sample_weights     = sw,
      num.trees          = num.trees,
      mtry               = mtry,
      min.node.size      = min.node.size,
      honesty            = honesty,
      honesty.fraction   = honesty.fraction,
      alpha              = alpha,
      imbalance.penalty  = imbalance.penalty,
      ci.group.size      = ci.group.size,
      n                  = n,
      p                  = p
    ),
    class = "causal_forest"
  )
}

# ============================================================
# PREDICT: predict.causal_forest()
# ============================================================

#' Predict treatment effects from a causal forest
#'
#' @param object       A causal_forest object
#' @param newdata      New covariate matrix; uses training data if NULL
#' @param estimate.variance  Return variance estimates alongside predictions
#' @param num.threads  Threads for parallel prediction
#' @param ...          Ignored
#' @return A list with `predictions` and optionally `variance.estimates`
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.causal_forest(...)
#' }
#' @export
predict.causal_forest <- function(object, newdata = NULL,
                                   estimate.variance = FALSE,
                                   num.threads = 1,
                                   ...) {

  if (is.null(newdata)) {
    # Return OOB predictions for training data
    if (estimate.variance) {
      message("Variance estimates use grouped-tree IJ method on OOB predictions.")
    }
    return(list(
      predictions        = object$oob_predictions,
      variance.estimates = if (estimate.variance)
        .compute_oob_variance(object) else NULL
    ))
  }

  X_new <- as.matrix(newdata)
  stopifnot(ncol(X_new) == object$p)
  m     <- nrow(X_new)
  trees <- object$trees
  B     <- length(trees)

  # --- Predict each new obs across all trees ---
  pred_matrix <- matrix(NA_real_, nrow = m, ncol = B)

  predict_one <- function(b) {
    sapply(seq_len(m), function(i) {
      leaf <- .predict_tree(trees[[b]], X_new[i, ])
      leaf$tau
    })
  }

  if (num.threads > 1 && requireNamespace("parallel", quietly = TRUE)) {
    cl_par <- parallel::makeCluster(num.threads)
    on.exit(parallel::stopCluster(cl_par), add = TRUE)
    parallel::clusterExport(cl_par,
      c(".predict_tree", "trees", "X_new", "m"),
      envir = environment())
    results <- parallel::parLapply(cl_par, seq_len(B), predict_one)
  } else {
    results <- lapply(seq_len(B), predict_one)
  }

  for (b in seq_len(B)) pred_matrix[, b] <- results[[b]]

  predictions <- rowMeans(pred_matrix, na.rm = TRUE)

  # --- Variance via grouped infinitesimal jackknife (IJ) ---
  var_estimates <- NULL
  if (estimate.variance) {
    var_estimates <- .compute_variance_ij(pred_matrix, object$ci.group.size)
  }

  list(
    predictions        = predictions,
    variance.estimates = var_estimates
  )
}

# ============================================================
# VARIANCE ESTIMATION (grouped IJ)
# ============================================================

#' Variance estimation for new-data predictions via grouped IJ
.compute_variance_ij <- function(pred_matrix, ci.group.size) {
  m <- nrow(pred_matrix)
  B <- ncol(pred_matrix)

  # Group trees into blocks for IJ variance estimate
  n_groups <- floor(B / ci.group.size)
  if (n_groups < 2) {
    warning("Too few trees for grouped variance estimation. Increase num.trees.")
    return(rep(NA_real_, m))
  }

  group_means <- matrix(NA_real_, nrow = m, ncol = n_groups)
  for (g in seq_len(n_groups)) {
    cols <- ((g - 1) * ci.group.size + 1):(g * ci.group.size)
    group_means[, g] <- rowMeans(pred_matrix[, cols, drop = FALSE], na.rm = TRUE)
  }

  # Variance across group means, scaled by number of groups
  apply(group_means, 1, var, na.rm = TRUE) / n_groups
}

#' OOB variance estimate (used when newdata = NULL)
.compute_oob_variance <- function(object) {
  n     <- object$n
  trees <- object$trees
  oob   <- object$oob_sets
  B     <- length(trees)
  ci_g  <- object$ci.group.size
  X     <- object$X

  oob_mat <- matrix(NA_real_, nrow = n, ncol = B)
  for (b in seq_len(B)) {
    for (j in oob[[b]]) {
      leaf <- .predict_tree(trees[[b]], X[j, ])
      oob_mat[j, b] <- leaf$tau
    }
  }
  .compute_variance_ij(oob_mat, ci_g)
}

# ============================================================
# SUMMARY / PRINT
# ============================================================

#' Print method for causal_forest
#' @return
#' Object returned by \code{print.causal_forest}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # print.causal_forest(...)
#' }
#' @export
print.causal_forest <- function(x, ...) {
  cat("Causal Forest\n")
  cat("-------------\n")
  cat(sprintf("  Trees         : %d\n", x$num.trees))
  cat(sprintf("  Observations  : %d\n", x$n))
  cat(sprintf("  Covariates    : %d\n", x$p))
  cat(sprintf("  Honesty       : %s (fraction = %.2f)\n",
              x$honesty, x$honesty.fraction))
  cat(sprintf("  min.node.size : %d\n", x$min.node.size))
  cat(sprintf("  mtry          : %d\n", x$mtry))
  cat(sprintf("  alpha         : %.3f\n", x$alpha))

  oob_valid <- sum(!is.na(x$oob_predictions))
  cat(sprintf("  OOB preds     : %d / %d obs covered\n", oob_valid, x$n))

  if (oob_valid > 0) {
    cat(sprintf("  ATE (OOB est) : %.4f\n", mean(x$oob_predictions, na.rm = TRUE)))
  }
  invisible(x)
}

#' Summary method for causal_forest
#' @return
#' Object returned by \code{summary.causal_forest}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # summary.causal_forest(...)
#' }
#' @export
summary.causal_forest <- function(object, ...) {
  cat("=== Causal Forest Summary ===\n\n")
  print(object)

  if (any(!is.na(object$oob_predictions))) {
    cat("\nOOB Prediction Distribution:\n")
    print(summary(object$oob_predictions))
  }
  invisible(object)
}

# ============================================================
# AVERAGE TREATMENT EFFECT
# ============================================================

#' Estimate Average Treatment Effect from a causal forest
#'
#' Uses augmented IPW (AIPW) / doubly robust estimator
#'
#' @param forest  A causal_forest object
#' @param subset  Optional integer index for subgroup ATE
#' @return A list with estimate, std.err, t-stat, p-value, confidence interval
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # average_treatment_effect(...)
#' }
#' @export
average_treatment_effect <- function(forest, subset = NULL) {
  n      <- forest$n
  idx    <- if (is.null(subset)) seq_len(n) else subset
  tau_hat <- forest$oob_predictions[idx]
  Y_res  <- forest$Y_res[idx]
  W_res  <- forest$W_res[idx]

  # Doubly-robust AIPW score
  dr_scores <- tau_hat + (W_res * (Y_res - tau_hat * W_res)) /
                 pmax(forest$W.hat[idx] * (1 - forest$W.hat[idx]), 1e-6)

  ate     <- mean(dr_scores, na.rm = TRUE)
  se      <- sd(dr_scores, na.rm = TRUE) / sqrt(sum(!is.na(dr_scores)))
  t_stat  <- ate / se
  p_val   <- 2 * pnorm(-abs(t_stat))
  ci      <- ate + c(-1, 1) * 1.96 * se

  list(
    estimate   = ate,
    std.err    = se,
    t.stat     = t_stat,
    p.value    = p_val,
    conf.int   = ci
  )
}

# ============================================================
# VARIABLE IMPORTANCE
# ============================================================

#' Variable importance from split frequency
#'
#' @param forest  A causal_forest object
#' @return Named numeric vector of importance scores (sums to 1)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # variable_importance(...)
#' }
#' @export
variable_importance <- function(forest) {
  counts <- integer(forest$p)

  .count_splits <- function(node) {
    if (node$is_leaf) return(invisible(NULL))
    counts[node$feature] <<- counts[node$feature] + 1L
    .count_splits(node$left)
    .count_splits(node$right)
  }

  for (tree in forest$trees) .count_splits(tree)

  total <- sum(counts)
  if (total == 0) return(setNames(rep(0, forest$p), colnames(forest$X)))

  importance <- counts / total
  col_names  <- if (!is.null(colnames(forest$X))) colnames(forest$X) else
                  paste0("X", seq_len(forest$p))
  setNames(importance, col_names)
}

# =============================================================================
# Graph Neural Network (GNN) Causal Models
# (GVAR, CausalGNN / CD-GNN, CUTS+)
# Translated from 05_graph_nn_causal_models_GNN.ipynb
# =============================================================================
# Model summaries:
#   • GVAR      — Graph Vector Autoregression: lag-specific soft adjacency
#                 matrices A^(k) ∈ [0,1]^{d×d} (sigmoid-gated logits, no
#                 self-loops) with L1 sparsity + NOTEARS DAG penalties.  Two
#                 stacked GNN message-passing layers with per-lag linear
#                 transforms and a GELU output head.
#                 Public API: gvar_model() / GVARModel().
#   • CausalGNN — CD-GNN (Causal Discovery GNN): GRU per-variable temporal
#                 encoder → bilinear-scoring graph learner (sigmoid, no
#                 self-loops) → stacked edge-conditioned message-passing layers
#                 (single-step GRU node update + LayerNorm) with NOTEARS DAG +
#                 sparsity penalties.
#                 Public API: causal_gnn_model() / CausalGNNModel().
#   • CUTS+     — Causal discovery Under missing Time Series: variational
#                 Bernoulli graph posterior q(G) ∼ ∏ Bernoulli(π_ij), joint
#                 imputation network (concat values + missing mask), GRU
#                 temporal encoder, edge-conv message passing; trained with
#                 MSE + KL(q ‖ sparse prior p=0.1) + NOTEARS DAG penalty.
#                 Public API: cuts_model() / CUTSModel().
#   Unified entry point:   gnn_causal_model() / gnnCausalModel()
#   Causal graph accessor: causal_matrix_gnn()
#   Prediction:            predict.gnn_causal_model()
# =============================================================================

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

.deepnet_gnn_select_device <- function(device) {
  if (!is.null(device)) return(device)
  if (torch::cuda_is_available()) "cuda" else "cpu"
}

.deepnet_gnn_acyclicity <- function(A) {
  d_local <- A$shape[1]
  torch::torch_trace(torch::torch_matrix_exp(A * A)) - d_local
}

.deepnet_gnn_build_dataset <- function(x, lag, ahead = 1L) {
  x     <- as.matrix(x)
  n_tot <- nrow(x)
  d     <- ncol(x)
  n     <- n_tot - lag - ahead + 1L
  if (n < 1L) stop("`lag` is too large for the supplied data.", call. = FALSE)
  X_arr <- array(0.0, c(n, lag, d))
  Y_mat <- matrix(0.0, n, d)
  for (t in seq_len(n)) {
    X_arr[t, , ] <- x[t:(t + lag - 1L), ]
    Y_mat[t, ]   <- x[t + lag + ahead - 1L, ]
  }
  list(X = X_arr, Y = Y_mat)
}

# ---------------------------------------------------------------------------
# Module: GVARGraphLearner — lag-specific soft adjacency (lag, d, d)
# ---------------------------------------------------------------------------

.deepnet_gnn_gvar_graph_learner <- function(d, lag) {
  d   <- as.integer(d)
  lag <- as.integer(lag)
  torch::nn_module(
    "GVARGraphLearner",
    initialize = function() {
      self$d      <- d
      self$lag    <- lag
      self$logits <- torch::nn_parameter(
        torch::torch_randn(c(lag, d, d)) * 0.1
      )
    },
    forward = function() {
      A   <- torch::torch_sigmoid(self$logits)
      eye <- torch::torch_eye(d, device = A$device)$unsqueeze(1L)  # (1, d, d)
      A * (1.0 - eye)
    },
    sparsity_loss = function(lam = 0.01) {
      lam * self$forward()$sum()
    },
    aggregate_causal = function() {
      self$forward()$detach()$cpu()
    }
  )
}

# ---------------------------------------------------------------------------
# Module: GVARMessagePass — per-lag feature transforms + adjacency aggregation
# ---------------------------------------------------------------------------

.deepnet_gnn_gvar_msg_pass <- function(d, lag, in_feat, out_feat) {
  d        <- as.integer(d)
  lag      <- as.integer(lag)
  in_feat  <- as.integer(in_feat)
  out_feat <- as.integer(out_feat)
  torch::nn_module(
    "GVARMessagePass",
    initialize = function() {
      self$lag <- lag
      self$lag_transforms <- torch::nn_module_list(
        lapply(seq_len(lag), function(i)
          torch::nn_sequential(
            torch::nn_linear(in_feat, out_feat),
            torch::nn_relu()
          )
        )
      )
      self$agg <- torch::nn_linear(out_feat * lag, out_feat)
    },
    forward = function(x_seq, A) {
      # x_seq: (batch, lag, d, in_feat)  A: (lag, d, d)
      batch <- x_seq$size(1L)
      msgs  <- vector("list", lag)
      for (k in seq_len(lag)) {
        h_k      <- self$lag_transforms[[k]](x_seq[, k, , ])
        A_k      <- A[k, , ]
        msgs[[k]] <- torch::torch_bmm(
          A_k$unsqueeze(1L)$expand(c(batch, -1L, -1L)), h_k
        )
      }
      out <- torch::torch_cat(msgs, dim = -1L)
      self$agg(out)
    }
  )
}

# ---------------------------------------------------------------------------
# Module: GVAR — full Graph VAR with two stacked message-passing layers
# ---------------------------------------------------------------------------

.deepnet_gnn_gvar_module <- function(d, lag, hidden = 32L,
                                      lam_sparse = 0.005, lam_dag = 0.1) {
  d          <- as.integer(d)
  lag        <- as.integer(lag)
  hidden     <- as.integer(hidden)
  GVARGraph  <- .deepnet_gnn_gvar_graph_learner(d = d, lag = lag)
  GVARMsg    <- .deepnet_gnn_gvar_msg_pass(d = d, lag = lag,
                                            in_feat = hidden, out_feat = hidden)
  torch::nn_module(
    "GVAR",
    initialize = function() {
      self$d           <- d
      self$lag         <- lag
      self$lam_sparse  <- lam_sparse
      self$lam_dag     <- lam_dag
      self$graph       <- GVARGraph()
      self$input_proj  <- torch::nn_linear(1L, hidden)
      self$gnn1        <- GVARMsg()
      self$gnn2        <- GVARMsg()
      self$output_head <- torch::nn_sequential(
        torch::nn_linear(hidden, hidden),
        torch::nn_gelu(),
        torch::nn_linear(hidden, 1L)
      )
    },
    forward = function(x_seq) {
      batch    <- x_seq$size(1L)
      A        <- self$graph()
      x_feat   <- self$input_proj(x_seq$unsqueeze(-1L))
      h1       <- self$gnn1(x_feat, A)
      x_feat2  <- h1$unsqueeze(2L)$expand(c(-1L, lag, -1L, -1L))
      h2       <- self$gnn2(x_feat2, A)
      pred     <- self$output_head(h2)$squeeze(-1L)
      sparse_l <- self$graph$sparsity_loss(self$lam_sparse)
      dag_l    <- self$lam_dag * .deepnet_gnn_acyclicity(A$mean(dim = 1L))
      list(pred = pred, A = A$detach(),
           sparse_loss = sparse_l, dag_loss = dag_l)
    },
    causal_matrix = function() {
      torch::with_no_grad({
        self$graph$aggregate_causal()$sum(dim = 1L)
      })
    }
  )
}

# ---------------------------------------------------------------------------
# Module: EdgeConvLayer — edge-conditioned messages + single-step GRU update
# ---------------------------------------------------------------------------

.deepnet_gnn_edge_conv_layer <- function(node_dim, out_dim) {
  node_dim <- as.integer(node_dim)
  out_dim  <- as.integer(out_dim)
  torch::nn_module(
    "EdgeConvLayer",
    initialize = function() {
      self$msg_net <- torch::nn_sequential(
        torch::nn_linear(node_dim * 2L + 1L, out_dim * 2L),
        torch::nn_relu(),
        torch::nn_linear(out_dim * 2L, out_dim)
      )
      self$update_gru <- torch::nn_gru(
        input_size = out_dim, hidden_size = node_dim,
        num_layers = 1L, batch_first = TRUE
      )
      self$norm <- torch::nn_layer_norm(node_dim)
    },
    forward = function(h, A) {
      # h: (batch, d, node_dim)   A: (d, d) or (batch, d, d)
      batch <- h$size(1L)
      d_n   <- h$size(2L)
      if (A$dim() == 2L)
        A <- A$unsqueeze(1L)$expand(c(batch, -1L, -1L))
      h_i    <- h$unsqueeze(3L)$expand(c(-1L, -1L, d_n, -1L))
      h_j    <- h$unsqueeze(2L)$expand(c(-1L, d_n, -1L, -1L))
      e_ij   <- A$unsqueeze(4L)
      msg_in <- torch::torch_cat(list(h_i, h_j, e_ij), dim = 4L)
      msgs   <- self$msg_net(msg_in)
      agg    <- (A$unsqueeze(4L) * msgs)$sum(dim = 3L)
      h_flat   <- h$reshape(c(batch * d_n, -1L))
      agg_flat <- agg$reshape(c(batch * d_n, -1L))
      gru_out  <- self$update_gru(
        agg_flat$unsqueeze(2L),
        h_flat$unsqueeze(1L)
      )
      h_new <- gru_out[[1L]]$squeeze(2L)$reshape(c(batch, d_n, -1L))
      self$norm(h_new)
    }
  )
}

# ---------------------------------------------------------------------------
# Module: CausalGraphLearner — bilinear-style soft adjacency from summaries
# ---------------------------------------------------------------------------

.deepnet_gnn_causal_graph_learner <- function(d, hidden = 32L) {
  d      <- as.integer(d)
  hidden <- as.integer(hidden)
  torch::nn_module(
    "CausalGraphLearner",
    initialize = function() {
      self$d          <- d
      self$node_embed <- torch::nn_sequential(
        torch::nn_linear(1L, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, hidden)
      )
      self$edge_score <- torch::nn_linear(hidden * 2L, 1L)
    },
    forward = function(x_summary) {
      x_mean <- x_summary$mean(dim = 1L, keepdim = TRUE)
      h      <- self$node_embed(x_mean$t()$unsqueeze(-1L))$squeeze(2L)
      h_i    <- h$unsqueeze(2L)$expand(c(-1L, d, -1L))
      h_j    <- h$unsqueeze(1L)$expand(c(d, -1L, -1L))
      logits <- self$edge_score(
        torch::torch_cat(
          list(h_i$reshape(c(-1L, hidden)),
               h_j$reshape(c(-1L, hidden))),
          dim = -1L
        )
      )$reshape(c(d, d))
      A   <- torch::torch_sigmoid(logits)
      eye <- torch::torch_eye(d, device = A$device)
      A * (1.0 - eye)
    }
  )
}

# ---------------------------------------------------------------------------
# Module: CausalGNN — GRU encoder + graph learner + edge-conv GNN layers
# ---------------------------------------------------------------------------

.deepnet_gnn_causal_gnn_module <- function(d, lag, hidden = 32L,
                                             n_gnn_layers = 3L,
                                             lam_dag = 0.5, lam_sparse = 0.01) {
  d            <- as.integer(d)
  lag          <- as.integer(lag)
  hidden       <- as.integer(hidden)
  n_gnn_layers <- as.integer(n_gnn_layers)
  GraphLearner <- .deepnet_gnn_causal_graph_learner(d = d, hidden = hidden)
  EdgeConv     <- .deepnet_gnn_edge_conv_layer(node_dim = hidden, out_dim = hidden)
  torch::nn_module(
    "CausalGNN",
    initialize = function() {
      self$d            <- d
      self$lag          <- lag
      self$lam_dag      <- lam_dag
      self$lam_sparse   <- lam_sparse
      self$graph_learner <- GraphLearner()
      self$temporal_rnn  <- torch::nn_gru(
        input_size = 1L, hidden_size = hidden,
        num_layers = 1L, batch_first = TRUE
      )
      self$gnn_layers <- torch::nn_module_list(
        lapply(seq_len(n_gnn_layers), function(i) EdgeConv())
      )
      self$output_fc <- torch::nn_sequential(
        torch::nn_linear(hidden, hidden),
        torch::nn_gelu(),
        torch::nn_linear(hidden, 1L)
      )
    },
    forward = function(x_seq) {
      batch   <- x_seq$size(1L)
      x_var   <- x_seq$permute(c(1L, 3L, 2L))$reshape(c(batch * d, lag, 1L))
      rnn_out <- self$temporal_rnn(x_var)
      h       <- rnn_out[[2L]]$squeeze(1L)$reshape(c(batch, d, -1L))
      x_sum   <- x_seq$mean(dim = 2L)
      A       <- self$graph_learner(x_sum)
      h_gnn   <- h
      for (i in seq_len(n_gnn_layers))
        h_gnn <- self$gnn_layers[[i]](h_gnn, A)
      pred        <- self$output_fc(h_gnn)$squeeze(-1L)
      dag_loss    <- self$lam_dag * .deepnet_gnn_acyclicity(A)
      sparse_loss <- self$lam_sparse * A$sum()
      list(pred = pred, A = A$detach(),
           dag_loss = dag_loss, sparse_loss = sparse_loss)
    },
    causal_matrix = function(x_ref) {
      torch::with_no_grad({ self$forward(x_ref)$A$cpu() })
    }
  )
}

# ---------------------------------------------------------------------------
# Module: CUTSPlusLike — variational Bernoulli graph + imputation + GRU
# ---------------------------------------------------------------------------

.deepnet_gnn_cuts_module <- function(d, lag, hidden = 32L,
                                      lam_kl = 1e-3, lam_dag = 0.2) {
  d      <- as.integer(d)
  lag    <- as.integer(lag)
  hidden <- as.integer(hidden)
  EdgeConv <- .deepnet_gnn_edge_conv_layer(node_dim = hidden, out_dim = hidden)
  torch::nn_module(
    "CUTSPlusLike",
    initialize = function() {
      self$d       <- d
      self$lag     <- lag
      self$lam_kl  <- lam_kl
      self$lam_dag <- lam_dag
      self$logits  <- torch::nn_parameter(torch::torch_zeros(c(d, d)))
      self$imputer <- torch::nn_sequential(
        torch::nn_linear(lag * d * 2L, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, lag * d)
      )
      self$encoder <- torch::nn_gru(
        input_size = 1L, hidden_size = hidden,
        num_layers = 1L, batch_first = TRUE
      )
      self$msg      <- EdgeConv()
      self$out_head <- torch::nn_sequential(
        torch::nn_linear(hidden, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, 1L)
      )
    },
    sample_graph = function() {
      pi  <- torch::torch_sigmoid(self$logits)
      eye <- torch::torch_eye(d, device = pi$device)
      pi * (1.0 - eye)
    },
    forward = function(x_seq) {
      batch  <- x_seq$size(1L)
      miss   <- (x_seq$abs()$lt(1e-12))$float()
      x_flat <- x_seq$reshape(c(batch, -1L))
      m_flat <- miss$reshape(c(batch, -1L))
      x_imp  <- self$imputer(
        torch::torch_cat(list(x_flat, m_flat), dim = -1L)
      )$reshape(c(batch, lag, d))
      x_fill <- x_seq * (1.0 - miss) + x_imp * miss
      x_var  <- x_fill$permute(c(1L, 3L, 2L))$reshape(c(batch * d, lag, 1L))
      rnn_o  <- self$encoder(x_var)
      h      <- rnn_o[[2L]]$squeeze(1L)$reshape(c(batch, d, -1L))
      A      <- self$sample_graph()
      h2     <- self$msg(h, A)
      pred   <- self$out_head(h2)$squeeze(-1L)
      pi     <- A$clamp(min = 1e-6, max = 1.0 - 1e-6)
      p0     <- 0.1
      kl     <- (pi * torch::torch_log(pi / p0) +
                   (1.0 - pi) * torch::torch_log((1.0 - pi) / (1.0 - p0)))$mean()
      list(pred    = pred,
           A       = A$detach(),
           dag_loss = self$lam_dag * .deepnet_gnn_acyclicity(A),
           kl_loss  = self$lam_kl * kl)
    },
    causal_matrix = function() {
      torch::with_no_grad({ self$sample_graph()$detach()$cpu() })
    }
  )
}

# ---------------------------------------------------------------------------
# Shared training loop (AdamW + cosine annealing + early stopping)
# ---------------------------------------------------------------------------

.deepnet_gnn_train <- function(model, X_tr, Y_tr, X_val, Y_val,
                                model_name = "model",
                                epochs     = 60L,
                                lr         = 3e-4,
                                patience   = 12L,
                                batch_size = 64L,
                                device     = "cpu",
                                verbose    = FALSE) {
  x_tr_t  <- torch::torch_tensor(X_tr,  dtype = torch::torch_float32(), device = device)
  y_tr_t  <- torch::torch_tensor(Y_tr,  dtype = torch::torch_float32(), device = device)
  x_val_t <- torch::torch_tensor(X_val, dtype = torch::torch_float32(), device = device)
  y_val_t <- torch::torch_tensor(Y_val, dtype = torch::torch_float32(), device = device)

  N        <- x_tr_t$size(1L)
  n_batch  <- max(1L, ceiling(N / batch_size))
  opt      <- torch::optim_adamw(model$parameters, lr = lr, weight_decay = 1e-4)
  sched    <- torch::lr_cosine_annealing(opt, T_max = epochs)

  best_val   <- Inf
  best_state <- NULL
  no_improve <- 0L
  hist       <- list(train = numeric(epochs), val = numeric(epochs))

  for (ep in seq_len(epochs)) {
    model$train()
    idx_perm    <- torch::torch_randperm(N, device = device) + 1L
    tr_loss_sum <- 0.0

    for (b in seq_len(n_batch)) {
      s   <- (b - 1L) * batch_size + 1L
      e   <- min(b * batch_size, N)
      idx <- idx_perm[s:e]
      xb  <- x_tr_t[idx, , ]
      yb  <- y_tr_t[idx, ]
      opt$zero_grad()
      out  <- model$forward(xb)
      loss <- torch::nnf_mse_loss(out$pred, yb)
      if (!is.null(out$sparse_loss)) loss <- loss + out$sparse_loss
      if (!is.null(out$dag_loss))    loss <- loss + out$dag_loss
      if (!is.null(out$kl_loss))     loss <- loss + out$kl_loss
      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      opt$step()
      tr_loss_sum <- tr_loss_sum + loss$item()
    }
    sched$step()

    model$eval()
    val_loss <- torch::with_no_grad({
      torch::nnf_mse_loss(model$forward(x_val_t)$pred, y_val_t)$item()
    })

    hist$train[ep] <- tr_loss_sum / n_batch
    hist$val[ep]   <- val_loss

    if (val_loss < best_val) {
      best_val   <- val_loss
      best_state <- lapply(model$state_dict(), function(p) p$clone())
      no_improve <- 0L
    } else {
      no_improve <- no_improve + 1L
      if (no_improve >= patience) {
        if (verbose)
          message(sprintf("[%s] Early stopping at epoch %d", toupper(model_name), ep))
        break
      }
    }

    if (verbose && (ep %% 10L == 0L || ep == 1L))
      message(sprintf("[%-12s] Ep %3d | Train %.5f | Val %.5f",
                      toupper(model_name), ep, hist$train[ep], val_loss))
  }

  if (!is.null(best_state)) model$load_state_dict(best_state)
  hist
}

# =============================================================================
# Public API: gnn_causal_model
# =============================================================================

#' Graph Neural Network (GNN) Causal Models for Multivariate Time Series
#'
#' Fits one or more GNN-based causal models to multivariate time-series data,
#' translated from \code{05_graph_nn_causal_models_GNN.ipynb}:
#' \itemize{
#'   \item \code{"gvar"} — GVAR (Graph Vector Autoregression): learns
#'     lag-specific soft adjacency matrices
#'     \eqn{A^{(k)} \in [0,1]^{d \times d}} (sigmoid-gated logits, no
#'     self-loops) with L1 sparsity and NOTEARS DAG penalties. Two stacked
#'     GNN message-passing layers transform per-lag features and aggregate
#'     them via a GELU output head.
#'   \item \code{"causalgnn"} — CausalGNN / CD-GNN: encodes each variable's
#'     temporal history via a per-variable GRU, infers a soft adjacency via a
#'     bilinear-style graph learner, then propagates messages through stacked
#'     edge-conditioned GNN layers (single-step GRU node update + LayerNorm)
#'     with NOTEARS DAG and sparsity penalties.
#'   \item \code{"cuts"} — CUTS+ inspired: models the graph as a variational
#'     Bernoulli posterior \eqn{q(G) \approx \prod_{ij} \text{Bernoulli}(\pi_{ij})},
#'     imputes missing values from a network that sees observed values and a
#'     binary mask, then applies GRU encoding and edge-conv message passing;
#'     trained with MSE + KL(q ‖ sparse prior) + NOTEARS DAG penalty.
#' }
#'
#' @param data        Numeric matrix or data frame (rows = time, cols = variables).
#' @param lag         Integer lag window (default 10).
#' @param models      Character vector subset of
#'   \code{c("gvar","causalgnn","cuts")} (default all three).
#' @param hidden      Hidden / node embedding dimension (default 32).
#' @param n_gnn_layers Number of stacked GNN layers for CausalGNN (default 3).
#' @param lam_sparse  L1 sparsity penalty weight for GVAR and CausalGNN
#'   (default 0.005).
#' @param lam_dag     NOTEARS acyclicity penalty weight (default 0.1).
#' @param lam_kl      KL-to-sparse-prior weight for CUTS+ (default 1e-3).
#' @param epochs      Maximum training epochs (default 60).
#' @param lr          AdamW learning rate (default 3e-4).
#' @param patience    Early-stopping patience in epochs (default 12).
#' @param batch_size  Mini-batch size (default 64).
#' @param val_split   Validation fraction in (0, 1) (default 0.2).
#' @param device      Torch device string (\code{"cpu"} or \code{"cuda"}).
#'   \code{NULL} auto-selects.
#' @param verbose     Print per-epoch progress (default \code{FALSE}).
#' @param ...         Ignored.
#' @return Object of class \code{gnn_causal_model} containing:
#'   \describe{
#'     \item{\code{models}}{Named list of fitted torch modules.}
#'     \item{\code{histories}}{Named list of train/val loss vectors.}
#'     \item{\code{val_mse}}{Named numeric vector of final validation MSE.}
#'     \item{\code{causal_matrices}}{Named list of \eqn{d \times d} causal
#'       adjacency matrices.}
#'     \item{\code{X_val}}{Validation input array (for causal_matrix_gnn).}
#'     \item{\code{lag}}{Integer lag used.}
#'     \item{\code{var_names}}{Character vector of variable names.}
#'     \item{\code{device}}{Torch device string.}
#'   }
#' @references
#'   Brouwer, E. D., Simm, J., Arany, A., & Moreau, Y. (2022). CUTS: Neural
#'   causal discovery from irregular time-series data. \emph{ICLR 2023}.
#'   \url{https://arxiv.org/abs/2302.05925}
#'
#'   Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with
#'   NOTEARS: Continuous optimization for structure learning. \emph{NeurIPS}.
#'   \url{https://arxiv.org/abs/1803.01422}
#' @seealso \code{\link{rnn_causal_model}}, \code{\link{attn_causal_model}},
#'   \code{\link{neural_granger_ml}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' set.seed(42)
#' d <- 5L; T_len <- 300L
#' A_true <- matrix(0, d, d)
#' for (i in seq_len(d)) {
#'   A_true[i, i] <- 0.45
#'   if (i > 1) A_true[i, i - 1] <- 0.25
#' }
#' x <- matrix(0, T_len, d)
#' x[1, ] <- rnorm(d)
#' for (t in 2:T_len) x[t, ] <- x[t - 1, ] %*% t(A_true) + 0.15 * rnorm(d)
#' fit <- gnn_causal_model(x, lag = 8L, epochs = 20L, verbose = TRUE)
#' print(fit$val_mse)
#' print(causal_matrix_gnn(fit, model = "gvar"))
#' }
#' @export
gnn_causal_model <- function(
    data,
    lag          = 10L,
    models       = c("gvar", "causalgnn", "cuts"),
    hidden       = 32L,
    n_gnn_layers = 3L,
    lam_sparse   = 0.005,
    lam_dag      = 0.1,
    lam_kl       = 1e-3,
    epochs       = 60L,
    lr           = 3e-4,
    patience     = 12L,
    batch_size   = 64L,
    val_split    = 0.2,
    device       = NULL,
    verbose      = FALSE,
    ...) {

  if (!requireNamespace("torch", quietly = TRUE))
    stop("gnn_causal_model() requires package 'torch'.", call. = FALSE)

  x <- as.matrix(data)
  if (!is.numeric(x)) stop("`data` must be numeric.", call. = FALSE)

  valid_models <- c("gvar", "causalgnn", "cuts")
  req_models   <- unique(tolower(models))
  bad <- setdiff(req_models, valid_models)
  if (length(bad))
    stop("Unknown models: ", paste(bad, collapse = ", "), call. = FALSE)

  dev <- .deepnet_gnn_select_device(device)
  d   <- ncol(x)
  lag <- as.integer(lag)

  ds  <- .deepnet_gnn_build_dataset(x, lag = lag)
  n   <- dim(ds$X)[1L]
  sp  <- max(1L, min(n - 1L, floor((1 - val_split) * n)))

  X_tr  <- ds$X[seq_len(sp), , , drop = FALSE]
  Y_tr  <- ds$Y[seq_len(sp), , drop = FALSE]
  X_val <- ds$X[(sp + 1L):n, , , drop = FALSE]
  Y_val <- ds$Y[(sp + 1L):n, , drop = FALSE]

  fit_models      <- list()
  histories       <- list()
  val_mse_list    <- list()
  causal_matrices <- list()
  var_names <- if (!is.null(colnames(x))) colnames(x) else paste0("V", seq_len(d))

  for (m in req_models) {
    model_obj <- switch(m,
      gvar = .deepnet_gnn_gvar_module(
        d = d, lag = lag, hidden = as.integer(hidden),
        lam_sparse = lam_sparse, lam_dag = lam_dag
      )(),
      causalgnn = .deepnet_gnn_causal_gnn_module(
        d = d, lag = lag, hidden = as.integer(hidden),
        n_gnn_layers = as.integer(n_gnn_layers),
        lam_dag = lam_dag, lam_sparse = lam_sparse
      )(),
      cuts = .deepnet_gnn_cuts_module(
        d = d, lag = lag, hidden = as.integer(hidden),
        lam_kl = lam_kl, lam_dag = lam_dag
      )()
    )
    model_obj <- model_obj$to(device = dev)

    hist <- .deepnet_gnn_train(
      model      = model_obj,
      X_tr = X_tr, Y_tr = Y_tr,
      X_val = X_val, Y_val = Y_val,
      model_name = m,
      epochs     = as.integer(epochs),
      lr         = lr,
      patience   = as.integer(patience),
      batch_size = as.integer(batch_size),
      device     = dev,
      verbose    = verbose
    )

    model_obj$eval()
    x_val_t <- torch::torch_tensor(X_val, dtype = torch::torch_float32(), device = dev)
    y_val_t <- torch::torch_tensor(Y_val, dtype = torch::torch_float32(), device = dev)
    vmse <- torch::with_no_grad({
      torch::nnf_mse_loss(model_obj$forward(x_val_t)$pred, y_val_t)$item()
    })

    n_ref   <- min(128L, nrow(X_val))
    x_ref_t <- torch::torch_tensor(
      X_val[seq_len(n_ref), , , drop = FALSE],
      dtype = torch::torch_float32(), device = dev
    )
    cmat <- tryCatch({
      raw <- switch(m,
        gvar      = as.matrix(model_obj$causal_matrix()),
        causalgnn = as.matrix(model_obj$causal_matrix(x_ref_t)),
        cuts      = as.matrix(model_obj$causal_matrix())
      )
      rownames(raw) <- colnames(raw) <- var_names
      raw
    }, error = function(e) NULL)

    fit_models[[m]]      <- model_obj
    histories[[m]]       <- hist
    val_mse_list[[m]]    <- vmse
    causal_matrices[[m]] <- cmat
  }

  structure(
    list(
      models          = fit_models,
      histories       = histories,
      val_mse         = unlist(val_mse_list),
      causal_matrices = causal_matrices,
      X_val           = X_val,
      lag             = lag,
      var_names       = var_names,
      device          = dev
    ),
    class = "gnn_causal_model"
  )
}

#' Predict from a fitted GNN causal model
#'
#' @param object  Fitted \code{gnn_causal_model} object.
#' @param model   One of \code{"gvar"}, \code{"causalgnn"}, \code{"cuts"}.
#' @param newdata Numeric array of shape \code{(batch, lag, d)} or matrix
#'   \code{(lag, d)} for a single sequence (batch dim added automatically).
#' @param ...     Ignored.
#' @return Numeric matrix of shape \code{(batch, d)}: one-step-ahead predictions.
#' @seealso \code{\link{gnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # predict(fit, model = "gvar", newdata = X_new)
#' }
#' @export
predict.gnn_causal_model <- function(object,
                                      model = c("gvar", "causalgnn", "cuts"),
                                      newdata, ...) {
  model <- match.arg(model)
  if (!model %in% names(object$models))
    stop("Model '", model, "' was not fitted.", call. = FALSE)
  fit <- object$models[[model]]
  fit$eval()
  x_new <- as.array(newdata)
  if (length(dim(x_new)) == 2L)
    x_new <- array(x_new, dim = c(1L, dim(x_new)))
  x_t <- torch::torch_tensor(x_new, dtype = torch::torch_float32(),
                              device = object$device)
  torch::with_no_grad({
    as.matrix(fit$forward(x_t)$pred$cpu())
  })
}

#' Extract learned causal adjacency matrix from a GNN causal model
#'
#' @param object Fitted \code{gnn_causal_model} object.
#' @param model  One of \code{"gvar"}, \code{"causalgnn"}, \code{"cuts"}.
#' @return Named \eqn{d \times d} matrix of causal edge weights.
#' @seealso \code{\link{gnn_causal_model}}, \code{\link{plot_scm_dag}},
#'   \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # causal_matrix_gnn(fit, model = "causalgnn")
#' }
#' @export
causal_matrix_gnn <- function(object,
                               model = c("gvar", "causalgnn", "cuts")) {
  model <- match.arg(model)
  if (!model %in% names(object$causal_matrices))
    stop("Model '", model, "' was not fitted.", call. = FALSE)
  object$causal_matrices[[model]]
}

#' @rdname gnn_causal_model
#' @export
gnnCausalModel <- function(data, ...) gnn_causal_model(data, ...)

#' @rdname gnn_causal_model
#' @export
GNNCausalModel <- function(data, ...) gnn_causal_model(data, ...)

#' Convenience wrapper: fit only the GVAR model
#'
#' Calls \code{\link{gnn_causal_model}} with \code{models = "gvar"}.
#' @inheritParams gnn_causal_model
#' @return Object of class \code{gnn_causal_model}.
#' @seealso \code{\link{gnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # gvar_model(x, lag = 10L)
#' }
#' @export
gvar_model <- function(data, ...) gnn_causal_model(data, models = "gvar", ...)

#' Convenience wrapper: fit only the CausalGNN model
#'
#' Calls \code{\link{gnn_causal_model}} with \code{models = "causalgnn"}.
#' @inheritParams gnn_causal_model
#' @return Object of class \code{gnn_causal_model}.
#' @seealso \code{\link{gnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # causal_gnn_model(x, lag = 10L)
#' }
#' @export
causal_gnn_model <- function(data, ...) gnn_causal_model(data, models = "causalgnn", ...)

#' Convenience wrapper: fit only the CUTS+ model
#'
#' Calls \code{\link{gnn_causal_model}} with \code{models = "cuts"}.
#' @inheritParams gnn_causal_model
#' @return Object of class \code{gnn_causal_model}.
#' @seealso \code{\link{gnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # cuts_model(x, lag = 10L)
#' }
#' @export
cuts_model <- function(data, ...) gnn_causal_model(data, models = "cuts", ...)

#' CamelCase alias: GVARModel
#' @rdname gvar_model
#' @export
GVARModel <- function(data, ...) gvar_model(data, ...)

#' CamelCase alias: CausalGNNModel
#' @rdname causal_gnn_model
#' @export
CausalGNNModel <- function(data, ...) causal_gnn_model(data, ...)

#' CamelCase alias: CUTSModel
#' @rdname cuts_model
#' @export
CUTSModel <- function(data, ...) cuts_model(data, ...)
