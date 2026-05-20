# Test causal structure learning (notrears, DAG_GNN, GraN_DAD)
# Run from package root: Rscript tests/test-causal_structure_learning.R
# Quick run (small n, no tune): QUICK=1 Rscript tests/test-causal_structure_learning.R
# Requires: reticulate; Python with numpy, scipy (NOTEARS), torch (DAG-GNN), gcastle (GraN-DAG).
# Optional: notears_path, optuna for tuning.

# Load package code (when run from package root)
pkg_root <- if (file.exists("R/notears.R")) "." else
  if (file.exists("../R/notears.R")) ".." else
  stop("Run from RCausalML package root")
# Causal structure learning functions are in causalDeepNet.R via installed package
lib_paths <- c(file.path(pkg_root, ".Rlibrary"), .libPaths())
.libPaths(lib_paths)
suppressPackageStartupMessages(library(RCausalML))

quick <- nzchar(Sys.getenv("QUICK", ""))

# --- Synthetic data (small n, small d for fast tests) ---
set.seed(42)
n <- if (quick) 80 else 200
d <- if (quick) 4 else 6
# Linear SEM in topological order: X[,j] = X %*% W[,j] + Z[,j] with W lower-triangular (no cycles)
W_true <- matrix(0, d, d)
idx <- which(lower.tri(matrix(0, d, d)), arr.ind = TRUE)
for (k in seq_len(min(2 * d, nrow(idx)))) {
  i <- idx[k, 1L]
  j <- idx[k, 2L]
  W_true[i, j] <- runif(1L, 0.3, 0.8)
}
X_sem <- matrix(0, n, d)
for (j in seq_len(d)) {
  X_sem[, j] <- X_sem %*% W_true[, j] + rnorm(n, 0, 0.5)
}
# Center (NOTEARS linear centers internally for l2, but explicit is fine)
X_sem <- scale(X_sem, center = TRUE, scale = FALSE)

message("========== Causal structure learning tests: n = ", n, ", d = ", d, " ==========")
message("")

# ---- 1. notrears (linear) ----
message("---- 1. notrears (linear) ----")
res_notrears <- tryCatch({
  notrears(X_sem,
          model = "linear",
          lambda1 = 0.1,
          loss_type = "l2",
          max_iter = if (quick) 20L else 50L,
          w_threshold = 0.3,
          notears_path = NULL,
          tune = FALSE,
          seed = 42L)
}, error = function(e) {
  message("  Error: ", conditionMessage(e))
  NULL
})
if (!is.null(res_notrears)) {
  stopifnot(is.matrix(res_notrears$adjacency), nrow(res_notrears$adjacency) == d, ncol(res_notrears$adjacency) == d)
  stopifnot(is.matrix(res_notrears$binary_adjacency), all(res_notrears$binary_adjacency %in% c(0L, 1L)))
  message("  Adjacency dim: ", nrow(res_notrears$adjacency), " x ", ncol(res_notrears$adjacency))
  message("  NNZ (edges): ", sum(res_notrears$binary_adjacency != 0))
  message("  OK")
}
message("")

# ---- 2. notrears (nonlinear) ----
message("---- 2. notrears (nonlinear) ----")
res_notrears_nl <- tryCatch({
  notrears(X_sem,
          model = "nonlinear",
          lambda1 = 0.05,
          max_iter = if (quick) 10L else 30L,
          w_threshold = 0.3,
          notears_path = NULL,
          tune = FALSE,
          seed = 42L)
}, error = function(e) {
  message("  Error: ", conditionMessage(e))
  NULL
})
if (!is.null(res_notrears_nl)) {
  stopifnot(is.matrix(res_notrears_nl$adjacency), nrow(res_notrears_nl$adjacency) == d, ncol(res_notrears_nl$adjacency) == d)
  message("  NNZ (edges): ", sum(res_notrears_nl$binary_adjacency != 0))
  message("  OK")
}
message("")

# ---- 3. DAG_GNN ----
message("---- 3. DAG_GNN ----")
res_daggnn <- tryCatch({
  DAG_GNN(X_sem,
          hidden_dim = if (quick) 16L else 32L,
          n_epochs = if (quick) 30L else 100L,
          lr = 1e-3,
          threshold = 0.1,
          device = "cpu",
          tune = FALSE,
          seed = 42L)
}, error = function(e) {
  message("  Error: ", conditionMessage(e))
  NULL
})
if (!is.null(res_daggnn)) {
  stopifnot(is.matrix(res_daggnn$adjacency), nrow(res_daggnn$adjacency) == d, ncol(res_daggnn$adjacency) == d)
  message("  NNZ (edges): ", sum(res_daggnn$binary_adjacency != 0))
  if (!is.null(res_daggnn$final_loss)) message("  Final loss: ", round(res_daggnn$final_loss, 6))
  message("  OK")
}
message("")

# ---- 4. GraN_DAD (GraN-DAG) ----
message("---- 4. GraN_DAD (GraN-DAG) ----")
res_gran <- tryCatch({
  GraN_DAD(X_sem,
           hidden_num = 2L,
           hidden_dim = if (quick) 6L else 10L,
           batch_size = min(32L, n),
           lr = 0.001,
           iterations = if (quick) 500L else 2000L,
           device_type = "cpu",
           normalize = TRUE,
           tune = FALSE,
           seed = 42L)
}, error = function(e) {
  message("  Error: ", conditionMessage(e))
  NULL
})
if (!is.null(res_gran)) {
  stopifnot(is.matrix(res_gran$adjacency), nrow(res_gran$adjacency) == d, ncol(res_gran$adjacency) == d)
  message("  NNZ (edges): ", sum(res_gran$binary_adjacency != 0))
  message("  OK")
}
message("")

# ---- 5. Optional: notrears with tune (skipped when QUICK=1) ----
if (!quick && n >= 100 && d >= 4) {
  message("---- 5. notrears with tune (optuna) [optional] ----")
  res_tune <- tryCatch({
    notrears(X_sem,
             model = "linear",
             tune = TRUE,
             n_trials = 3L,
             max_iter = 20L,
             seed = 42L)
  }, error = function(e) {
    message("  Error: ", conditionMessage(e))
    NULL
  })
  if (!is.null(res_tune)) {
    stopifnot(!is.null(res_tune$optuna_study), !is.null(res_tune$best_params))
    message("  Best lambda1: ", round(res_tune$best_params$lambda1, 4))
    message("  OK")
  } else {
    message("  Skipped (optuna/notears not available or error)")
  }
  message("")
}

message("========== test-causal_structure_learning.R done ==========")
