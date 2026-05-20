# Test: R/causal_forest_confounder.R (causal forest with confounders)
# Run from package root: Rscript tests/test-causal_forest_confounder.R
# Or: source("tests/test-causal_forest_confounder.R")
# Requires: install.packages("grf")

pkg_root <- if (file.exists("R/causal_forest_confounder.R")) "." else
  if (file.exists("../R/causal_forest_confounder.R")) ".." else stop("Run from RCausalML package root")

# Skip all tests if grf is not installed (causal_forest_confounder uses grf internals)
if (!requireNamespace("grf", quietly = TRUE)) {
  message("========== test-causal_forest_confounder.R ==========")
  message("SKIP: Package 'grf' not installed. Install with: install.packages(\"grf\")")
  message("=====================================================")
  quit(save = "no", status = 0)
}

# Load grf; causal_forest_confounder uses grf internals (not all exported), so attach them
library(grf)
grf_internals <- c(
  "validate_X", "validate_sample_weights", "validate_observations", "validate_clusters",
  "validate_equalize_cluster_weights", "validate_num_threads",
  "regression_forest", "create_train_matrices", "causal_train", "do.call.rcpp",
  "get_legacy_seed", "get_verbose", "tune_forest"
)
for (fn in grf_internals) {
  tryCatch(
    assign(fn, getFromNamespace(fn, "grf"), envir = .GlobalEnv),
    error = function(e) NULL
  )
}
source(file.path(pkg_root, "R/causal_forest_confounder.R"))

set.seed(42)

# ---- Synthetic data (from roxygen examples) ----
n <- 500
p <- 10
X <- matrix(rnorm(n * p), n, p)
W <- rbinom(n, 1, 0.5)
Y <- pmax(X[, 1], 0) * W + X[, 2] + pmin(X[, 3], 0) + rnorm(n)

message("========== test-causal_forest_confounder.R ==========")
message("n = ", n, ", p = ", p)
message("")

# ---- 1. Without confounders (same as causal_forest) ----
message("---- 1. causal_forest_confounder(X, Y, W) [no confounders] ----")
c.forest <- causal_forest_confounder(X, Y, W, num.trees = 200, seed = 123)
stopifnot(inherits(c.forest, "causal_forest"))
stopifnot(inherits(c.forest, "grf"))
stopifnot(nrow(c.forest[["X.orig"]]) == n)
stopifnot(length(c.forest[["Y.hat"]]) == n, length(c.forest[["W.hat"]]) == n)
stopifnot(is.null(c.forest[["confounders"]]))
message("  OK: fit without confounders, structure correct")

# ---- 2. Predict (OOB and newdata) ----
message("---- 2. predict(c.forest) ----")
pred_oob <- predict(c.forest)
stopifnot(is.data.frame(pred_oob) || is.list(pred_oob))
stopifnot("predictions" %in% names(pred_oob))
stopifnot(length(pred_oob$predictions) == n)
n_test <- 50
X_test <- matrix(rnorm(n_test * p), n_test, p)
pred_test <- predict(c.forest, X_test)
stopifnot(length(pred_test$predictions) == n_test)
message("  OK: OOB and newdata predictions")

# ---- 3. With confounders ----
message("---- 3. causal_forest_confounder(X, Y, W, confounders = Z) ----")
Z <- matrix(rnorm(n * 3), n, 3)
c.forest.z <- causal_forest_confounder(X, Y, W, confounders = Z, num.trees = 200, seed = 456)
stopifnot(inherits(c.forest.z, "causal_forest"))
stopifnot(!is.null(c.forest.z[["confounders"]]))
stopifnot(nrow(c.forest.z[["confounders"]]) == n, ncol(c.forest.z[["confounders"]]) == 3)
pred.z <- predict(c.forest.z, X[1:20, ])
stopifnot(length(pred.z$predictions) == 20)
message("  OK: fit with confounders, predict works")

# ---- 4. average_treatment_effect (grf) ----
message("---- 4. average_treatment_effect() ----")
ate <- grf::average_treatment_effect(c.forest)
stopifnot(is.vector(ate) || is.list(ate))
stopifnot(all(c("estimate", "std.err") %in% names(ate)))
message("  ATE estimate: ", round(ate[["estimate"]], 4), ", std.err: ", round(ate[["std.err"]], 4))
message("  OK: average_treatment_effect()")

# ---- 5. Confounder validation: wrong nrow ----
message("---- 5. Confounder validation (wrong nrow) ----")
Z_bad <- matrix(rnorm(100 * 3), 100, 3)
err <- tryCatch(
  causal_forest_confounder(X, Y, W, confounders = Z_bad, num.trees = 50),
  error = identity
)
stopifnot(inherits(err, "error"))
stopifnot(grepl("same number of rows", conditionMessage(err), fixed = TRUE))
message("  OK: confounders nrow validated")

# ---- 6. Confounder validation: non-numeric ----
message("---- 6. Confounder validation (non-numeric) ----")
Z_char <- data.frame(a = letters[1:5], b = 1:5, c = rnorm(5))
err2 <- tryCatch(
  causal_forest_confounder(X[1:5, ], Y[1:5], W[1:5], confounders = Z_char, num.trees = 50),
  error = identity
)
stopifnot(inherits(err2, "error"))
message("  OK: confounders must be numeric")

message("")
message("========== All causal_forest_confounder tests passed. ==========")
