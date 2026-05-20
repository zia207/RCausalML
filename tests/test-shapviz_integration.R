# Test: R/shapviz_integration.R (explain_cate, rcausalml_predict_numeric, shapviz flow)
# Run from package root: Rscript tests/test-shapviz_integration.R

pkg_root <- if (file.exists("R/shapviz_integration.R")) "." else
  if (file.exists("../R/shapviz_integration.R")) ".." else stop("Run from package root")

# Load package (prefer load_all so we have rcausalml_predict_numeric in scope for direct tests)
if (requireNamespace("devtools", quietly = TRUE)) {
  devtools::load_all(pkg_root, quiet = TRUE)
} else {
  # Fallback: install and load from .R_libs or default
  if (dir.exists(file.path(pkg_root, ".R_libs"))) {
    .libPaths(c(file.path(pkg_root, ".R_libs"), .libPaths()))
  }
  if (!requireNamespace("RCausalML", quietly = TRUE)) {
    stop("Package RCausalML not found. Install it or run with devtools::load_all().")
  }
  library(RCausalML)
}

set.seed(42)

message("========== test-shapviz_integration.R ==========")

# ---- Synthetic data ----
n <- 200
p <- 4
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("X", seq_len(p))
W <- rbinom(n, 1, 0.5)
Y <- 1 + 0.3 * X[, 1] + 0.2 * X[, 2] + 0.5 * W + rnorm(n, 0, 0.5)

# ---- 1. rcausalml_predict_numeric (internal) ----
# SLearner returns vector from predict(); causal_forest returns list with "predictions"
message("---- 1. rcausalml_predict_numeric ----")
sl <- SLearner(learner = "lm")
sl <- fit(sl, X, W, Y)
pred_sl <- predict(sl, X[1:5, ])
# SLearner predict can return list with components or vector; ensure we get numeric
vec <- RCausalML:::rcausalml_predict_numeric(sl, X[1:5, ])
stopifnot(is.numeric(vec), length(vec) == 5)
message("  OK: rcausalml_predict_numeric(SLearner) returns numeric vector")

# Causal forest returns list with "predictions"
cf <- causal_forest(X, Y, W, num.trees = 20, seed = 1)
vec_cf <- RCausalML:::rcausalml_predict_numeric(cf, X[1:3, ])
stopifnot(is.numeric(vec_cf), length(vec_cf) == 3)
message("  OK: rcausalml_predict_numeric(causal_forest) returns numeric vector")

# ---- 2. explain_cate without kernelshap ----
message("---- 2. explain_cate() without kernelshap ----")
has_kernelshap <- requireNamespace("kernelshap", quietly = TRUE)
if (!has_kernelshap) {
  err <- tryCatch(explain_cate(sl, X[1:5, ]), error = identity)
  stopifnot(inherits(err, "error"), grepl("kernelshap", conditionMessage(err)))
  message("  OK: explain_cate() gives informative error when kernelshap not installed")
} else {
  message("  SKIP: kernelshap is installed (test 3 will run)")
}

# ---- 3. explain_cate + shapviz when kernelshap (and optionally shapviz) available ----
if (has_kernelshap) {
  message("---- 3. explain_cate() + shapviz() ----")
  X_explain <- X[1:10, ]
  bg_X <- X[1:30, ]
  ks <- explain_cate(sl, X_explain, bg_X = bg_X, use_permshap = TRUE, verbose = FALSE)
  stopifnot(inherits(ks, "kernelshap"))
  stopifnot("S" %in% names(ks), "X" %in% names(ks))
  stopifnot(nrow(ks$S) == nrow(X_explain), ncol(ks$S) == p)
  message("  OK: explain_cate() returns kernelshap object with S and X")

  has_shapviz <- requireNamespace("shapviz", quietly = TRUE)
  if (has_shapviz) {
    shp <- shapviz::shapviz(ks)
    stopifnot(inherits(shp, "shapviz"))
    stopifnot("S" %in% names(shp), "X" %in% names(shp))
    # One plot call (no error)
    capture.output(shapviz::sv_importance(shp))
    message("  OK: shapviz(explain_cate(...)) and sv_importance() run without error")
  } else {
    message("  OK: kernelshap done; install 'shapviz' for full visualization")
  }
}

message("")
message("========== All shapviz_integration tests passed. ==========")
