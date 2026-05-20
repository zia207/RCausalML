# Test: R/causalXGBoost.R — PEHE, ATE, CXGBoost
# Run from package root: Rscript tests/test-cxgboost.R

pkg_root <- if (file.exists("R/causalXGBoost.R")) "." else
  if (file.exists("../R/causalXGBoost.R")) ".." else stop("Run from RCausalML package root")

library(R6)
library(ranger)
source(file.path(pkg_root, "R/causalXGBoost.R"))

has_xgboost <- requireNamespace("xgboost", quietly = TRUE)

set.seed(42)

message("========== test-cxgboost.R ==========")

# ---- 1. PEHE ----
message("\n---- 1. PEHE ----")
y <- matrix(c(1, 2, 0.5, 1.5, 0, 1), ncol = 2, byrow = TRUE)
y_hat <- matrix(c(1.1, 2.2, 0.4, 1.6, 0.1, 0.9), ncol = 2, byrow = TRUE)
pehe_val <- PEHE(y, y_hat)
stopifnot(is.numeric(pehe_val), length(pehe_val) == 1L, pehe_val >= 0)
message("  OK: PEHE returns scalar >= 0")

# PEHE with perfect predictions is 0
y_hat_same <- y
stopifnot(PEHE(y, y_hat_same) == 0)
message("  OK: PEHE = 0 when predictions equal truth")

# PEHE errors on wrong dimensions
ok <- tryCatch(PEHE(matrix(1:4, ncol = 1), y_hat), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: PEHE rejects non-2-column y")

# ---- 2. ATE ----
message("\n---- 2. ATE ----")
ate_val <- ATE(y, y_hat)
stopifnot(is.numeric(ate_val), length(ate_val) == 1L, ate_val >= 0)
message("  OK: ATE returns scalar >= 0")

# ATE with perfect predictions is 0
stopifnot(ATE(y, y_hat_same) == 0)
message("  OK: ATE = 0 when predictions equal truth")

ok <- tryCatch(ATE(y, matrix(1:4, ncol = 1)), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: ATE rejects non-2-column y_hat")

# ---- 3. CXGBoost class ----
message("\n---- 3. CXGBoost ----")

# Initialize
model <- CXGBoost$new(parameters = list(eta = 0.1))
stopifnot(inherits(model, "R6"), inherits(model, "CXGBoost"))
stopifnot(is.list(model$parameters), model$parameters$eta == 0.1)
message("  OK: CXGBoost$new() and parameters")

# Input validation: fit with non-binary t
n <- 100
X <- matrix(rnorm(n * 3), nrow = n)
t_bad <- rep(2L, n)
y <- rnorm(n)
ok <- tryCatch(model$fit(X, t_bad, y), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: fit() rejects non-0/1 treatment")

# Input validation: length mismatch
ok <- tryCatch(model$fit(X, rep(0L, 50), y), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: fit() rejects length mismatch")

# Predict before fit
model2 <- CXGBoost$new()
ok <- tryCatch(model2$predict(X), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: predict() errors when model not fitted")

if (!has_xgboost) {
  message("\n(xgboost not installed — skipping fit/predict tests)")
  message("========== test-cxgboost.R done (validation only) ==========")
  quit(save = "no", status = 0)
}

# ---- 4. Fit and predict ----
message("\n---- 4. Fit and predict ----")
n <- 200
X <- matrix(rnorm(n * 5), nrow = n)
t <- rbinom(n, 1, 0.4 + 0.2 * (X[, 1] > 0))
tau <- 0.5 + 0.3 * X[, 1]
y <- 1 + 0.2 * X[, 1] + 0.1 * X[, 2] + tau * t + rnorm(n, 0, 0.5)

model <- CXGBoost$new(parameters = list(eta = 0.05, max_depth = 4))
capture.output(model$fit(X, t, y, nrounds = 30L, verbose = 0L))
stopifnot(!is.null(model$booster), !is.null(model$propensity_model))
message("  OK: fit() produces booster and propensity_model")

pred <- model$predict(X)
stopifnot(
  is.list(pred),
  identical(names(pred), c("y0_hat", "y1_hat", "propensity_score", "tau_hat")),
  length(pred$y0_hat) == n,
  length(pred$y1_hat) == n,
  length(pred$propensity_score) == n,
  length(pred$tau_hat) == n,
  all(pred$propensity_score >= 0 & pred$propensity_score <= 1)
)
message("  OK: predict() returns y0_hat, y1_hat, propensity_score, tau_hat")

# Predict on new data (same ncol)
X_new <- matrix(rnorm(50 * 5), nrow = 50)
pred_new <- model$predict(X_new)
stopifnot(length(pred_new$y0_hat) == 50L)
message("  OK: predict() on new X with same ncol")

# Optional: PEHE/ATE on synthetic (no ground-truth potential outcomes here, just sanity)
y_fake <- cbind(1 + 0.2 * X[, 1], 1.5 + 0.5 * X[, 1])
y_hat_fake <- cbind(pred$y0_hat, pred$y1_hat)
pehe_fake <- PEHE(y_fake, y_hat_fake)
ate_fake <- ATE(y_fake, y_hat_fake)
stopifnot(is.numeric(pehe_fake), is.numeric(ate_fake))
message("  OK: PEHE/ATE applicable to model predictions")

message("\n========== test-cxgboost.R done (all passed) ==========")
