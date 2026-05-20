# Test: R/multi_arm_causal_boost.r — MultiArmCausalBoost, multi_arm_PEHE, multi_arm_ATE
# Run from package root: Rscript tests/test-multi-arm-causal-boost.R

pkg_root <- if (file.exists("R/multi_arm_causal_boost.R")) "." else
  if (file.exists("../R/multi_arm_causal_boost.R")) ".." else stop("Run from RCausalML package root")

library(R6)
library(ranger)
source(file.path(pkg_root, "R/multi_arm_causal_boost.R"))

has_xgboost <- requireNamespace("xgboost", quietly = TRUE)

set.seed(42)

message("========== test-multi-arm-causal-boost.R ==========")

# ---- 1. multi_arm_PEHE / multi_arm_ATE ----
message("\n---- 1. multi_arm_PEHE / multi_arm_ATE ----")

n <- 5
tau_true <- array(rnorm(n * 2 * 1), dim = c(n, 2, 1))
tau_hat  <- tau_true + rnorm(n * 2 * 1, sd = 0.1)

pehe_val <- multi_arm_PEHE(tau_true, tau_hat, rmse = TRUE)
stopifnot(is.numeric(pehe_val), length(pehe_val) == 1L, pehe_val >= 0)
stopifnot(multi_arm_PEHE(tau_true, tau_true, rmse = TRUE) == 0)
message("  OK: multi_arm_PEHE returns scalar >= 0 and is 0 for perfect preds")

ate_err <- multi_arm_ATE(tau_true, tau_hat)
stopifnot(is.numeric(ate_err), identical(dim(ate_err), c(2L, 1L)))
stopifnot(all(ate_err >= 0))
message("  OK: multi_arm_ATE returns [contrasts, outcomes] absolute error")

ok <- tryCatch(multi_arm_PEHE(tau_true, array(1, dim = c(n, 1, 1))), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
ok <- tryCatch(multi_arm_ATE(tau_true, array(1, dim = c(n, 1, 1))), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: helpers reject dimension mismatch")

# ---- 2. MultiArmCausalBoost input validation ----
message("\n---- 2. MultiArmCausalBoost input validation ----")

model <- MultiArmCausalBoost$new(parameters = list(eta = 0.1), baseline = "A")
stopifnot(inherits(model, "R6"), inherits(model, "MultiArmCausalBoost"))
message("  OK: MultiArmCausalBoost$new()")

X <- matrix(rnorm(30), nrow = 10, ncol = 3)
Y <- rnorm(10)

ok <- tryCatch(model$fit(X, Y, W = c("A", "B")), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: fit() rejects non-factor W")

W1 <- factor(rep("A", 10))
ok <- tryCatch(model$fit(X, Y, W1), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: fit() rejects <2 treatment arms")

W_na <- factor(c(rep("A", 5), rep(NA, 5)), levels = c("A", "B"))
ok <- tryCatch(model$fit(X, Y, W_na), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: fit() rejects NA in W")

W_small <- factor(c(rep("A", 9), "B"), levels = c("A", "B"))
ok <- tryCatch(model$fit(X, Y, W_small), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: fit() rejects arms with <2 observations")

model2 <- MultiArmCausalBoost$new()
ok <- tryCatch(model2$predict(X), error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: predict() errors when model not fitted")

if (!has_xgboost) {
  message("\n(xgboost not installed — skipping fit/predict tests)")
  message("========== test-multi-arm-causal-boost.R done (validation only) ==========")
  quit(save = "no", status = 0)
}

# ---- 3. Fit and predict (single outcome) ----
message("\n---- 3. Fit and predict (single outcome) ----")

n <- 120
p <- 5
X <- matrix(rnorm(n * p), nrow = n, ncol = p, dimnames = list(NULL, paste0("X", 1:p)))

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
Y <- ifelse(W == "A", mu_A, ifelse(W == "B", mu_B, mu_C)) + rnorm(n, sd = 0.3)

# include a little missingness to exercise imputation
X_na <- X
X_na[sample.int(n * p, size = 10)] <- NA_real_

model <- MultiArmCausalBoost$new(parameters = list(eta = 0.1, max_depth = 3L), baseline = "A")
capture.output(model$fit(X_na, Y, W, nrounds = 15L, verbose = 0L))
stopifnot(!is.null(model$outcome_models), !is.null(model$propensity_model))
message("  OK: fit() produces outcome_models and propensity_model")

pred <- model$predict(X_na, drop = FALSE)
ni <- as.integer(n)
stopifnot(
  is.list(pred),
  identical(names(pred), c("mu_hat", "propensity_score", "tau_hat", "baseline")),
  identical(pred$baseline, "A"),
  is.array(pred$mu_hat), identical(dim(pred$mu_hat), c(ni, 3L, 1L)),
  is.matrix(pred$propensity_score), identical(dim(pred$propensity_score), c(ni, 3L)),
  is.array(pred$tau_hat), identical(dim(pred$tau_hat), c(ni, 2L, 1L)),
  all(pred$propensity_score >= 0 & pred$propensity_score <= 1),
  !anyNA(pred$mu_hat),
  !anyNA(pred$tau_hat)
)
message("  OK: predict() returns correct shapes and bounded propensities")

# Baseline switching changes contrast labels/dimension names
pred_B <- model$predict(X, baseline = "B", drop = FALSE)
stopifnot(identical(pred_B$baseline, "B"))
stopifnot(identical(dim(pred_B$tau_hat), c(ni, 2L, 1L)))
message("  OK: predict() supports alternate baseline")

# ---- 4. Fit and predict (multi-outcome) ----
message("\n---- 4. Fit and predict (multi-outcome) ----")

Y2 <- cbind(
  Y = Y,
  Y2 = 0.2 + 0.5 * (W == "C") + 0.2 * X[, 2] + rnorm(n, sd = 0.4)
)
model_m <- MultiArmCausalBoost$new(parameters = list(eta = 0.1, max_depth = 3L), baseline = "A")
capture.output(model_m$fit(X, Y2, W, nrounds = 10L, verbose = 0L))
pred_m <- model_m$predict(X, drop = FALSE)
stopifnot(
  identical(dim(pred_m$mu_hat), c(ni, 3L, 2L)),
  identical(dim(pred_m$tau_hat), c(ni, 2L, 2L))
)
message("  OK: multi-outcome fit/predict shapes")

message("\n========== test-multi-arm-causal-boost.R done (all passed) ==========")

