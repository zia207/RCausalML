# Test: R/causalForest.R (causal forest, predict, ATE, variable importance)
# Run from package root: Rscript tests/test-causalForest.R
# Or: source("tests/test-causalForest.R")

pkg_root <- if (file.exists("R/causalForest.R")) "." else
  if (file.exists("../R/causalForest.R")) ".." else stop("Run from Causal_ML package root")

source(file.path(pkg_root, "R/causalForest.R"))

set.seed(42)

# ---- Synthetic data: known CATE ----
n <- 400
p <- 6
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("X", seq_len(p))
W <- rbinom(n, 1, 0.5)
tau_true <- 0.5 + 0.3 * X[, 1] + 0.2 * pmax(X[, 2], 0)
Y <- 1 + 0.5 * X[, 1] + 0.3 * X[, 2] + tau_true * W + rnorm(n, 0, 0.5)
true_ate <- mean(tau_true)

message("========== test-causalForest.R ==========")
message("n = ", n, ", p = ", p)
message("True ATE: ", round(true_ate, 4))
message("")

# ---- 1. Fit causal forest (small forest for speed) ----
message("---- 1. causal_forest() ----")
cf <- causal_forest(X, Y, W, num.trees = 100, seed = 123)
stopifnot(inherits(cf, "causal_forest"))
stopifnot(length(cf$trees) == 100)
stopifnot(nrow(cf$X) == n, ncol(cf$X) == p)
stopifnot(length(cf$Y.hat) == n, length(cf$W.hat) == n)
stopifnot(length(cf$oob_predictions) == n)
message("  OK: causal_forest() returns object with expected structure")

# ---- 2. OOB predictions (newdata = NULL) ----
message("---- 2. predict(forest) OOB ----")
pred_oob <- predict(cf)
stopifnot(is.list(pred_oob))
stopifnot("predictions" %in% names(pred_oob))
stopifnot(length(pred_oob$predictions) == n)
valid_oob <- sum(!is.na(pred_oob$predictions))
message("  OOB predictions: ", valid_oob, " / ", n, " non-NA")
stopifnot(valid_oob > 0)
message("  OK: OOB predictions returned")

# ---- 3. Predict on new data ----
message("---- 3. predict(forest, newdata) ----")
n_test <- 50
X_test <- matrix(rnorm(n_test * p), n_test, p)
colnames(X_test) <- colnames(X)
pred_test <- predict(cf, X_test)
stopifnot(is.list(pred_test))
stopifnot(length(pred_test$predictions) == n_test)
message("  Mean CATE (test): ", round(mean(pred_test$predictions, na.rm = TRUE), 4))
message("  OK: newdata predictions returned")

# ---- 4. Predict with variance estimates ----
message("---- 4. predict(..., estimate.variance = TRUE) ----")
pred_var <- predict(cf, X_test, estimate.variance = TRUE)
stopifnot("variance.estimates" %in% names(pred_var))
stopifnot(length(pred_var$variance.estimates) == n_test)
message("  OK: variance estimates returned")

# ---- 5. average_treatment_effect() ----
message("---- 5. average_treatment_effect() ----")
ate <- average_treatment_effect(cf)
stopifnot(is.list(ate))
stopifnot(all(c("estimate", "std.err", "conf.int") %in% names(ate)))
message("  ATE estimate: ", round(ate$estimate, 4), " (true: ", round(true_ate, 4), ")")
message("  Std err: ", round(ate$std.err, 4))
message("  95% CI: [", round(ate$conf.int[1], 4), ", ", round(ate$conf.int[2], 4), "]")
message("  OK: average_treatment_effect() returned")

# ---- 6. variable_importance() ----
message("---- 6. variable_importance() ----")
vi <- variable_importance(cf)
stopifnot(is.numeric(vi))
stopifnot(length(vi) == p)
stopifnot(all(vi >= 0))
message("  Top 2 vars: ", paste(names(sort(vi, decreasing = TRUE))[1:2], collapse = ", "))
message("  OK: variable_importance() returned")

# ---- 7. print / summary ----
message("---- 7. print / summary ----")
capture.output(print(cf))
capture.output(summary(cf))
message("  OK: print and summary run without error")

# ---- 8. Correlation of predictions with true CATE (sanity) ----
message("---- 8. Prediction quality (cor with true CATE) ----")
pred_train <- predict(cf, X)$predictions
use <- !is.na(pred_train) & is.finite(pred_train)
if (sum(use) > 10) {
  cor_tau <- cor(pred_train[use], tau_true[use], use = "complete.obs")
  message("  Cor(pred CATE, true CATE): ", round(cor_tau, 4))
}

message("")
message("========== All tests passed. ==========")
