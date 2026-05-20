# Test all DR-Learner variants (R/meta_learners.R) with simulated data
# Run from package root: Rscript tests/test-DRLearner.R
# Quick run (n=300): QUICK=1 Rscript tests/test-DRLearner.R
# Or: source("tests/test-DRLearner.R")

# Load package code (when run from package root)
pkg_root <- if (file.exists("R/meta_learners.R")) "." else if (file.exists("../R/meta_learners.R")) ".." else stop("Run from RCausalML package root")
source(file.path(pkg_root, "R/utils.R"))
source(file.path(pkg_root, "R/meta_learners.R"))

# Optional: set QUICK=1 to use n=300 for a faster run (e.g. QUICK=1 Rscript tests/test-DRLearner.R)
n <- if (nzchar(Sys.getenv("QUICK", ""))) 300 else 1000
p_x <- 5
X <- matrix(rnorm(n * p_x), n, p_x)
colnames(X) <- paste0("X", 1:p_x)
W <- matrix(rnorm(n * 5), n, 5)

# Propensity: Pr[T=1 | X, W]
propensity <- 1 / (1 + exp(-X[, 1] - 0.5 * W[, 1]))
T_bin <- rbinom(n, 1, propensity)

# True CATE: non-linear in X1 (motivates forest)
true_cate <- 1 + 0.5 * X[, 1] + 0.3 * (X[, 1]^2)
Y <- as.vector(2 + 0.3 * (W[, 1:3] %*% c(1, -0.5, 0.5)) + true_cate * T_bin + rnorm(n, 0, 0.5))

x_cols <- paste0("X", 1:p_x)
data <- data.frame(
  Y = Y,
  T = factor(T_bin),
  T_num = T_bin,
  X,
  W1 = W[, 1], W2 = W[, 2], W3 = W[, 3],
  true_cate = true_cate
)

# Covariates for CATE: X only (or add W if desired)
X_mat <- as.matrix(data[, x_cols])
treatment <- data$T_num
y <- data$Y
p_prop <- propensity  # known propensity from DGP
true_cate_vec <- data$true_cate

message("========== Simulated data: n = ", n, ", p_x = ", p_x, " ==========")
message("True ATE (mean true_cate): ", round(mean(true_cate_vec), 4))
message("")

# ---- 1. DRLearner (ranger) ----
message("---- 1. DRLearner (ranger) ----")
dr <- DRLearner(learner = "ranger", n_fold = 3, seed = 123)
dr <- fit(dr, X_mat, treatment, y, p = p_prop)
te_dr <- predict(dr, X_mat)
if (is.vector(te_dr)) te_dr <- matrix(te_dr, ncol = 1)
cor_dr <- cor(te_dr[, 1], true_cate_vec, use = "pairwise.complete.obs")
mse_dr <- mean((te_dr[, 1] - true_cate_vec)^2, na.rm = TRUE)
message("  Cor(pred CATE, true CATE): ", round(cor_dr, 4))
message("  MSE(pred vs true CATE): ", round(mse_dr, 4))
ate_dr <- estimate_ate(dr, X_mat, treatment, y, p = p_prop, return_ci = TRUE)
message("  ATE estimate: ", round(ate_dr$ate, 4), " [", round(ate_dr$ate_lb, 4), ", ", round(ate_dr$ate_ub, 4), "]")
message("")

# ---- 2. DRLearner fit_predict with return_components ----
message("---- 2. DRLearner fit_predict (return_components) ----")
dr2 <- DRLearner(learner = "ranger", n_fold = 3, seed = 123)
fp <- fit_predict(dr2, X_mat, treatment, y, p = p_prop, return_components = TRUE)
message("  fit_predict returned: ", paste(names(fp), collapse = ", "))
message("  Mean CATE: ", round(mean(fp$te), 4))
message("")

# ---- 3. LinearDRLearner ----
message("---- 3. LinearDRLearner ----")
lin <- LinearDRLearner(model_regression = "ranger", fit_cate_intercept = TRUE, n_fold = 3, seed = 123)
lin <- fit(lin, X_mat, treatment, y, p = p_prop)
te_lin <- predict(lin, X_mat)
if (is.vector(te_lin)) te_lin <- matrix(te_lin, ncol = 1)
cor_lin <- cor(te_lin[, 1], true_cate_vec, use = "pairwise.complete.obs")
mse_lin <- mean((te_lin[, 1] - true_cate_vec)^2, na.rm = TRUE)
message("  Cor(pred CATE, true CATE): ", round(cor_lin, 4))
message("  MSE(pred vs true CATE): ", round(mse_lin, 4))
cf_lin <- coef(lin, treatment = 1)
ic_lin <- intercept(lin, treatment = 1)
message("  coef(treatment=1): ", paste(round(cf_lin, 3), collapse = ", "))
message("  intercept(treatment=1): ", round(ic_lin, 4))
message("")

# ---- 4. SparseLinearDRLearner ----
message("---- 4. SparseLinearDRLearner ----")
sparse <- SparseLinearDRLearner(alpha = 1, n_fold = 3, seed = 123)
sparse <- fit(sparse, X_mat, treatment, y, p = p_prop)
te_sparse <- predict(sparse, X_mat)
if (is.vector(te_sparse)) te_sparse <- matrix(te_sparse, ncol = 1)
cor_sparse <- cor(te_sparse[, 1], true_cate_vec, use = "pairwise.complete.obs")
mse_sparse <- mean((te_sparse[, 1] - true_cate_vec)^2, na.rm = TRUE)
message("  Cor(pred CATE, true CATE): ", round(cor_sparse, 4))
message("  MSE(pred vs true CATE): ", round(mse_sparse, 4))
cf_sparse <- coef(sparse, treatment = 1)
ic_sparse <- intercept(sparse, treatment = 1)
message("  coef(treatment=1) (first 3): ", if (length(cf_sparse) >= 3) paste(round(head(cf_sparse, 3), 3), collapse = ", ") else paste(round(cf_sparse, 3), collapse = ", "))
message("  intercept(treatment=1): ", round(ic_sparse, 4))
message("")

# ---- 5. XGBDRLearner ----
message("---- 5. XGBDRLearner ----")
if (requireNamespace("xgboost", quietly = TRUE)) {
  xgb <- XGBDRLearner(n_fold = 3, seed = 123)
  xgb <- fit(xgb, X_mat, treatment, y, p = p_prop)
  te_xgb <- predict(xgb, X_mat)
  if (is.vector(te_xgb)) te_xgb <- matrix(te_xgb, ncol = 1)
  cor_xgb <- cor(te_xgb[, 1], true_cate_vec, use = "pairwise.complete.obs")
  mse_xgb <- mean((te_xgb[, 1] - true_cate_vec)^2, na.rm = TRUE)
  message("  Cor(pred CATE, true CATE): ", round(cor_xgb, 4))
  message("  MSE(pred vs true CATE): ", round(mse_xgb, 4))
} else {
  message("  (skipped: xgboost not installed)")
}
message("")

# ---- 6. ForestDRLearner ----
message("---- 6. ForestDRLearner ----")
forest <- ForestDRLearner(n_fold = 3, seed = 123)
forest <- fit(forest, X_mat, treatment, y, p = p_prop)
te_forest <- predict(forest, X_mat)
if (is.vector(te_forest)) te_forest <- matrix(te_forest, ncol = 1)
cor_forest <- cor(te_forest[, 1], true_cate_vec, use = "pairwise.complete.obs")
mse_forest <- mean((te_forest[, 1] - true_cate_vec)^2, na.rm = TRUE)
message("  Cor(pred CATE, true CATE): ", round(cor_forest, 4))
message("  MSE(pred vs true CATE): ", round(mse_forest, 4))
message("")

# ---- 7. DRLearner without supplied propensity ----
message("---- 7. DRLearner (estimated propensity) ----")
dr_estp <- DRLearner(learner = "ranger", n_fold = 3, seed = 123)
dr_estp <- fit(dr_estp, X_mat, treatment, y, p = NULL)
te_estp <- predict(dr_estp, X_mat)
if (is.vector(te_estp)) te_estp <- matrix(te_estp, ncol = 1)
cor_estp <- cor(te_estp[, 1], true_cate_vec, use = "pairwise.complete.obs")
message("  Cor(pred CATE, true CATE): ", round(cor_estp, 4))
message("")

# ---- 8. estimate_ate with pretrain ----
message("---- 8. estimate_ate(..., pretrain = TRUE) ----")
ate_pretrain <- estimate_ate(dr, X_mat, treatment, y, p = p_prop, pretrain = TRUE, return_ci = TRUE)
message("  ATE (pretrain): ", round(ate_pretrain$ate, 4), " [", round(ate_pretrain$ate_lb, 4), ", ", round(ate_pretrain$ate_ub, 4), "]")
message("")

# ---- Summary ----
message("========== Summary ==========")
message("All DRLearner variants ran successfully.")
message("True ATE: ", round(mean(true_cate_vec), 4))
message("DR (ranger) ATE: ", round(ate_dr$ate, 4))
message("Done.")
