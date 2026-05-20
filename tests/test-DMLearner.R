# Test all DMLearner variants (R/DMLearner.R) with simulated data
# Run from package root: Rscript tests/test-DMLearner.R
# Quick run (n=300): QUICK=1 Rscript tests/test-DMLearner.R
# Or: source("tests/test-DMLearner.R")
#
# Covers: LinearDML, SparseLinearDML, KernelDML, NonParamDML, CausalForestDML, base DMLearner,
#         fit_predict, estimate_ate, supplied propensity;
#         DynamicDMLearner (panel, native R) and
#         OrthoIVLearner, DMLIVLearner, NonParamDMLIVLearner (DML with IV, native R);
#         optional: doubleml_did_* (DoubleML + mlr3 + mlr3learners + mlr3measures);
#         optional: doubleml_pliv, doubleml_plr_fit_data, doubleml_plr_tune_data,
#         doubleml_make_*, doubleml_data_from_* (DoubleML + mlr3; tune needs mlr3tuning + paradox).

# Load package code (works from package root and R CMD check temp dirs)
pkg_root <- if (file.exists("R/DMLearner.R")) "." else if (file.exists("../R/DMLearner.R")) ".." else NA_character_
if (!is.na(pkg_root) && file.exists(file.path(pkg_root, "DESCRIPTION")) && requireNamespace("devtools", quietly = TRUE)) {
  suppressPackageStartupMessages(devtools::load_all(pkg_root, quiet = TRUE))
} else if (requireNamespace("RCausalML", quietly = TRUE)) {
  library(RCausalML)
} else {
  if (is.na(pkg_root)) {
    message("Skipping test-DMLearner.R: cannot locate package root and RCausalML is not installed.")
    quit(save = "no", status = 0)
  }
  source(file.path(pkg_root, "R/utils.R"))
  source(file.path(pkg_root, "R/meta_learners.R"))  # check_xgboost_dr, .xgb_reg_args
  source(file.path(pkg_root, "R/DMLearner.R"))
}

# Optional: set QUICK=1 for faster run (n=300)
n <- if (nzchar(Sys.getenv("QUICK", ""))) 300 else 800
p_x <- 5
set.seed(42)
X <- matrix(rnorm(n * p_x), n, p_x)
colnames(X) <- paste0("X", 1:p_x)

# Binary treatment (propensity from X1)
propensity <- 1 / (1 + exp(-X[, 1]))
treatment <- rbinom(n, 1, propensity)

# True CATE: linear in X1 (DML is well-suited for this)
true_cate <- 0.3 + 0.5 * X[, 1]
y <- 2 + 0.3 * X[, 1] + true_cate * treatment + rnorm(n, 0, 0.5)

true_ate <- mean(true_cate)

message("========== DMLearner test: n = ", n, ", p = ", p_x, " ==========")
message("True ATE: ", round(true_ate, 4))
message("")

# ---- 1. LinearDML ----
message("---- 1. LinearDML ----")
dml_lin <- LinearDML(model_y = "ranger", model_t = "ranger", n_fold = 3, seed = 123)
dml_lin <- fit(dml_lin, X, treatment, y)
te_lin <- predict(dml_lin, X)
cor_lin <- cor(te_lin, true_cate, use = "pairwise.complete.obs")
mse_lin <- mean((te_lin - true_cate)^2, na.rm = TRUE)
message("  Cor(pred CATE, true CATE): ", round(cor_lin, 4))
message("  MSE(pred vs true CATE): ", round(mse_lin, 4))
cf_lin <- coef(dml_lin)
ic_lin <- tryCatch(intercept(dml_lin), error = function(e) NA_real_)
message("  coef (first 3): ", paste(round(head(cf_lin, 3), 4), collapse = ", "))
message("  intercept: ", round(ic_lin, 4))
ate_lin <- estimate_ate(dml_lin, X, treatment, y, pretrain = TRUE, return_ci = TRUE)
message("  ATE: ", round(ate_lin$ate, 4), " [", round(ate_lin$ate_lb, 4), ", ", round(ate_lin$ate_ub, 4), "]")
message("")

# ---- 2. SparseLinearDML ----
message("---- 2. SparseLinearDML ----")
dml_sparse <- SparseLinearDML(model_y = "ranger", model_t = "ranger", n_fold = 3, seed = 123)
dml_sparse <- fit(dml_sparse, X, treatment, y)
te_sparse <- predict(dml_sparse, X)
cor_sparse <- cor(te_sparse, true_cate, use = "pairwise.complete.obs")
mse_sparse <- mean((te_sparse - true_cate)^2, na.rm = TRUE)
message("  Cor(pred CATE, true CATE): ", round(cor_sparse, 4))
message("  MSE(pred vs true CATE): ", round(mse_sparse, 4))
cf_sparse <- coef(dml_sparse)
message("  coef (first 3): ", if (length(cf_sparse) >= 3) paste(round(head(cf_sparse, 3), 4), collapse = ", ") else "OK")
message("")

# ---- 3. KernelDML ----
message("---- 3. KernelDML ----")
dml_kern <- KernelDML(model_y = "ranger", model_t = "ranger", dim = 15L, n_fold = 3, seed = 123)
dml_kern <- fit(dml_kern, X, treatment, y)
te_kern <- predict(dml_kern, X)
cor_kern <- cor(te_kern, true_cate, use = "pairwise.complete.obs")
mse_kern <- mean((te_kern - true_cate)^2, na.rm = TRUE)
message("  Cor(pred CATE, true CATE): ", round(cor_kern, 4))
message("  MSE(pred vs true CATE): ", round(mse_kern, 4))
message("")

# ---- 4. NonParamDML (ranger) ----
message("---- 4. NonParamDML (ranger) ----")
dml_np <- NonParamDML(model_y = "ranger", model_t = "ranger", model_final = "ranger", n_fold = 3, seed = 123)
dml_np <- fit(dml_np, X, treatment, y)
te_np <- predict(dml_np, X)
cor_np <- cor(te_np, true_cate, use = "pairwise.complete.obs")
mse_np <- mean((te_np - true_cate)^2, na.rm = TRUE)
message("  Cor(pred CATE, true CATE): ", round(cor_np, 4))
message("  MSE(pred vs true CATE): ", round(mse_np, 4))
message("")

# ---- 5. CausalForestDML (if grf available) ----
message("---- 5. CausalForestDML ----")
if (requireNamespace("grf", quietly = TRUE)) {
  dml_cf <- CausalForestDML(model_y = "ranger", model_t = "ranger", n_fold = 3, seed = 123, num_trees = 500L)
  dml_cf <- fit(dml_cf, X, treatment, y)
  te_cf <- predict(dml_cf, X)
  cor_cf <- cor(te_cf, true_cate, use = "pairwise.complete.obs")
  mse_cf <- mean((te_cf - true_cate)^2, na.rm = TRUE)
  message("  Cor(pred CATE, true CATE): ", round(cor_cf, 4))
  message("  MSE(pred vs true CATE): ", round(mse_cf, 4))
  ate_cf <- estimate_ate(dml_cf, X, treatment, y, pretrain = TRUE, return_ci = TRUE)
  message("  ATE: ", round(ate_cf$ate, 4), " [", round(ate_cf$ate_lb, 4), ", ", round(ate_cf$ate_ub, 4), "]")
} else {
  message("  (skipped: grf not installed)")
}
message("")

# ---- 6. DMLearner base (model_final = "lm") ----
message("---- 6. DMLearner (base, model_final = lm) ----")
dml_base <- DMLearner(model_y = "ranger", model_t = "ranger", model_final = "lm", n_fold = 3, seed = 123)
dml_base <- fit(dml_base, X, treatment, y)
te_base <- predict(dml_base, X)
message("  Mean pred CATE: ", round(mean(te_base), 4))
message("")

# ---- 7. fit_predict and estimate_ate ----
message("---- 7. fit_predict & estimate_ate ----")
dml2 <- LinearDML(n_fold = 3, seed = 456)
fp <- fit_predict(dml2, X, treatment, y, return_components = TRUE)
message("  fit_predict returned: ", paste(names(fp), collapse = ", "))
message("  Mean CATE: ", round(mean(fp$te), 4))
ate2 <- estimate_ate(fp$fit, X, treatment, y, pretrain = TRUE, return_ci = TRUE)
message("  ATE (pretrain): ", round(ate2$ate, 4), " [", round(ate2$ate_lb, 4), ", ", round(ate2$ate_ub, 4), "]")
message("")

# ---- 8. With supplied propensity ----
message("---- 8. LinearDML with supplied propensity ----")
dml_p <- LinearDML(n_fold = 3, seed = 123)
dml_p <- fit(dml_p, X, treatment, y, p = propensity)
te_p <- predict(dml_p, X)
message("  Cor(pred CATE, true CATE): ", round(cor(te_p, true_cate, use = "pairwise.complete.obs"), 4))
message("")

# ---- 9. DynamicDMLearner (panel; native R) ----
message("---- 9. DynamicDMLearner (panel, native R) ----")
n_panels <- 50
n_periods <- 3
n_panel <- n_panels * n_periods
set.seed(123)
groups_panel <- rep(seq_len(n_panels), each = n_periods)
X_panel <- matrix(rnorm(n_panel * 2), n_panel, 2)
colnames(X_panel) <- c("X1", "X2")
T_panel <- matrix(rnorm(n_panel * 2), n_panel, 2)
Y_panel <- rnorm(n_panel)
dyn <- DynamicDMLearner(cv = 2L, random_state = 123L)
dyn <- fit(dyn, Y = Y_panel, T = T_panel, X = X_panel, groups = groups_panel)
pred_dyn <- predict(dyn, X_panel[1:5, , drop = FALSE])
message("  Fitted OK; predict(X[1:5,]): ", paste(round(pred_dyn, 4), collapse = ", "))
eff_dyn <- tryCatch(effect(dyn, newdata = X_panel[1:2, , drop = FALSE], T0 = 0, T1 = 1), error = function(e) NULL)
if (!is.null(eff_dyn)) message("  effect(T0=0, T1=1) at 2 units: ", paste(round(eff_dyn, 4), collapse = ", "))
message("")

# ---- 10. DML with IV: OrthoIV, DMLIV, NonParamDMLIV (native R) ----
message("---- 10. DML with IV (OrthoIVLearner, DMLIVLearner, NonParamDMLIVLearner, native R) ----")
set.seed(456)
n_iv <- 200
X_iv <- matrix(rnorm(n_iv * 3), n_iv, 3)
Z_iv <- rbinom(n_iv, 1, 0.5)
T_iv <- as.numeric(Z_iv + rnorm(n_iv, 0, 0.5) > 0.5)
y_iv <- 1 + 0.5 * X_iv[, 1] + 0.3 * T_iv + rnorm(n_iv, 0, 0.3)
# OrthoIV
oiv <- OrthoIVLearner(cv = 2L, random_state = 456L, discrete_treatment = TRUE, discrete_instrument = TRUE)
oiv <- fit(oiv, Y = y_iv, T = T_iv, Z = Z_iv, X = X_iv)
pred_oiv <- predict(oiv, X_iv[1:3, , drop = FALSE])
message("  OrthoIVLearner: fitted OK; mean CATE (first 3): ", round(mean(pred_oiv), 4))
# DMLIV
div <- DMLIVLearner(cv = 2L, random_state = 456L, discrete_treatment = TRUE, discrete_instrument = TRUE)
div <- fit(div, Y = y_iv, T = T_iv, Z = Z_iv, X = X_iv)
pred_div <- predict(div, X_iv[1:3, , drop = FALSE])
message("  DMLIVLearner: fitted OK; mean CATE (first 3): ", round(mean(pred_div), 4))
# NonParamDMLIV
npiv <- NonParamDMLIVLearner(cv = 2L, random_state = 456L, discrete_treatment = TRUE, discrete_instrument = TRUE)
npiv <- fit(npiv, Y = y_iv, T = T_iv, Z = Z_iv, X = X_iv)
pred_npiv <- predict(npiv, X_iv[1:3, , drop = FALSE])
message("  NonParamDMLIVLearner: fitted OK; mean CATE (first 3): ", round(mean(pred_npiv), 4))
message("")

# ---- 11. DoubleML DiD wrappers (optional Suggests) ----
message("---- 11. DoubleML DiD (doubleml_did_*, optional) ----")
if (requireNamespace("DoubleML", quietly = TRUE) &&
    requireNamespace("mlr3", quietly = TRUE) &&
    requireNamespace("mlr3learners", quietly = TRUE) &&
    requireNamespace("mlr3measures", quietly = TRUE)) {
  suppressPackageStartupMessages({
    library(mlr3)
    library(mlr3learners)
  })
  set.seed(99)
  n_did <- 180L
  X_d <- matrix(rnorm(n_did * 3L), n_did, 3L)
  D_bin <- rbinom(n_did, 1L, 0.45)
  y0_d <- 0.2 * X_d[, 1L] + rnorm(n_did, 0, 0.4)
  y1_d <- y0_d + 0.15 * D_bin + rnorm(n_did, 0, 0.4)
  r_irm <- doubleml_did_linear(y1_d, y0_d, D_bin, X_d, n_folds = 3L, return_ml_object = FALSE)
  stopifnot(is.numeric(r_irm$ATT), identical(r_irm$mode, "binary"))
  r_ev <- doubleml_did_eval_linear(y1_d, y0_d, D_bin, X_d, n_folds = 3L, print_eval = FALSE)
  stopifnot(is.list(r_ev$eval_predictions), !is.null(r_ev$eval_predictions$ml_m))
  D_tri <- sample(0:2, n_did, replace = TRUE)
  r_plr <- doubleml_did_linear(y1_d, y0_d, D_tri, X_d, n_folds = 3L)
  stopifnot(identical(r_plr$mode, "discrete_multi"), length(r_plr$coef) >= 2L)
  D_ct <- runif(n_did)
  r_ct <- doubleml_did_linear(y1_d, y0_d, D_ct, X_d, n_folds = 3L)
  stopifnot(identical(r_ct$mode, "continuous"))
  message("  doubleml_did_linear / doubleml_did_eval_linear: binary, discrete multi, continuous OK")
} else {
  message("  (skipped: need DoubleML, mlr3, mlr3learners, mlr3measures)")
}
message("")

# ---- 12. DoubleML PLIV, simulators, data wrappers (optional Suggests) ----
message("---- 12. doubleml_pliv / doubleml_make_* / doubleml_data_from_* (optional) ----")
if (requireNamespace("DoubleML", quietly = TRUE) && requireNamespace("mlr3", quietly = TRUE)) {
  d_plr <- doubleml_make_plr_CCDDHNR2018(120L, dim_x = 8L, alpha = 0.4, return_type = "data.frame")
  stopifnot(nrow(d_plr) == 120L, all(c("y", "d") %in% names(d_plr)))
  d_pliv <- doubleml_make_pliv_CHS2015(100L, alpha = 1, dim_x = 12L, dim_z = 1L, return_type = "data.frame")
  stopifnot(nrow(d_pliv) == 100L, all(grepl("^X", names(d_pliv)[1:12])), "Z1" %in% names(d_pliv))
  Xp <- as.matrix(d_pliv[, grep("^X", names(d_pliv))])
  Zp <- as.matrix(d_pliv[, grep("^Z", names(d_pliv))])
  r_pliv <- doubleml_pliv(Xp, d_pliv$d, d_pliv$y, Zp, n_folds = 3L, n_rep = 1L)
  stopifnot(is.numeric(r_pliv$ate), is.numeric(r_pliv$ate_se))
  message("  doubleml_make_plr_CCDDHNR2018 / doubleml_make_pliv_CHS2015 / doubleml_pliv: OK")
  df_wrap <- data.frame(y = d_plr$y, d = d_plr$d, d_plr[, grep("^X", names(d_plr))])
  xn <- grep("^X", names(df_wrap), value = TRUE)
  dm <- doubleml_data_from_data_frame(df_wrap, y_col = "y", d_cols = "d", x_cols = xn)
  stopifnot(!is.null(dm), inherits(dm, "DoubleMLData"))
  message("  doubleml_data_from_data_frame: OK")
  ml <- doubleml_make_plr_CCDDHNR2018(60L, dim_x = 5L, return_type = "matrix")
  dm2 <- doubleml_data_from_matrix(X = ml$X, y = ml$y, d = ml$d)
  stopifnot(!is.null(dm2), inherits(dm2, "DoubleMLData"))
  message("  doubleml_data_from_matrix: OK")
  ml_l <- mlr3::lrn("regr.rpart", cp = 0.02)
  ml_m <- ml_l$clone()
  r_plr_fd <- doubleml_plr_fit_data(dm2, ml_l = ml_l, ml_m = ml_m, n_folds = 3L)
  stopifnot(!is.null(r_plr_fd), is.numeric(r_plr_fd$ate), is.numeric(r_plr_fd$ate_se))
  message("  doubleml_plr_fit_data: OK")
  if (requireNamespace("mlr3tuning", quietly = TRUE) && requireNamespace("paradox", quietly = TRUE)) {
    param_set <- list(
      ml_l = paradox::ps(cp = paradox::p_dbl(lower = 0.02, upper = 0.04)),
      ml_m = paradox::ps(cp = paradox::p_dbl(lower = 0.02, upper = 0.04))
    )
    tune_settings <- list(
      n_folds_tune = 2L,
      rsmp_tune = "cv",
      terminator = mlr3tuning::trm("evals", n_evals = 2L),
      algorithm = "grid_search",
      resolution = 2L
    )
    r_plr_tn <- doubleml_plr_tune_data(
      dm2,
      ml_l = mlr3::lrn("regr.rpart"),
      ml_m = mlr3::lrn("regr.rpart"),
      param_set = param_set,
      tune_settings = tune_settings,
      tune_on_folds = FALSE,
      n_folds = 3L
    )
    stopifnot(!is.null(r_plr_tn), is.numeric(r_plr_tn$ate))
    message("  doubleml_plr_tune_data: OK")
  } else {
    message("  doubleml_plr_tune_data: (skipped: need mlr3tuning and paradox)")
  }
} else {
  message("  (skipped: need DoubleML and mlr3)")
}
message("")

# ---- Summary ----
message("========== DMLearner test summary ==========")
message("True ATE: ", round(true_ate, 4))
message("LinearDML ATE: ", round(ate_lin$ate, 4))
message("DynamicDMLearner (panel) and OrthoIV/DMLIV/NonParamDMLIV (IV) are native R implementations.")
message("All DMLearner variants ran successfully.")
message("========== test-DMLearner.R done ==========")
