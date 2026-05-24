# Test: uplift XGBoost (R/uplift_xgboost.R)
# Run from package root: Rscript tests/test-uplift_xgboost.R

pkg_root <- if (file.exists("R/uplift_xgboost.R")) "." else
  if (file.exists("../R/uplift_xgboost.R")) ".." else
    stop("Run from RCausalML package root")

source(file.path(pkg_root, "R/utils.R"))
source(file.path(pkg_root, "R/dataset.R"))
source(file.path(pkg_root, "R/uplift_xgboost.R"))

has_xgboost <- requireNamespace("xgboost", quietly = TRUE)

set.seed(42)
message("========== test-uplift_xgboost.R ==========")

if (!has_xgboost) {
  ok <- tryCatch(uplift_xgboost(matrix(1), c(0, 1), c(0, 1)), error = function(e) TRUE)
  stopifnot(identical(ok, TRUE))
  message("  xgboost not installed — validation-only checks passed")
  message("========== test-uplift_xgboost.R done (validation only) ==========")
  quit(save = "no", status = 0)
}

# ---- 1. Classification: binary treatment ----
message("---- 1. Classification (binary) ----")
out_cls <- make_uplift_classification(
  treatment_name = c("control", "treatment1"),
  n_samples = 800,
  n_classification_features = 6,
  n_classification_informative = 4,
  n_uplift_increase_dict = list(treatment1 = 3),
  n_uplift_decrease_dict = list(treatment1 = 2),
  delta_uplift_increase_dict = list(treatment1 = 0.12),
  delta_uplift_decrease_dict = list(treatment1 = -0.08),
  random_seed = 42
)
df <- out_cls$data
X <- as.matrix(df[, out_cls$X_names])
w <- as.integer(df$treatment_group_key != "control")
y <- df$conversion
idx <- sample(nrow(X), 600)
X_tr <- X[idx, , drop = FALSE]
X_te <- X[-idx, , drop = FALSE]
w_tr <- w[idx]
y_tr <- y[idx]

fit_cls <- uplift_xgboost(
  X_tr, w_tr, y_tr,
  n_estimators = 20,
  max_depth = 4,
  evaluation_function = "TLearner",
  random_state = 123
)
stopifnot(inherits(fit_cls, "uplift_xgboost"))
stopifnot(inherits(fit_cls, "uplift_xgb"))
stopifnot(fit_cls$task == "classification")
stopifnot(!is.null(fit_cls$fit_0), !is.null(fit_cls$fit_1))

pred_cls <- predict(fit_cls, X_te)
stopifnot(length(pred_cls) == nrow(X_te))
stopifnot(is.finite(mean(pred_cls)))
message("  mean CATE (test): ", round(mean(pred_cls), 4))

pred_full <- predict(fit_cls, X_te, full_output = TRUE)
stopifnot(all(c("control", "treatment1", "delta_treatment1") %in% names(pred_full)))
stopifnot(all(pred_full$control >= 0 & pred_full$control <= 1))
stopifnot(all(pred_full$treatment1 >= 0 & pred_full$treatment1 <= 1))
message("  full_output OK")

fit_cls2 <- uplift_xgboostClassifier(
  X_tr, w_tr, y_tr,
  n_estimators = 15,
  max_depth = 3,
  random_state = 456
)
stopifnot(fit_cls2$task == "classification")
message("  uplift_xgboostClassifier OK")

# ---- 2. Classification: multi-arm ----
message("---- 2. Classification (multi-arm) ----")
out_m <- make_uplift_classification(
  treatment_name = c("control", "treatment1", "treatment2"),
  n_samples = 600,
  n_classification_features = 5,
  n_classification_informative = 3,
  n_uplift_increase_dict = list(treatment1 = 2, treatment2 = 2),
  n_uplift_decrease_dict = list(treatment1 = 1, treatment2 = 1),
  delta_uplift_increase_dict = list(treatment1 = 0.1, treatment2 = 0.12),
  delta_uplift_decrease_dict = list(treatment1 = -0.05, treatment2 = -0.06),
  random_seed = 42
)
df_m <- out_m$data
X_m <- as.matrix(df_m[, out_m$X_names])

fit_multi <- uplift_xgboost(
  X_m, df_m$treatment_group_key, df_m$conversion,
  control_name = "control",
  n_estimators = 15,
  max_depth = 3,
  random_state = 1
)
stopifnot(inherits(fit_multi, "uplift_xgb_multi"))
stopifnot(length(fit_multi$treatment_names) == 2L)
stopifnot(length(fit_multi$models) == 2L)

pred_m <- predict(fit_multi, X_m[1:30, , drop = FALSE])
stopifnot(is.matrix(pred_m) || is.data.frame(pred_m))
stopifnot(ncol(pred_m) == 2L)

pred_m_full <- predict(fit_multi, X_m[1:30, , drop = FALSE], full_output = TRUE)
stopifnot("recommended_treatment" %in% names(pred_m_full))
stopifnot("max_delta" %in% names(pred_m_full))
message("  multi-arm full_output OK")

# ---- 3. Regression: T-learner XGBoost ----
message("---- 3. Regression (TLearner) ----")
out_reg <- make_uplift_regression(
  treatment_name = c("control", "treatment1"),
  n_samples = 700,
  n_regression_features = 6,
  n_regression_informative = 4,
  n_uplift_increase_dict = list(treatment1 = 3),
  n_uplift_decrease_dict = list(treatment1 = 2),
  delta_uplift_increase_dict = list(treatment1 = 0.5),
  delta_uplift_decrease_dict = list(treatment1 = -0.3),
  sigma = 1,
  random_seed = 42
)
df_r <- out_reg$data
X_r <- as.matrix(df_r[, out_reg$X_names])
w_r <- as.integer(df_r$treatment_group_key != "control")
y_r <- df_r$outcome

fit_reg <- uplift_xgboostRegressor(
  X_r, w_r, y_r,
  n_estimators = 30,
  max_depth = 4,
  evaluation_function = "TLearner",
  random_state = 99
)
stopifnot(fit_reg$task == "regression")
stopifnot(fit_reg$type == "tlearner")
stopifnot(!is.null(fit_reg$fit_0), !is.null(fit_reg$fit_1))

pred_reg <- predict(fit_reg, X_r[1:50, , drop = FALSE])
stopifnot(length(pred_reg) == 50L)
stopifnot(is.finite(mean(pred_reg)))
message("  T-learner mean CATE: ", round(mean(pred_reg), 4))

pred_reg_full <- predict(fit_reg, X_r[1:50, , drop = FALSE], full_output = TRUE)
stopifnot(all(c("control", "treatment1", "delta_treatment1") %in% names(pred_reg_full)))
message("  regression full_output OK")

# ---- 4. Unsupported criterion ----
message("---- 4. Unsupported criterion ----")
ok <- tryCatch(
  uplift_xgboost(X_tr, w_tr, y_tr, evaluation_function = "KL"),
  error = function(e) TRUE
)
stopifnot(identical(ok, TRUE))
message("  rejects non-TLearner criterion OK")

# ---- 5. print method ----
message("---- 5. print ----")
capture.output(print(fit_cls))
capture.output(print(fit_multi))
message("  print OK")

message("")
message("========== All uplift_xgboost tests passed. ==========")
