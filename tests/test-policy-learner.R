# Test: policy learners (R/policy_learner.R) with synthetic data
# Covers: policy_learner (Athey & Wager), DRPolicyTree, DRPolicyForest
# Run from package root: Rscript tests/test-policy-learner.R
# Or: source("tests/test-policy-learner.R")

pkg_root <- if (file.exists("R/policy_learner.R")) "." else if (file.exists("../R/policy_learner.R")) ".." else stop("Run from Causal_ML package root")

# Load package from source when run from package root (tests current R/*.R)
if (file.exists(file.path(pkg_root, "DESCRIPTION")) && requireNamespace("devtools", quietly = TRUE)) {
  tryCatch(
    { devtools::load_all(pkg_root, quiet = TRUE); message("Loaded CausalML from source.") },
    error = function(e) NULL
  )
}
if (!exists("DRPolicyTree", mode = "function")) {
  source(file.path(pkg_root, "R/utils.R"))
  source(file.path(pkg_root, "R/dataset.R"))
  source(file.path(pkg_root, "R/policy_learner.R"))
  if (!exists("fit", mode = "function")) fit <- function(obj, ...) UseMethod("fit")
  if (!exists("predict_proba", mode = "function")) predict_proba <- function(object, newdata, ...) UseMethod("predict_proba")
  if (!exists("feature_importances", mode = "function")) feature_importances <- function(object, ...) UseMethod("feature_importances")
}

# -------- Synthetic data (binary treatment) --------
set.seed(42)
n <- 800
p <- 6
d <- synthetic_data(mode = 1, n = n, p = p, sigma = 1.0)
X <- d$X
w <- d$w
y <- d$y
tau_true <- d$tau
e <- d$e

message("========== Policy learner tests: synthetic data (mode 1) ==========")
message("n = ", n, ", p = ", p)
message("Treatment rate: ", round(mean(w), 3))
message("Mean CATE (true): ", round(mean(tau_true), 4))
message("")

# =============================================================================
# 1. policy_learner (Athey & Wager: DR score + weighted classifier)
# =============================================================================
message("---- 1. policy_learner (Athey & Wager) ----")
pl <- policy_learner(
  outcome_learner = "ranger",
  treatment_learner = "glmnet",
  policy_learner = "rpart",
  n_fold = 3L,
  random_state = 123
)
pl <- fit(pl, X, w, y, control = rpart::rpart.control(maxdepth = 2))

policy_pred <- predict(pl, X)
value_pl <- mean(policy_pred * tau_true)
value_oracle <- mean((tau_true > 0) * tau_true)
agreement_pl <- mean(policy_pred == as.integer(tau_true > 0))

message("  Policy: treat ", sum(policy_pred), " of ", length(policy_pred), " units")
message("  Value (policy learner): ", round(value_pl, 6))
message("  Value (true optimal):   ", round(value_oracle, 6))
message("  Agreement with oracle:  ", round(agreement_pl, 4))

prob_treat <- predict_proba(pl, X)
stopifnot(length(prob_treat) == n, all(prob_treat >= 0 & prob_treat <= 1))
message("  predict_proba: min = ", round(min(prob_treat), 4), ", max = ", round(max(prob_treat), 4))
message("")

# =============================================================================
# 2. DRPolicyTree (EconML-style: cross-fitted outcome + policy tree)
# =============================================================================
message("---- 2. DRPolicyTree ----")
pt <- DRPolicyTree(
  model_regression = "ranger",
  model_propensity = "glmnet",
  control_name = 0,
  cv = 2L,
  max_depth = 4L,
  min_samples_split = 20L,
  min_samples_leaf = 10L,
  random_state = 42
)
pt <- fit(pt, X, w, y)

pred_pt <- predict(pt, X)
# pred_pt is recommended treatment (0 or 1); compare to oracle (1 when tau > 0)
agreement_pt <- mean(pred_pt == as.integer(tau_true > 0))
value_pt <- mean(pred_pt * tau_true)

message("  Policy: treat ", sum(pred_pt), " of ", length(pred_pt), " units")
message("  Value (DRPolicyTree):   ", round(value_pt, 6))
message("  Value (true optimal):   ", round(value_oracle, 6))
message("  Agreement with oracle:  ", round(agreement_pt, 4))

proba_pt <- predict_proba(pt, X)
stopifnot(nrow(proba_pt) == n, ncol(proba_pt) == 2)  # control + 1 treatment
stopifnot(all(proba_pt >= 0 & proba_pt <= 1))
message("  predict_proba dim: ", nrow(proba_pt), " x ", ncol(proba_pt))

imp_pt <- feature_importances(pt)
if (length(imp_pt) > 0) {
  message("  feature_importances (top 2): ", paste(head(names(sort(imp_pt, decreasing = TRUE)), 2), collapse = ", "))
}
message("")

# =============================================================================
# 3. DRPolicyForest (ensemble of policy trees)
# =============================================================================
message("---- 3. DRPolicyForest (n_estimators=5 for speed) ----")
pf <- DRPolicyForest(
  model_regression = "ranger",
  model_propensity = "glmnet",
  control_name = 0,
  cv = 2L,
  n_estimators = 5L,
  max_samples = 0.6,
  max_depth = 4L,
  min_samples_split = 20L,
  min_samples_leaf = 10L,
  random_state = 123
)
pf <- fit(pf, X, w, y)

pred_pf <- predict(pf, X)
agreement_pf <- mean(pred_pf == as.integer(tau_true > 0))
value_pf <- mean(pred_pf * tau_true)

message("  Policy: treat ", sum(pred_pf), " of ", length(pred_pf), " units")
message("  Value (DRPolicyForest): ", round(value_pf, 6))
message("  Value (true optimal):   ", round(value_oracle, 6))
message("  Agreement with oracle: ", round(agreement_pf, 4))

proba_pf <- predict_proba(pf, X)
stopifnot(nrow(proba_pf) == n, ncol(proba_pf) == 2)

imp_pf <- feature_importances(pf)
if (length(imp_pf) > 0) {
  message("  feature_importances (top 2): ", paste(head(names(sort(imp_pf, decreasing = TRUE)), 2), collapse = ", "))
}
message("")

# =============================================================================
# 4. Summary table
# =============================================================================
message("---- Summary ----")
message("  Model              | Value (policy) | Value (oracle) | Agreement")
message("  -------------------|---------------|----------------|----------")
message("  policy_learner     | ", sprintf("%.4f", value_pl), "       | ", sprintf("%.4f", value_oracle), "       | ", sprintf("%.2f", agreement_pl))
message("  DRPolicyTree       | ", sprintf("%.4f", value_pt), "       | ", sprintf("%.4f", value_oracle), "       | ", sprintf("%.2f", agreement_pt))
message("  DRPolicyForest     | ", sprintf("%.4f", value_pf), "       | ", sprintf("%.4f", value_oracle), "       | ", sprintf("%.2f", agreement_pf))
message("")
message("========== test-policy-learner.R done ==========")
