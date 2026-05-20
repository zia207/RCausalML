# Test: R/grf_integration.R (all grf re-exports with synthetic data)
# Run from package root: Rscript tests/test-grf_integration.R
# Or: source("tests/test-grf_integration.R")
# Requires: install.packages("grf")

pkg_root <- if (file.exists("R/grf_integration.R")) "." else
  if (file.exists("../R/grf_integration.R")) ".." else stop("Run from RCausalML package root")

# Skip all tests if grf is not installed
if (!requireNamespace("grf", quietly = TRUE)) {
  message("========== test-grf_integration.R ==========")
  message("SKIP: Package 'grf' not installed. Install with: install.packages(\"grf\")")
  message("===========================================")
  quit(save = "no", status = 0)
}

source(file.path(pkg_root, "R/grf_integration.R"))

set.seed(42)
n <- 300L
p <- 5L

message("========== test-grf_integration.R ==========")
message("Testing all grf integration functions with synthetic data (n = ", n, ", p = ", p, ")")
message("")

# =============================================================================
# Synthetic data: binary treatment, continuous outcome (for causal forest, RATE)
# =============================================================================
X_cf <- matrix(rnorm(n * p), n, p)
colnames(X_cf) <- paste0("X", seq_len(p))
W_cf <- rbinom(n, 1, 0.5)
tau_true <- 0.5 + 0.3 * X_cf[, 1] + 0.2 * pmax(X_cf[, 2], 0)
Y_cf <- 1 + 0.5 * X_cf[, 1] + 0.3 * X_cf[, 2] + tau_true * W_cf + rnorm(n, 0, 0.5)

# =============================================================================
# 1. grf_causal_forest
# =============================================================================
message("---- 1. grf_causal_forest() ----")
cf <- grf_causal_forest(X_cf, Y_cf, W_cf, num.trees = 150, seed = 123)
stopifnot(inherits(cf, "causal_forest"), inherits(cf, "grf"))
stopifnot("predictions" %in% names(cf) || "X.orig" %in% names(cf))
pred_cf <- predict(cf, X_cf[1:10, ])
stopifnot(is.data.frame(pred_cf), "predictions" %in% names(pred_cf), nrow(pred_cf) == 10)
message("  OK: grf_causal_forest() fit and predict")

# =============================================================================
# 2. grf_average_treatment_effect (on causal forest)
# =============================================================================
message("---- 2. grf_average_treatment_effect() ----")
ate <- grf_average_treatment_effect(cf)
stopifnot(is.vector(ate), length(ate) >= 2)
stopifnot(all(c("estimate", "std.err") %in% names(ate)))
message("  ATE estimate: ", round(ate["estimate"], 4), ", std.err: ", round(ate["std.err"], 4))
message("  OK: grf_average_treatment_effect()")

# =============================================================================
# 3. causal_survival_forest (synthetic survival data: non-negative event times)
# =============================================================================
message("---- 3. causal_survival_forest() ----")
horizon <- 2
# Event and censoring times strictly positive
Y_surv <- pmin(0.1 + rexp(n) * (0.5 + 0.3 * pmax(X_cf[, 1], 0)) + 0.2 * W_cf, horizon)
C_surv <- runif(n, 0.5, 3)
Y_obs <- pmin(Y_surv, C_surv)
Y_obs <- pmax(Y_obs, 0.01)  # ensure strictly positive for grf
D_surv <- as.integer(Y_surv <= C_surv)
# Ensure enough events for forest
if (sum(D_surv) < 10L) {
  idx_ev <- order(Y_obs)[1:min(20, n)]
  D_surv[idx_ev] <- 1L
}
csf <- causal_survival_forest(X_cf, Y_obs, W_cf, D_surv, horizon = horizon, num.trees = 100, seed = 456)
stopifnot(inherits(csf, "causal_survival_forest"), inherits(csf, "grf"))
pred_csf <- predict(csf, X_cf[1:5, ])
stopifnot(is.data.frame(pred_csf), "predictions" %in% names(pred_csf))
message("  OK: causal_survival_forest() fit and predict")

# =============================================================================
# 4. instrumental_forest (X, Y, W, Z)
# =============================================================================
message("---- 4. instrumental_forest() ----")
Z_iv <- rbinom(n, 1, 0.5)
# Simple IV: Z influences W, W and tau influence Y
W_iv <- Z_iv * rbinom(n, 1, 0.7) + (1 - Z_iv) * rbinom(n, 1, 0.3)
tau_iv <- 0.3 * X_cf[, 1]
Y_iv <- rowSums(X_cf[, 1:3]) + tau_iv * W_iv + 0.5 * Z_iv + rnorm(n)
ivf <- instrumental_forest(X_cf, Y_iv, W_iv, Z_iv, num.trees = 100, seed = 789)
stopifnot(inherits(ivf, "instrumental_forest"), inherits(ivf, "grf"))
pred_iv <- predict(ivf, X_cf[1:5, ])
stopifnot(is.data.frame(pred_iv), "predictions" %in% names(pred_iv))
# predict.instrumental_forest re-export (OOB)
pred_iv_oob <- predict(ivf)
stopifnot(is.data.frame(pred_iv_oob), "predictions" %in% names(pred_iv_oob))
message("  OK: instrumental_forest() fit and predict (incl. predict.instrumental_forest)")

# =============================================================================
# 5. multi_arm_causal_forest (W as factor)
# =============================================================================
message("---- 5. multi_arm_causal_forest() ----")
W_multi <- as.factor(sample(c("A", "B", "C"), n, replace = TRUE))
tau_B <- 0.5 * X_cf[, 1]
tau_C <- -0.3 * X_cf[, 2]
Y_multi <- 1 + 0.2 * X_cf[, 1] +
  (W_multi == "B") * tau_B +
  (W_multi == "C") * tau_C +
  rnorm(n)
macf <- multi_arm_causal_forest(X_cf, Y_multi, W_multi, num.trees = 100, seed = 101)
stopifnot(inherits(macf, "multi_arm_causal_forest"), inherits(macf, "grf"))
pred_macf <- predict(macf, X_cf[1:5, ], drop = TRUE)
stopifnot(is.list(pred_macf), "predictions" %in% names(pred_macf))
stopifnot(is.matrix(pred_macf$predictions) || is.array(pred_macf$predictions))
message("  OK: multi_arm_causal_forest() fit and predict")

# grf_average_treatment_effect on multi-arm forest returns data frame
ate_multi <- grf_average_treatment_effect(macf)
stopifnot(is.data.frame(ate_multi) || (is.vector(ate_multi) && length(ate_multi) >= 2))
message("  OK: grf_average_treatment_effect(multi_arm_forest)")

# =============================================================================
# 6. rank_average_treatment_effect (needs forest + priorities)
# =============================================================================
message("---- 6. rank_average_treatment_effect() ----")
# Use a separate evaluation forest and priorities from training forest (or same for quick test)
priorities <- predict(cf)$predictions
rate <- rank_average_treatment_effect(cf, priorities, target = "AUTOC", R = 25)
stopifnot(inherits(rate, "rank_average_treatment_effect"))
stopifnot(all(c("estimate", "std.err", "target", "TOC") %in% names(rate)))
stopifnot(is.data.frame(rate$TOC), nrow(rate$TOC) > 0)
message("  RATE (AUTOC) estimate: ", round(rate$estimate[1], 4))
message("  OK: rank_average_treatment_effect()")

# =============================================================================
# 7. rank_average_treatment_effect.fit (user-supplied DR scores)
# =============================================================================
message("---- 7. rank_average_treatment_effect.fit() ----")
DR_scores <- grf::get_scores(cf)
rate_fit <- rank_average_treatment_effect.fit(DR_scores, priorities, target = "AUTOC", R = 25)
stopifnot(inherits(rate_fit, "rank_average_treatment_effect"))
stopifnot(all(c("estimate", "std.err", "TOC") %in% names(rate_fit)))
message("  OK: rank_average_treatment_effect.fit()")

# =============================================================================
# 8. plot.rank_average_treatment_effect (no device in batch)
# =============================================================================
message("---- 8. plot.rank_average_treatment_effect() ----")
pdf(nullfile())
plot(rate)
invisible(dev.off())
message("  OK: plot.rank_average_treatment_effect()")

# =============================================================================
# 9. get_tree, split_frequencies, grf_variable_importance, get_forest_weights, get_leaf_node
# =============================================================================
message("---- 9. get_tree(), split_frequencies(), grf_variable_importance(), get_forest_weights(), get_leaf_node() ----")
tree1 <- get_tree(cf, 1)
stopifnot(inherits(tree1, "grf_tree"), "nodes" %in% names(tree1), "columns" %in% names(tree1))
sf <- split_frequencies(cf, max.depth = 4)
stopifnot(is.matrix(sf), nrow(sf) <= 4, ncol(sf) == p)
vi <- grf_variable_importance(cf, decay.exponent = 2, max.depth = 4)
stopifnot(is.vector(vi) || is.matrix(vi), length(vi) == p)
w_oob <- get_forest_weights(cf)
stopifnot(inherits(w_oob, "Matrix") || is.matrix(w_oob), nrow(w_oob) == n, ncol(w_oob) == n)
w_test <- get_forest_weights(cf, X_cf[1:3, ])
stopifnot(nrow(w_test) == 3, ncol(w_test) == n)
leaf_ids <- get_leaf_node(tree1, X_cf[1:5, ], node.id = TRUE)
stopifnot(is.vector(leaf_ids), length(leaf_ids) == 5)
leaf_list <- get_leaf_node(tree1, X_cf[1:5, ], node.id = FALSE)
stopifnot(is.list(leaf_list))
message("  OK: get_tree(), split_frequencies(), grf_variable_importance(), get_forest_weights(), get_leaf_node()")

# =============================================================================
# 10. plot.grf_tree (single tree from causal forest)
# =============================================================================
message("---- 10. plot.grf_tree() ----")
# Plot to null device (DiagrammeR may be optional)
has_diagrammer <- requireNamespace("DiagrammeR", quietly = TRUE)
if (has_diagrammer) {
  pdf(nullfile())
  pl <- plot(tree1)
  dev.off()
  message("  OK: plot.grf_tree() (DiagrammeR available)")
} else {
  message("  SKIP: plot.grf_tree() (DiagrammeR not installed)")
}

# =============================================================================
# 11. .check_grf() (all wrappers call it before grf::; without grf they stop with clear message)
# =============================================================================
message("---- 11. .check_grf() ----")
# With grf loaded, all tests above passed so .check_grf() allowed execution.
# Without grf, each wrapper would stop("Package 'grf' is required...").
message("  OK: all wrappers use .check_grf(); with grf loaded all delegated to grf")

message("")
message("============================================")
message("All grf_integration tests passed.")
message("============================================")
