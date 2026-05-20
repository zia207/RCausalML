# Test: CATE interpreters (R/cate_interpreter.R)
# Covers: SingleTreeCateInterpreter, SingleTreePolicyInterpreter, interpret(), treat(), predict()
# Run from package root: Rscript tests/test-cate-interpreter.R
# Or: source("tests/test-cate-interpreter.R")

pkg_root <- if (file.exists("R/cate_interpreter.R")) "." else if (file.exists("../R/cate_interpreter.R")) ".." else stop("Run from Causal_ML package root")

if (file.exists(file.path(pkg_root, "DESCRIPTION")) && requireNamespace("devtools", quietly = TRUE)) {
  tryCatch(
    { devtools::load_all(pkg_root, quiet = TRUE); message("Loaded CausalML from source.") },
    error = function(e) NULL
  )
}
if (!exists("SingleTreeCateInterpreter", mode = "function")) {
  source(file.path(pkg_root, "R/utils.R"))
  source(file.path(pkg_root, "R/dataset.R"))
  source(file.path(pkg_root, "R/meta_learners.R"))
  source(file.path(pkg_root, "R/cate_interpreter.R"))
  if (!exists("fit", mode = "function")) fit <- function(obj, ...) UseMethod("fit")
}

# -------- Synthetic data (binary treatment); mode 1 requires p >= 5 --------
set.seed(42)
n <- 400
p <- 6
d <- synthetic_data(mode = 1, n = n, p = p, sigma = 1.0)
X <- d$X
w <- d$w
y <- d$y
tau_true <- d$tau
colnames(X) <- paste0("X", seq_len(p))

message("========== CATE interpreter tests: synthetic data (mode 1) ==========")
message("n = ", n, ", p = ", p)
message("Mean true CATE: ", round(mean(tau_true), 4))
message("")

# =============================================================================
# 1. SingleTreeCateInterpreter with SLearner
# =============================================================================
message("---- 1. SingleTreeCateInterpreter + SLearner ----")
sl <- SLearner(learner = "ranger", control_name = 0)
sl <- fit(sl, X, w, y)

interp_cate <- SingleTreeCateInterpreter(
  max_depth = 3L,
  min_samples_split = 20L,
  min_samples_leaf = 10L,
  random_state = 123L
)
interp_cate <- interpret(interp_cate, sl, X)

stopifnot(!is.null(interp_cate$tree_model_))
stopifnot(inherits(interp_cate$tree_model_, "rpart"))
stopifnot(!is.null(interp_cate$node_dict_))
stopifnot(length(interp_cate$node_dict_) >= 1L)

pred_cate <- predict(interp_cate, X)
stopifnot(length(pred_cate) == n, is.numeric(pred_cate), all(!is.na(pred_cate)))
message("  Tree CATE predictions: length ", length(pred_cate), ", range [", round(min(pred_cate), 4), ", ", round(max(pred_cate), 4), "]")

# Predict on subset (newdata)
X_sub <- X[1:50, , drop = FALSE]
pred_sub <- predict(interp_cate, X_sub)
stopifnot(length(pred_sub) == 50L)
message("  predict(newdata): OK (50 rows)")
message("")

# =============================================================================
# 2. SingleTreeCateInterpreter with TLearner
# =============================================================================
message("---- 2. SingleTreeCateInterpreter + TLearner ----")
tl <- TLearner(learner = "ranger", control_name = 0)
tl <- fit(tl, X, w, y)

interp_t <- SingleTreeCateInterpreter(max_depth = 4L, min_samples_leaf = 15L, random_state = 456L)
interp_t <- interpret(interp_t, tl, X)

pred_t <- predict(interp_t, X)
stopifnot(length(pred_t) == n)
message("  TLearner tree CATE: OK")
message("")

# =============================================================================
# 3. SingleTreePolicyInterpreter with SLearner
# =============================================================================
message("---- 3. SingleTreePolicyInterpreter + SLearner ----")
pol_interp <- SingleTreePolicyInterpreter(
  max_depth = 3L,
  min_samples_split = 20L,
  min_samples_leaf = 10L,
  random_state = 789L
)
pol_interp <- interpret(pol_interp, sl, X)

stopifnot(!is.null(pol_interp$tree_model_))
stopifnot(!is.null(pol_interp$policy_value_))
stopifnot(!is.null(pol_interp$always_treat_value_))
stopifnot(is.numeric(pol_interp$policy_value_), is.numeric(pol_interp$always_treat_value_))
stopifnot(!is.null(pol_interp$treatment_levels_))
stopifnot(!is.null(pol_interp$node_dict_))

trt <- treat(pol_interp, X)
stopifnot(length(trt) == n)
stopifnot(all(trt %in% c(0, 1)))
message("  policy_value_: ", round(pol_interp$policy_value_, 6))
message("  always_treat_value_: ", round(pol_interp$always_treat_value_, 6))
message("  treat(X): ", sum(trt == 1), " treated, ", sum(trt == 0), " control")

pred_pol <- predict(pol_interp, X)
stopifnot(identical(pred_pol, trt))
message("  predict() matches treat(): OK")

# treat on newdata
trt_sub <- treat(pol_interp, X_sub)
stopifnot(length(trt_sub) == 50L, all(trt_sub %in% c(0, 1)))
message("  treat(newdata): OK (50 rows)")
message("")

# =============================================================================
# 4. SingleTreePolicyInterpreter with sample_treatment_costs
# =============================================================================
message("---- 4. SingleTreePolicyInterpreter + sample_treatment_costs ----")
cost <- 0.1
pol_cost <- SingleTreePolicyInterpreter(max_depth = 3L, min_samples_leaf = 10L, random_state = 111L)
pol_cost <- interpret(pol_cost, sl, X, sample_treatment_costs = cost)

stopifnot(!is.null(pol_cost$policy_value_))
trt_cost <- treat(pol_cost, X)
stopifnot(length(trt_cost) == n, all(trt_cost %in% c(0, 1)))
message("  With cost ", cost, ": policy_value_ = ", round(pol_cost$policy_value_, 6))
message("")

# =============================================================================
# 5. Error: treat() before interpret()
# =============================================================================
message("---- 5. Error handling: treat() before interpret() ----")
pol_empty <- SingleTreePolicyInterpreter(max_depth = 2L)
err <- tryCatch(treat(pol_empty, X), error = function(e) e)
stopifnot(inherits(err, "error"))
stopifnot(grepl("interpret", conditionMessage(err), ignore.case = TRUE))
message("  treat() before interpret() raises error: OK")
message("")

# =============================================================================
# 6. SingleTreeCateInterpreter with data.frame X
# =============================================================================
message("---- 6. SingleTreeCateInterpreter with data.frame X ----")
X_df <- as.data.frame(X)
interp_df <- SingleTreeCateInterpreter(max_depth = 2L, min_samples_leaf = 20L)
interp_df <- interpret(interp_df, sl, X_df)
pred_df <- predict(interp_df, X_df)
stopifnot(length(pred_df) == n)
message("  data.frame X: OK")
message("")

# =============================================================================
# 7. Node dict structure (CATE interpreter)
# =============================================================================
message("---- 7. Node dict structure ----")
node1 <- interp_cate$node_dict_[[1L]]
stopifnot(is.list(node1), c("mean", "std") %in% names(node1))
stopifnot(is.numeric(node1$mean), is.numeric(node1$std))
message("  node_dict_ entries have mean and std: OK")
message("")

message("========== All cate_interpreter tests passed. ==========")
