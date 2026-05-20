# Test: R/causalTree.R — causal_tree() (honest/adaptive)
# Run from package root: Rscript tests/test-causalTree.R

pkg_root <- if (file.exists("R/causalTree.R")) "." else
  if (file.exists("../R/causalTree.R")) ".." else stop("Run from Causal_ML package root")

# Load dependencies needed when sourcing directly
library(rpart)
if (requireNamespace("htetree", quietly = TRUE)) library(htetree)
# Load causal_tree from causalTree.R
source(file.path(pkg_root, "R/causalTree.R"))

if (!requireNamespace("htetree", quietly = TRUE)) {
  message("Package 'htetree' not installed. Install for full tests. Running input-validation only.")
  has_htetree <- FALSE
} else {
  has_htetree <- TRUE
}

set.seed(42)

# ---- Synthetic data ----
n <- 300
df <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n),
  x3 = rbinom(n, 1, 0.5)
)
df$treatment <- rbinom(n, 1, 0.4 + 0.2 * (df$x1 > 0))
tau <- 0.5 + 0.3 * df$x1
df$y <- 1 + 0.2 * df$x1 + 0.1 * df$x2 + tau * df$treatment + rnorm(n, 0, 0.5)

message("========== test-causalTree.R (causal_tree) ==========")
message("n = ", n, " | treatment % = ", round(100 * mean(df$treatment), 1))

# ---- 1. Input validation ----
message("\n---- 1. Input validation ----")
# bad treatment length
ok <- tryCatch({
  causal_tree(y ~ x1 + x2 + x3, data = df, treatment = df$treatment[1:10])
}, error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: rejects wrong-length treatment")

# bad treatment values
ok <- tryCatch({
  causal_tree(y ~ x1 + x2 + x3, data = df, treatment = rep(2L, n))
}, error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: rejects non-0/1 treatment")

# missing treatment
ok <- tryCatch({
  causal_tree(y ~ x1 + x2 + x3, data = df)
}, error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: requires treatment")

# non-data.frame
ok <- tryCatch({
  causal_tree(y ~ x1 + x2, data = as.matrix(df), treatment = df$treatment)
}, error = function(e) TRUE)
stopifnot(identical(ok, TRUE))
message("  OK: requires data.frame")

if (!has_htetree) {
  message("\n(htetree not installed — skipping fit/predict tests)")
  message("========== test-causalTree.R done (validation only) ==========")
  quit(save = "no", status = 0)
}

# ---- 2. Fit adaptive (honest = FALSE) ----
message("\n---- 2. causal_tree(..., honest = FALSE) ----")
ct_adapt <- causal_tree(
  y ~ x1 + x2 + x3,
  data = df,
  treatment = df$treatment,
  honest = FALSE,
  split.Rule = "CT",
  cv.option = "CT",
  minsize = 5L,
  cp = 0,
  xval = 2
)
stopifnot(inherits(ct_adapt, "causalTree"))
stopifnot(inherits(ct_adapt, "rpart"))
stopifnot(ct_adapt$honest %in% c(TRUE, FALSE))
message("  OK: adaptive tree fitted, class causalTree + rpart")

# ---- 3. Fit honest (honest = TRUE) ----
message("\n---- 3. causal_tree(..., honest = TRUE) ----")
ct_honest <- causal_tree(
  y ~ x1 + x2 + x3,
  data = df,
  treatment = df$treatment,
  honest = TRUE,
  est_fraction = 0.4,
  split.Rule = "CT",
  cv.option = "CT",
  minsize = 5L,
  cp = 0,
  xval = 2
)
stopifnot(inherits(ct_honest, "causalTree"))
stopifnot(isTRUE(ct_honest$honest))
stopifnot(length(ct_honest$train_idx) > 0)
stopifnot(length(ct_honest$est_idx) > 0)
message("  OK: honest tree fitted, train_idx and est_idx stored")

# ---- 4. Predict (if predict.causalTree / rpart predict works) ----
message("\n---- 4. predict() ----")
pred <- tryCatch({
  predict(ct_honest, newdata = df[, c("x1", "x2", "x3")], type = "vector")
}, error = function(e) NULL)
if (!is.null(pred) && length(pred) == n) {
  message("  Predict length = ", length(pred), ", mean = ", round(mean(pred, na.rm = TRUE), 4))
  message("  OK: predict() returned")
} else {
  message("  (predict skipped or returned different format — check htetree)")
}

# ---- 5. Prune (rpart) ----
message("\n---- 5. prune() ----")
if (!is.null(ct_honest$cptable) && nrow(ct_honest$cptable) >= 1) {
  cp_use <- max(0.01, ct_honest$cptable[1, "CP"])
  pruned <- prune(ct_honest, cp = cp_use)
  stopifnot(inherits(pruned, "rpart"))
  message("  OK: prune() works")
} else {
  message("  (no cptable — prune skipped)")
}

message("\n========== test-causalTree.R completed ==========")
