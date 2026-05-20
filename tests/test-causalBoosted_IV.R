# Test BoostedIVForest (R/causalBoosted_IV.R)
# Run from package root: Rscript tests/test-causalBoosted_IV.R
#
# Notes:
# - This is a lightweight smoke test (fast settings) to ensure the module loads,
#   a model can fit end-to-end, predict on new data with reordered columns, and
#   save/load round-trips.

pkg_root <- if (file.exists("R/causalBoosted_IV.R")) "." else
  if (file.exists("../R/causalBoosted_IV.R")) ".." else
    stop("Run from RCausalML package root")

req <- c("R6", "future", "future.apply", "xgboost", "ranger", "grf")
missing <- req[!vapply(req, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  message("SKIP: Missing required packages: ", paste(missing, collapse = ", "),
          ". Install them to run this test.")
  quit(status = 0, save = "no")
}

suppressPackageStartupMessages(library(future))

# Module dependency noted in causalBoosted_IV.R header.
source(file.path(pkg_root, "R/causalXGBoost.R"))
source(file.path(pkg_root, "R/causalBoosted_IV.R"))

message("========== BoostedIVForest tests (R/causalBoosted_IV.R) ==========")
message("")

future::plan(future::sequential)
on.exit(future::plan(future::sequential), add = TRUE)

# --- 1. Fit end-to-end on small simulated IV DGP ---
message("---- 1. fit() smoke test ----")
set.seed(1)
n <- 250L
p <- 5L
X <- matrix(rnorm(n * p), n, p, dimnames = list(NULL, paste0("X", seq_len(p))))
Z <- rbinom(n, 1, 0.5)
U <- rnorm(n)
sigmoid <- function(x) 1 / (1 + exp(-x))
# Make the instrument reasonably strong to avoid flaky weak-IV warnings.
W <- rbinom(n, 1, sigmoid(1.5 * Z - 0.5 * U))
tau_true <- 1 + X[, 1] - 0.5 * X[, 2]
Y <- tau_true * W + 0.25 * X[, 3] + 0.5 * U + rnorm(n)

# Inject some missing values to exercise imputation.
na_idx <- sample(length(X), size = floor(0.03 * length(X)))
X[na_idx] <- NA_real_

model <- BoostedIVForest$new(
  n_folds = 2L,
  nrounds = 10L,
  forest_args = list(num.trees = 200L)
)
model$fit(X, Y, W, Z, verbose = FALSE)
stopifnot(inherits(model, "BoostedIVForest"))
stopifnot(is.list(model$iv_diagnostics))
stopifnot(all(c("F_stat", "strong") %in% names(model$iv_diagnostics)))
message("  OK")
message("")

# --- 2. predict() with reordered columns (alignment) + newdata imputation ---
message("---- 2. predict() newdata column alignment ----")
X_new <- X[1:25, , drop = FALSE]

# Shuffle column order; keep names to allow alignment.
perm <- sample(seq_len(ncol(X_new)))
X_new_shuffled <- X_new[, perm, drop = FALSE]
stopifnot(!identical(colnames(X_new_shuffled), colnames(X_new)))

pred <- model$predict(X_new_shuffled)
stopifnot(is.list(pred))
stopifnot("predictions" %in% names(pred))
stopifnot(is.numeric(pred$predictions), length(pred$predictions) == nrow(X_new_shuffled))
stopifnot(all(is.finite(pred$predictions)))
message("  OK")
message("")

# --- 3. Evaluate helper ---
message("---- 3. evaluate() ----")
eval_res <- model$evaluate(tau_true)
stopifnot(is.list(eval_res), all(c("PEHE", "ATE_error") %in% names(eval_res)))
stopifnot(is.numeric(eval_res$PEHE), length(eval_res$PEHE) == 1L, is.finite(eval_res$PEHE))
stopifnot(is.numeric(eval_res$ATE_error), length(eval_res$ATE_error) == 1L, is.finite(eval_res$ATE_error))
message("  PEHE      : ", round(eval_res$PEHE, 4))
message("  ATE error : ", round(eval_res$ATE_error, 4))
message("  OK")
message("")

# --- 4. Save/load round-trip ---
message("---- 4. save/load ----")
tmp <- tempfile(fileext = ".rds")
save_boosted_iv(model, tmp)
model2 <- load_boosted_iv(tmp)
stopifnot(inherits(model2, "BoostedIVForest"))
ate1 <- model$average_treatment_effect()[["estimate"]]
ate2 <- model2$average_treatment_effect()[["estimate"]]
stopifnot(is.numeric(ate1), is.numeric(ate2), is.finite(ate1), is.finite(ate2))
stopifnot(abs(ate1 - ate2) < 1e-8)
message("  OK")
message("")

message("========== BoostedIVForest tests done ==========")

