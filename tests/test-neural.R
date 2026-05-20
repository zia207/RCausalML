# Test causalDeepNet.R (CEVAE, DragonNet, TARNet, CFRNet, GANITE, DCEVAE) with user dataset
# Run from package root: Rscript tests/test-neural.R
# Quick run (fewer epochs): QUICK=1 Rscript tests/test-neural.R

# Load package code (when run from package root)
pkg_root <- if (file.exists("R/causalDeepNet.R")) "." else if (file.exists("../R/causalDeepNet.R")) ".." else stop("Run from Causal_ML package root")
# So DCEVAE() finds inst/python and inst/dcevae when sourcing instead of library()
Sys.setenv(RCAUSALML_SOURCE_ROOT = normalizePath(pkg_root))
source(file.path(pkg_root, "R/causalDeepNet.R"))

# Optional: QUICK=1 reduces epochs for a faster run
quick <- nzchar(Sys.getenv("QUICK", ""))

# --- User dataset (exact specification) ---
set.seed(123)
n <- 1000
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
if (requireNamespace("data.table", quietly = TRUE)) {
  data <- data.table::data.table(
    Y = Y,
    T = factor(T_bin),
    T_num = T_bin,
    as.data.frame(X),
    W1 = W[, 1], W2 = W[, 2], W3 = W[, 3],
    true_cate = true_cate
  )
} else {
  data <- data.frame(
    Y = Y,
    T = factor(T_bin),
    T_num = T_bin,
    as.data.frame(X),
    W1 = W[, 1], W2 = W[, 2], W3 = W[, 3],
    true_cate = true_cate
  )
}

# Covariates for CATE (X only; CATE is function of X1)
if (inherits(data, "data.table")) {
  X_mat <- as.matrix(data[, ..x_cols])
} else {
  X_mat <- as.matrix(data[, x_cols])
}
if (is.null(colnames(X_mat))) colnames(X_mat) <- x_cols
treatment <- as.integer(data$T_num)
y <- as.numeric(data$Y)
true_cate_vec <- as.numeric(data$true_cate)

message("========== Neural.R test dataset: n = ", n, ", p = ", ncol(X_mat), " ==========")
message("True ATE (mean true_cate): ", round(mean(true_cate_vec), 4))
message("")

# Small epochs in quick mode
n_epoch_cevae   <- if (quick) 5L else 20L
n_epoch_dragon  <- if (quick) 5L else 20L
n_epoch_tarnet  <- if (quick) 10L else 50L
n_epoch_cfrnet  <- if (quick) 10L else 50L
n_iter_ganite   <- if (quick) 80L else 1000L
n_epoch_dcevae  <- if (quick) 3L else 30L

pehe <- function(pred, true) sqrt(mean((pred - true)^2, na.rm = TRUE))

# ---- 1. CEVAE ----
message("---- 1. CEVAE ----")
cv <- tryCatch({
  cevae(X_mat, treatment, y,
        num_epochs = n_epoch_cevae,
        batch_size = 100L,
        num_samples = if (quick) 50L else 200L,
        verbose = !quick)
}, error = function(e) { message("  Error: ", conditionMessage(e)); NULL })
if (!is.null(cv)) {
  ite_cevae <- predict(cv, X_mat, num_samples = if (quick) 50L else NULL)
  if (is.matrix(ite_cevae)) ite_cevae <- ite_cevae[, 1]
  cor_cevae <- cor(ite_cevae, true_cate_vec, use = "pairwise.complete.obs")
  pehe_cevae <- pehe(ite_cevae, true_cate_vec)
  message("  Cor(pred CATE, true CATE): ", round(cor_cevae, 4))
  message("  PEHE: ", round(pehe_cevae, 4))
  message("  Mean pred CATE: ", round(mean(ite_cevae), 4))
}
message("")

# ---- 2. DragonNet ----
message("---- 2. DragonNet ----")
drn <- tryCatch({
  dragonnet(X_mat, treatment, y,
            adam_epochs = n_epoch_dragon,
            sgd_epochs = if (quick) 10L else 50L,
            verbose = !quick)
}, error = function(e) { message("  Error: ", conditionMessage(e)); NULL })
if (!is.null(drn)) {
  ite_drn <- predict(drn, X_mat)
  cor_drn <- cor(ite_drn, true_cate_vec, use = "pairwise.complete.obs")
  pehe_drn <- pehe(ite_drn, true_cate_vec)
  message("  Cor(pred CATE, true CATE): ", round(cor_drn, 4))
  message("  PEHE: ", round(pehe_drn, 4))
  message("  Mean pred CATE: ", round(mean(ite_drn), 4))
}
message("")

# ---- 3. TARNet ----
message("---- 3. TARNet ----")
tar <- tryCatch({
  tarnet(X_mat, treatment, y,
         epochs = n_epoch_tarnet,
         verbose = !quick)
}, error = function(e) { message("  Error: ", conditionMessage(e)); NULL })
if (!is.null(tar)) {
  ite_tar <- predict(tar, X_mat)
  cor_tar <- cor(ite_tar, true_cate_vec, use = "pairwise.complete.obs")
  pehe_tar <- pehe(ite_tar, true_cate_vec)
  message("  Cor(pred CATE, true CATE): ", round(cor_tar, 4))
  message("  PEHE: ", round(pehe_tar, 4))
  message("  Mean pred CATE: ", round(mean(ite_tar), 4))
}
message("")

# ---- 4. CFRNet ----
message("---- 4. CFRNet ----")
cfr <- tryCatch({
  cfrnet(X_mat, treatment, y,
         epochs = n_epoch_cfrnet,
         verbose = !quick)
}, error = function(e) { message("  Error: ", conditionMessage(e)); NULL })
if (!is.null(cfr)) {
  ite_cfr <- predict(cfr, X_mat)
  cor_cfr <- cor(ite_cfr, true_cate_vec, use = "pairwise.complete.obs")
  pehe_cfr <- pehe(ite_cfr, true_cate_vec)
  message("  Cor(pred CATE, true CATE): ", round(cor_cfr, 4))
  message("  PEHE: ", round(pehe_cfr, 4))
  message("  Mean pred CATE: ", round(mean(ite_cfr), 4))
}
message("")

# ---- 5. GANITE ----
message("---- 5. GANITE ----")
gan <- tryCatch({
  ganite(X_mat, treatment, y,
         iterations = n_iter_ganite,
         verbose = !quick)
}, error = function(e) { message("  Error: ", conditionMessage(e)); NULL })
if (!is.null(gan)) {
  ite_gan <- predict(gan, X_mat)
  cor_gan <- cor(ite_gan, true_cate_vec, use = "pairwise.complete.obs")
  pehe_gan <- pehe(ite_gan, true_cate_vec)
  message("  Cor(pred CATE, true CATE): ", round(cor_gan, 4))
  message("  PEHE: ", round(pehe_gan, 4))
  message("  Mean pred CATE: ", round(mean(ite_gan), 4))
}
message("")

# ---- 6. DCEVAE (reticulate + Python torch; tabular r, d, a, y) ----
message("---- 6. DCEVAE ----")
if (!requireNamespace("reticulate", quietly = TRUE)) {
  message("  Skipped: package 'reticulate' not installed.")
} else {
  r_dce <- X_mat[, 1:2, drop = FALSE]
  d_dce <- X_mat[, 3:5, drop = FALSE]
  y_bin <- as.numeric(y > stats::median(y))
  dce <- tryCatch({
    DCEVAE(
      r_dce,
      d_dce,
      sensitive = treatment,
      y = y_bin,
      n_epochs = n_epoch_dcevae,
      n_batch_size = 128L,
      early_stop = FALSE,
      break_epoch = 100L,
      device = "cpu",
      seed = 123L
    )
  }, error = function(e) {
    message("  Error: ", conditionMessage(e))
    NULL
  })
  if (!is.null(dce)) {
    pr <- predict(dce, r = r_dce[1:20, , drop = FALSE], d = d_dce[1:20, , drop = FALSE],
                  a = treatment[1:20], y = y_bin[1:20])
    stopifnot(length(pr) == 20L, is.numeric(pr), all(is.finite(pr)))
    message("  predict() length ", length(pr), "; range [", round(min(pr), 4), ", ", round(max(pr), 4), "]")
  }
}
message("")

# ---- Summary ----
message("========== Summary ==========")
message("Neural learners (CEVAE, DragonNet, TARNet, CFRNet, GANITE, DCEVAE) tested on dataset.")
message("True ATE: ", round(mean(true_cate_vec), 4))
message("Done.")
