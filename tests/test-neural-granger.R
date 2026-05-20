# Test neural Granger causality models in R/causalDeepNet.R
# Run from package root: Rscript tests/test-neural-granger.R
# Quick run: QUICK=1 Rscript tests/test-neural-granger.R

pkg_root <- if (file.exists("R/causalDeepNet.R")) "." else if (file.exists("../R/causalDeepNet.R")) ".." else stop("Run from RCausalML package root")
source(file.path(pkg_root, "R/causalDeepNet.R"))

quick <- nzchar(Sys.getenv("QUICK", ""))

message("========== neural-granger test ==========")
if (!requireNamespace("torch", quietly = TRUE)) {
  message("Skipped: package 'torch' is not installed.")
  quit(status = 0L)
}

set.seed(123)
n <- if (quick) 90L else 240L
d <- 5L
lag <- 4L

# Build a simple multivariate AR process with cross-series effects.
x <- matrix(0, nrow = n, ncol = d)
noise <- matrix(rnorm(n * d, sd = 0.2), nrow = n, ncol = d)
for (t in 3:n) {
  x[t, 1] <- 0.55 * x[t - 1L, 1] + 0.20 * x[t - 1L, 2] + noise[t, 1]
  x[t, 2] <- 0.45 * x[t - 1L, 2] + 0.15 * x[t - 2L, 3] + noise[t, 2]
  x[t, 3] <- 0.60 * x[t - 1L, 3] + 0.10 * x[t - 1L, 1] + noise[t, 3]
  x[t, 4] <- 0.40 * x[t - 1L, 4] + 0.25 * x[t - 1L, 5] + noise[t, 4]
  x[t, 5] <- 0.50 * x[t - 1L, 5] + 0.15 * x[t - 2L, 1] + noise[t, 5]
}
colnames(x) <- paste0("V", seq_len(d))

epochs <- if (quick) 1L else 3L
batch_size <- if (quick) 16L else 32L
hidden <- if (quick) 8L else 16L

models <- c("cmlp", "clstm", "economysru", "nri")
fit <- neural_granger_ml(
  data = x,
  lag = lag,
  models = models,
  hidden = hidden,
  lam = 0.005,
  epochs = epochs,
  batch_size = batch_size,
  lr = 5e-4,
  val_split = 0.2,
  device = "cpu",
  verbose = FALSE
)

stopifnot(inherits(fit, "neural_granger_ml"))
stopifnot(all(models %in% names(fit$models)))
stopifnot(all(models %in% names(fit$histories)))
stopifnot(all(models %in% names(fit$causal_matrices)))
stopifnot(all(models %in% names(fit$val_mse)))

for (m in models) {
  cm <- fit$causal_matrices[[m]]
  stopifnot(is.matrix(cm), nrow(cm) == d, ncol(cm) == d)
  stopifnot(is.numeric(fit$val_mse[[m]]), length(fit$val_mse[[m]]) == 1L, is.finite(fit$val_mse[[m]]))
}

# Predict checks: cMLP expects flat lag features, others expect sequence array.
n_pred <- 7L
x_flat <- matrix(rnorm(n_pred * d * lag), nrow = n_pred, ncol = d * lag)
x_seq <- array(rnorm(n_pred * lag * d), dim = c(n_pred, lag, d))

pred_cmlp <- predict(fit, model = "cmlp", x_lagged = x_flat)
pred_clstm <- predict(fit, model = "clstm", x_lagged = x_seq)
pred_sru <- predict(fit, model = "economysru", x_lagged = x_seq)
pred_nri <- predict(fit, model = "nri", x_lagged = x_seq)

for (pred in list(pred_cmlp, pred_clstm, pred_sru, pred_nri)) {
  stopifnot(is.matrix(pred), nrow(pred) == n_pred, ncol(pred) == d)
  stopifnot(all(is.finite(pred)))
}

message("Models fitted: ", paste(names(fit$models), collapse = ", "))
message("Validation MSE: ", paste(sprintf("%s=%.4f", names(fit$val_mse), unlist(fit$val_mse)), collapse = " | "))
message("Prediction checks passed for cMLP, cLSTM, EconomySRU, NRI.")
message("========== test-neural-granger.R done ==========")
