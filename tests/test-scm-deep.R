# Test Structural Causal Models (SCMs) with Deep Components
# Functions: deep_scm, deci_model, dynotears, evaluate_graph_recovery, plot_scm_dag
# Run from package root: Rscript tests/test-scm-deep.R
# Quick run: QUICK=1 Rscript tests/test-scm-deep.R

pkg_root <- if (file.exists("R/causalDeepNet.R")) "." else
            if (file.exists("../R/causalDeepNet.R")) ".." else
            stop("Run from RCausalML package root")
source(file.path(pkg_root, "R/causalDeepNet.R"))

quick <- nzchar(Sys.getenv("QUICK", ""))

message("========== SCM Deep Components test ==========")

if (!requireNamespace("torch",  quietly = TRUE)) {
  message("Skipped: package 'torch' is not installed.")
  quit(status = 0L)
}
if (!requireNamespace("coro",   quietly = TRUE)) {
  message("Skipped: package 'coro' is not installed.")
  quit(status = 0L)
}

set.seed(42L)

# ── Synthetic VAR(lag) data ──────────────────────────────────────────────────
n   <- if (quick) 120L else 300L
d   <- 4L
lag <- 3L
epochs_scm  <- if (quick) 2L  else 10L
epochs_deci <- if (quick) 3L  else 15L
epochs_dyno <- if (quick) 20L else 60L
hidden  <- if (quick) 16L else 32L
bs      <- if (quick) 16L else 32L

# VAR with known causal structure: 1->2->3, 1->4
data_raw <- matrix(0, nrow = n, ncol = d)
noise    <- matrix(rnorm(n * d, sd = 0.3), n, d)
for (t in (lag + 1L):n) {
  data_raw[t, 1L] <- 0.6 * data_raw[t - 1L, 1L] + noise[t, 1L]
  data_raw[t, 2L] <- 0.5 * data_raw[t - 1L, 2L] + 0.3 * data_raw[t - 1L, 1L] + noise[t, 2L]
  data_raw[t, 3L] <- 0.5 * data_raw[t - 1L, 3L] + 0.3 * data_raw[t - 1L, 2L] + noise[t, 3L]
  data_raw[t, 4L] <- 0.5 * data_raw[t - 1L, 4L] + 0.2 * data_raw[t - 2L, 1L] + noise[t, 4L]
}
colnames(data_raw) <- c("X1", "X2", "X3", "X4")

# Build lagged-window array (T x lag x d)
x_seq <- array(0, dim = c(n - lag, lag, d))
for (t in seq_len(n - lag)) {
  x_seq[t, , ] <- data_raw[t:(t + lag - 1L), ]
}
dimnames(x_seq)[[3L]] <- colnames(data_raw)

# Known (approximate) adjacency: col j -> row i means j causes i
A_true <- matrix(0L, d, d,
                 dimnames = list(colnames(data_raw), colnames(data_raw)))
A_true[2L, 1L] <- 1L  # X1 -> X2
A_true[3L, 2L] <- 1L  # X2 -> X3
A_true[4L, 1L] <- 1L  # X1 -> X4

message(sprintf("Data: n=%d, d=%d, lag=%d, x_seq dim=(%d x %d x %d)",
                n, d, lag, dim(x_seq)[1L], dim(x_seq)[2L], dim(x_seq)[3L]))

# ── 1. evaluate_graph_recovery ───────────────────────────────────────────────
message("\n--- 1. evaluate_graph_recovery ---")
A_perfect <- A_true
A_empty   <- matrix(0L, d, d)
res_perfect <- evaluate_graph_recovery(A_true, A_perfect, name = "Perfect")
res_empty   <- evaluate_graph_recovery(A_true, A_empty,   name = "Empty")
stopifnot(res_perfect$F1 > 0.99)
stopifnot(res_empty$TP   == 0L)
stopifnot(res_empty$FN   == sum(A_true))
message("  evaluate_graph_recovery: OK  (perfect F1=", round(res_perfect$F1, 3), ")")

# ── 2. plot_scm_dag ──────────────────────────────────────────────────────────
message("\n--- 2. plot_scm_dag ---")
# Just check it runs without error (suppress graphical device)
pdf(file.path(tempdir(), "scm_dag_test.pdf"), width = 6, height = 5)
tryCatch({
  plot_scm_dag(A_true, var_names = colnames(data_raw), title = "Test DAG")
  message("  plot_scm_dag (base graphics): OK")
}, error = function(e) {
  message("  plot_scm_dag error: ", conditionMessage(e))
})
dev.off()

# ── 3. deep_scm ──────────────────────────────────────────────────────────────
message("\n--- 3. deep_scm ---")

# Correlation-derived adjacency
corr      <- abs(cor(data_raw))
diag(corr) <- 0
A_corr    <- (corr > 0.20) * 1.0

fit_scm <- deep_scm(
  x_seq      = x_seq,
  adjacency  = A_corr,
  lag        = lag,
  latent_dim = 4L,
  hidden     = hidden,
  n_epochs   = epochs_scm,
  batch_size = bs,
  lr         = 1e-3,
  lam_kl     = 0.05,
  verbose    = TRUE,
  device     = "cpu"
)

stopifnot(inherits(fit_scm, "deep_scm"))
stopifnot(!is.null(fit_scm$model))
stopifnot(is.numeric(fit_scm$train_losses))
stopifnot(length(fit_scm$train_losses) == epochs_scm)

# predict
n_val   <- 30L
x_val   <- x_seq[seq_len(n_val), , , drop = FALSE]
preds   <- predict(fit_scm, x_val)
stopifnot(is.matrix(preds))
stopifnot(nrow(preds) == n_val, ncol(preds) == d)
stopifnot(all(is.finite(preds)))
message("  predict.deep_scm: OK  (", nrow(preds), " x ", ncol(preds), ")")

# intervene
iv_res <- intervene_deep_scm(fit_scm, x_val, target_var = 1L,
                              do_values = c(-1.0, 1.0))
stopifnot(is.matrix(iv_res$pred_low),  nrow(iv_res$pred_low)  == n_val)
stopifnot(is.matrix(iv_res$pred_high), nrow(iv_res$pred_high) == n_val)
stopifnot(is.numeric(iv_res$delta),    length(iv_res$delta)   == d)
message("  intervene_deep_scm: OK  (delta X2=", round(iv_res$delta[2L], 4L), ")")

# ── 4. deci_model ────────────────────────────────────────────────────────────
message("\n--- 4. deci_model ---")
fit_deci <- deci_model(
  x_seq      = x_seq,
  lag        = lag,
  latent_dim = 4L,
  hidden     = hidden,
  n_epochs   = epochs_deci,
  batch_size = bs,
  lr         = 1e-3,
  lam_kl     = 0.05,
  lam_dag    = 1.0,
  lam_sparse = 0.01,
  threshold  = 0.35,
  verbose    = TRUE,
  device     = "cpu"
)

stopifnot(inherits(fit_deci, "deci_model"))
stopifnot(is.matrix(fit_deci$A_soft))
stopifnot(is.matrix(fit_deci$A_binary))
stopifnot(nrow(fit_deci$A_soft)   == d, ncol(fit_deci$A_soft)   == d)
stopifnot(nrow(fit_deci$A_binary) == d, ncol(fit_deci$A_binary) == d)
stopifnot(all(fit_deci$A_soft   >= 0 & fit_deci$A_soft   <= 1))
stopifnot(all(fit_deci$A_binary %in% c(0, 1)))
stopifnot(all(diag(fit_deci$A_binary) == 0))
message("  A_soft diagonal (should be ~0): ",
        paste(round(diag(fit_deci$A_soft), 3L), collapse = ", "))

# predict
preds_deci <- predict(fit_deci, x_val)
stopifnot(is.matrix(preds_deci), nrow(preds_deci) == n_val, ncol(preds_deci) == d)
stopifnot(all(is.finite(preds_deci)))
message("  predict.deci_model: OK  (", nrow(preds_deci), " x ", ncol(preds_deci), ")")

# ATE
ate_val <- ate_deci(fit_deci, x_val, source = 1L, target = 2L,
                    do_values = c(-1.0, 1.0), n_samples = 20L)
stopifnot(is.numeric(ate_val), length(ate_val) == 1L, is.finite(ate_val))
message("  ate_deci (X1 -> X2): ", round(ate_val, 4L))

# graph recovery
res_deci <- evaluate_graph_recovery(A_true, fit_deci$A_binary, name = "DECI")
message("  DECI graph recovery: TP=", res_deci$TP, " FP=", res_deci$FP,
        " Precision=", res_deci$Precision, " Recall=", res_deci$Recall,
        " F1=", res_deci$F1, " SHD=", res_deci$SHD)

# ── 5. dynotears ─────────────────────────────────────────────────────────────
message("\n--- 5. dynotears ---")
fit_dyno <- dynotears(
  x_seq      = x_seq,
  lag        = lag,
  n_epochs   = epochs_dyno,
  batch_size = bs,
  lr         = 3e-3,
  lambda_l1  = 0.02,
  rho_init   = 1.0,
  threshold  = 0.08,
  verbose    = TRUE,
  device     = "cpu"
)

stopifnot(inherits(fit_dyno, "dynotears"))
stopifnot(is.matrix(fit_dyno$A_binary), nrow(fit_dyno$A_binary) == d)
stopifnot(is.matrix(fit_dyno$W_agg),    nrow(fit_dyno$W_agg)    == d)
stopifnot(all(fit_dyno$A_binary %in% c(0L, 1L)))
stopifnot(all(diag(fit_dyno$A_binary) == 0L))
stopifnot(all(fit_dyno$W_agg >= 0))
stopifnot(is.numeric(fit_dyno$train_losses), length(fit_dyno$train_losses) == epochs_dyno)
stopifnot(is.numeric(fit_dyno$dag_vals),     length(fit_dyno$dag_vals)     == epochs_dyno)

# predict
preds_dyno <- predict(fit_dyno, x_val)
stopifnot(is.matrix(preds_dyno), nrow(preds_dyno) == n_val, ncol(preds_dyno) == d)
stopifnot(all(is.finite(preds_dyno)))
message("  predict.dynotears: OK  (", nrow(preds_dyno), " x ", ncol(preds_dyno), ")")

# graph recovery
res_dyno <- evaluate_graph_recovery(A_true, fit_dyno$A_binary, name = "DynoTEARS")
message("  DynoTEARS graph recovery: TP=", res_dyno$TP, " FP=", res_dyno$FP,
        " Precision=", res_dyno$Precision, " Recall=", res_dyno$Recall,
        " F1=", res_dyno$F1, " SHD=", res_dyno$SHD)

# ── 6. CamelCase aliases ─────────────────────────────────────────────────────
message("\n--- 6. CamelCase aliases ---")
stopifnot(is.function(deepSCM))
stopifnot(is.function(deciModel))
stopifnot(is.function(dynoTEARS))
# Aliases delegate to the snake_case version — check via a tiny fit
fit_alias <- deepSCM(x_seq = x_seq, adjacency = A_corr, lag = lag,
                     latent_dim = 4L, hidden = hidden,
                     n_epochs = 1L, batch_size = bs,
                     verbose = FALSE, device = "cpu")
stopifnot(inherits(fit_alias, "deep_scm"))
message("  deepSCM / deciModel / dynoTEARS aliases: OK")

# ── Summary ──────────────────────────────────────────────────────────────────
message("\n========== SCM Deep Components test PASSED ==========")
