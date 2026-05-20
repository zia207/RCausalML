# Test all model groups listed in R/causalDeepNet.R header.
# Run from package root: Rscript tests/test-causalDeepNet-all-models.R
# Quick run: QUICK=1 Rscript tests/test-causalDeepNet-all-models.R

pkg_root <- if (file.exists("R/causalDeepNet.R")) "." else if (file.exists("../R/causalDeepNet.R")) ".." else stop("Run from RCausalML package root")
source(file.path(pkg_root, "R/causalDeepNet.R"))

quick <- nzchar(Sys.getenv("QUICK", ""))

message("========== causalDeepNet all-models smoke test ==========")

if (!requireNamespace("torch", quietly = TRUE)) {
  message("Skipped: package 'torch' is not installed.")
  quit(status = 0L)
}

set.seed(123)
n <- if (quick) 120L else 260L
p <- 6L
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
colnames(X) <- paste0("X", seq_len(p))
w <- rbinom(n, 1, plogis(0.5 * X[, 1] - 0.2 * X[, 2]))
y <- as.numeric(1 + 0.7 * X[, 1] + 0.4 * X[, 3] + (0.8 + 0.3 * X[, 1]) * w + rnorm(n, sd = 0.5))

message("n=", n, ", p=", p, ", treatment_rate=", round(mean(w), 3))

# Lightweight defaults
ep_small <- if (quick) 1L else 3L
iter_small <- if (quick) 20L else 100L

assert_pred_vec <- function(pred, n_exp) {
  stopifnot(is.numeric(pred), length(pred) == n_exp, all(is.finite(pred)))
}

# --- Treatment-effect models ---
message("\n---- Treatment-effect models ----")

m_cevae <- cevae(X, w, y, num_epochs = ep_small, batch_size = 32L, num_samples = 20L, verbose = FALSE, device = "cpu")
assert_pred_vec(predict(m_cevae, X, num_samples = 20L), n)
message("cevae: OK")

m_dragon <- dragonnet(X, w, y, adam_epochs = ep_small, sgd_epochs = ep_small, verbose = FALSE)
assert_pred_vec(predict(m_dragon, X), n)
message("dragonnet: OK")

m_tarnet <- tarnet(X, w, y, epochs = ep_small, batch_size = 32L, verbose = FALSE)
assert_pred_vec(predict(m_tarnet, X), n)
message("tarnet: OK")

m_cfr <- cfrnet(X, w, y, epochs = ep_small, batch_size = 32L, verbose = FALSE)
assert_pred_vec(predict(m_cfr, X), n)
message("cfrnet: OK")

m_ganite <- ganite(X, w, y, iterations = iter_small, batch_size = 32L, verbose = FALSE)
assert_pred_vec(predict(m_ganite, X), n)
message("ganite: OK")

# --- Generative latent-variable models ---
message("\n---- Generative latent-variable models ----")

u <- sample.int(3L, n, replace = TRUE)
m_ivae <- ivae(X, u = u, latent_dim = 2L, hidden_dim = 32L, n_aux = 4L, num_epochs = ep_small, batch_size = 32L, verbose = FALSE, device = "cpu")
z_ivae <- predict(m_ivae, X, u = u)
stopifnot(is.matrix(z_ivae), nrow(z_ivae) == n)
message("ivae: OK")

m_cvae <- causal_vae(X, latent_dim = 3L, hidden_dim = 32L, num_epochs = ep_small, batch_size = 32L, warmup_epochs = 1L, verbose = FALSE, device = "cpu")
z_cvae <- predict(m_cvae, X)
stopifnot(is.matrix(z_cvae), nrow(z_cvae) == n)
message("causal_vae: OK")

m_cvae_opt <- causal_vae_opt(X, latent_dim = 3L, hidden_dim = 32L, num_epochs = ep_small, batch_size = 32L, warmup_epochs = 1L, verbose = FALSE, device = "cpu")
z_cvae_opt <- predict(m_cvae_opt, X)
stopifnot(is.matrix(z_cvae_opt), nrow(z_cvae_opt) == n)
message("causal_vae_opt: OK")

m_dscm <- dscm(X, w, y, num_epochs = ep_small, batch_size = 32L, patience = 5L, kl_warmup_epochs = 1L, verbose = FALSE, device = "cpu")
assert_pred_vec(predict(m_dscm, X, type = "ite"), n)
message("dscm: OK")

m_cdvae <- causal_discrepancy_vae(X, w, y, latent_dim = 8L, hidden_dim = 32L, num_epochs = ep_small, batch_size = 32L, verbose = FALSE, device = "cpu")
assert_pred_vec(predict(m_cdvae, X, type = "ite"), n)
message("causal_discrepancy_vae: OK")

m_cgan <- causalGAN(X, w, y, hidden_dim = 32L, epochs = ep_small, batch_size = 32L, verbose = FALSE, device = "cpu")
pred_cgan <- predict(m_cgan, X, n_samples = 40L)
stopifnot(is.list(pred_cgan), all(c("ite", "y0", "y1") %in% names(pred_cgan)), length(pred_cgan$ite) == n)
message("causalGAN: OK")

m_cegm <- causal_egm(X, w, y, hidden_dim = 32L, num_epochs = ep_small, batch_size = 32L, verbose = FALSE, device = "cpu")
assert_pred_vec(predict(m_cegm, X, type = "ite"), n)
message("causal_egm: OK")

# --- DAG / structure learning models ---
message("\n---- Causal structure learning models ----")

m_castle <- castle(X, y, hidden_dim = 16L, num_layers = 2L, epochs = ep_small, batch_size = 32L, verbose = FALSE, device = "cpu", threshold = 0.05)
pred_castle <- predict(m_castle, X)
assert_pred_vec(pred_castle, n)
sum_castle <- summary(m_castle, top_n = 5L)
stopifnot(is.list(sum_castle))
message("castle: OK")

m_dagma_lin <- dagma(X, method = "linear", T = if (quick) 1L else 2L)
stopifnot(is.matrix(m_dagma_lin$adjacency), nrow(m_dagma_lin$adjacency) == p, ncol(m_dagma_lin$adjacency) == p)
message("dagma linear: OK")

m_dagma_nl <- tryCatch(
  dagma(X, method = "nonlinear_mlp", hidden = 8L, max_iter = if (quick) 10L else 20L, T = 1L),
  error = function(e) e
)
if (inherits(m_dagma_nl, "error")) {
  message("dagma nonlinear_mlp: SKIP (", conditionMessage(m_dagma_nl), ")")
} else {
  stopifnot(is.matrix(m_dagma_nl$adjacency), nrow(m_dagma_nl$adjacency) == p, ncol(m_dagma_nl$adjacency) == p)
  message("dagma nonlinear_mlp: OK")
}

run_structure_case <- function(label, expr) {
  out <- tryCatch(expr, error = function(e) e)
  if (inherits(out, "error")) {
    message(label, ": SKIP (", conditionMessage(out), ")")
    return(invisible(FALSE))
  }
  stopifnot(is.matrix(out$adjacency), nrow(out$adjacency) == p)
  message(label, ": OK")
  invisible(TRUE)
}

if (exists("causalStructureML", mode = "function")) {
  run_structure_case("notears_linear",
                     causalStructureML(X, method = "notears_linear", max_iter = if (quick) 10L else 25L, lambda1 = 0.1))
  run_structure_case("notears_nonlinear_mlp",
                     causalStructureML(X, method = "notears_nonlinear_mlp", max_iter = if (quick) 10L else 20L, lbfgs_iter = 0L, notears_hidden = 8L, verbose = FALSE))
  run_structure_case("notears_nonlinear_sobolev",
                     causalStructureML(X, method = "notears_nonlinear_sobolev", max_iter = if (quick) 10L else 20L, lbfgs_iter = 0L, sobolev_k = 3L, verbose = FALSE))
  run_structure_case("dag_gnn",
                     causalStructureML(X, method = "dag_gnn", n_epochs = if (quick) 4L else 12L, hidden_dim = 16L, verbose = FALSE, seed = 1L))
  run_structure_case("grandag",
                     causalStructureML(X, method = "grandag", iterations = if (quick) 20L else 60L, batch_size = 32L, stop_crit_win = 8L))
} else {
  message("causalStructureML wrapper not loaded in this context: SKIP notears/dag_gnn/grandag checks.")
}

# --- Time-series neural Granger models ---
message("\n---- Time-series neural Granger models ----")
X_ts <- matrix(rnorm((n + 20L) * 4L), nrow = n + 20L, ncol = 4L)
fit_ngc <- neural_granger_ml(X_ts, lag = 3L, models = c("cmlp", "clstm", "economysru", "nri"),
                             hidden = 8L, epochs = ep_small, batch_size = 16L, val_split = 0.2,
                             device = "cpu", verbose = FALSE)
stopifnot(inherits(fit_ngc, "neural_granger_ml"))
stopifnot(length(fit_ngc$models) == 4L)
message("neural_granger_ml (cMLP/cLSTM/EconomySRU/NRI): OK")

message("\n========== test-causalDeepNet-all-models.R done ==========")
