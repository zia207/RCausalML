# Test CausalVAE (R/causalVAE.R)
# Run from package root: Rscript tests/test-causalVAE.R
# Requires: install.packages("torch")

if (!requireNamespace("torch", quietly = TRUE)) {
  message("SKIP: Package 'torch' not installed — skipping CausalVAE tests.")
  message("Install with: install.packages('torch')")
  quit(status = 0, save = "no")
}
suppressPackageStartupMessages(library(torch))

# CausalVAE is bundled in causalDeepNet.R; load from installed package
pkg_root <- if (file.exists("R/causalDeepNet.R")) "." else
  if (file.exists("../R/causalDeepNet.R")) ".." else
    stop("Run from RCausalML package root")

lib_paths <- c(file.path(pkg_root, ".Rlibrary"), .libPaths())
.libPaths(lib_paths)
suppressPackageStartupMessages(library(RCausalML))

message("========== CausalVAE tests (R/causalVAE.R) ==========")
message("")

# --- 1. generate_data ---
message("---- 1. generate_data ----")
set.seed(42)
dat <- generate_data(n_samples = 100L, latent_dim = 3L)
stopifnot(is.list(dat), c("x", "z", "epsilon") %in% names(dat))
stopifnot(inherits(dat$x, "torch_tensor"), dat$x$dim() == 2L, dat$x$size(1) == 100L, dat$x$size(2) == 3L)
stopifnot(inherits(dat$z, "torch_tensor"), dat$z$dim() == 2L)
stopifnot(inherits(dat$epsilon, "torch_tensor"), dat$epsilon$dim() == 2L)
message("  OK")
message("")

# --- 2. CausalVAE instantiation and forward ---
message("---- 2. CausalVAE forward ----")
device <- torch_device("cpu")
model <- CausalVAE(input_dim = 3L, latent_dim = 3L, hidden_dim = 32L)$to(device = device)
x_batch <- dat$x[1:8, , drop = FALSE]  # batch of 8
out <- model(x_batch)
stopifnot(is.list(out), c("enc_mu", "enc_logvar", "dec_mu", "dec_logvar", "z", "epsilon") %in% names(out))
stopifnot(out$enc_mu$size(1) == 8L, out$enc_mu$size(2) == 3L)
stopifnot(out$z$size(1) == 8L, out$z$size(2) == 3L)
stopifnot(out$dec_mu$size(1) == 8L, out$dec_mu$size(2) == 3L)
message("  OK")
message("")

# --- 3. loss_function ---
message("---- 3. loss_function ----")
loss <- loss_function(
  x = x_batch,
  dec_mu = out$dec_mu, dec_logvar = out$dec_logvar,
  enc_mu = out$enc_mu, enc_logvar = out$enc_logvar,
  model = model,
  gamma_scale = 1.0
)
stopifnot(inherits(loss, "torch_tensor"), loss$numel() == 1L)
message("  Loss value: ", round(loss$item(), 4))
message("  OK")
message("")

# --- 4. CausalVAE_ATE forward ---
message("---- 4. CausalVAE_ATE forward ----")
model_ate <- CausalVAE_ATE(input_dim = 3L, latent_dim = 3L, hidden_dim = 32L)$to(device = device)
T_batch <- torch_randint(0L, 2L, 8L)$unsqueeze(2L)$to(dtype = torch_float())  # [8, 1]
out_ate <- model_ate(x_batch, T = T_batch)
stopifnot("y_pred" %in% names(out_ate))
stopifnot(out_ate$y_pred$size(1) == 8L, out_ate$y_pred$size(2) == 1L)
message("  OK")
message("")

# --- 5. estimate_ate ---
message("---- 5. estimate_ate ----")
ate <- estimate_ate(model_ate, n_samples = 200L, treatment_dim = 1L, shift = 1.0)
stopifnot(is.numeric(ate), length(ate) == 1L)
message("  ATE estimate: ", round(ate, 4))
message("  OK")
message("")

# --- 6. dag_penalty ---
message("---- 6. dag_penalty ----")
h <- model$dag_penalty()
stopifnot(inherits(h, "torch_tensor"), h$numel() == 1L)
message("  DAG penalty h(M): ", round(h$item(), 4))
message("  OK")
message("")

message("========== CausalVAE tests done ==========")
