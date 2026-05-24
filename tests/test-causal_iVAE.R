# Test causal_iVAE (R/causal_iVAE.R)
# Run from package root: Rscript tests/test-causal_iVAE.R
# Requires: install.packages("torch")

if (!requireNamespace("torch", quietly = TRUE)) {
  message("SKIP: Package 'torch' not installed — skipping causal_iVAE tests.")
  message("Install with: install.packages('torch')")
  quit(status = 0, save = "no")
}
suppressPackageStartupMessages(library(torch))

# iVAE (ivae) is bundled in causalDeepNet.R; load from installed package
pkg_root <- if (file.exists("R/causalDeepNet.R")) "." else
  if (file.exists("../R/causalDeepNet.R")) ".." else
    stop("Run from RCausalML package root")

if (requireNamespace("devtools", quietly = TRUE)) {
  devtools::load_all(pkg_root, quiet = TRUE)
} else {
  lib_paths <- c(file.path(pkg_root, ".Rlibrary"), .libPaths())
  .libPaths(lib_paths)
  suppressPackageStartupMessages(library(RCausalML))
}

message("========== causal_iVAE tests (R/causal_iVAE.R) ==========")
message("")

# --- 1. iVAE instantiation ---
message("---- 1. iVAE instantiation ----")
model <- iVAE(input_dim = 4L, latent_dim = 2L, hidden_dim = 32L, n_aux = 3L)
stopifnot(inherits(model, "nn_module"))
message("  OK")
message("")

# --- 2. Forward pass (x, u) ---
message("---- 2. iVAE forward ----")
batch_size <- 8L
x_batch <- torch_randn(batch_size, 4L)
u_batch <- torch_randint(0L, 3L, batch_size, dtype = torch_long())  # 0-indexed, values in {0,1,2}
out <- model(x_batch, u_batch)
stopifnot(is.list(out))
stopifnot(c("enc_mu", "enc_logvar", "dec_mu", "dec_logvar", "prior_mu", "prior_logvar") %in% names(out))
stopifnot(out$enc_mu$size(1) == batch_size, out$enc_mu$size(2) == 2L)
stopifnot(out$dec_mu$size(1) == batch_size, out$dec_mu$size(2) == 4L)
stopifnot(out$prior_mu$size(1) == batch_size, out$prior_mu$size(2) == 2L)
message("  OK")
message("")

# --- 3. elbo_loss ---
message("---- 3. elbo_loss ----")
loss <- elbo_loss(
  x = x_batch,
  dec_mu = out$dec_mu, dec_logvar = out$dec_logvar,
  enc_mu = out$enc_mu, enc_logvar = out$enc_logvar,
  prior_mu = out$prior_mu, prior_logvar = out$prior_logvar
)
stopifnot(inherits(loss, "torch_tensor"), loss$numel() == 1L)
message("  Negative ELBO: ", round(loss$item(), 4))
message("  OK")
message("")

# --- 4. encode / decode / prior shapes ---
message("---- 4. encode, decode, prior ----")
enc <- model$encode(x_batch, u_batch)
stopifnot(length(enc) == 2L, names(enc) == c("mu", "logvar"))
stopifnot(enc$mu$size(1) == batch_size, enc$mu$size(2) == 2L)

z <- model$reparameterize(enc$mu, enc$logvar)
stopifnot(z$size(1) == batch_size, z$size(2) == 2L)

dec <- model$decode(z)
stopifnot(dec$mu$size(1) == batch_size, dec$mu$size(2) == 4L)

pri <- model$prior(u_batch)
stopifnot(pri$mu$size(1) == batch_size, pri$mu$size(2) == 2L)
stopifnot(all(as_array(pri$logvar == 0)))
message("  OK")
message("")

# --- 5. Gradient flow (backward) ---
message("---- 5. backward ----")
loss <- elbo_loss(
  x = x_batch,
  dec_mu = out$dec_mu, dec_logvar = out$dec_logvar,
  enc_mu = out$enc_mu, enc_logvar = out$enc_logvar,
  prior_mu = out$prior_mu, prior_logvar = out$prior_logvar
)
loss$backward()
message("  OK")
message("")

message("========== causal_iVAE tests done ==========")
