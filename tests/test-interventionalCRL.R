# Test InterventionalCRL (R/interventionalCRL.R)
# Run from package root: Rscript tests/test-interventionalCRL.R
# Requires: install.packages("torch")

pkg_root <- if (file.exists("R/interventionalCRL.R")) "." else
  if (file.exists("../R/interventionalCRL.R")) ".." else
    stop("Run from Causal_ML package root")

if (!requireNamespace("torch", quietly = TRUE)) {
  message("SKIP: Package 'torch' not installed — skipping InterventionalCRL tests.")
  message("Install with: install.packages('torch')")
  quit(status = 0, save = "no")
}
suppressPackageStartupMessages(library(torch))

# Source the module (defines InterventionalCRL, interventional_elbo_loss, device, .interventional_crl_demo_setup)
source(file.path(pkg_root, "R/interventionalCRL.R"))

message("========== InterventionalCRL tests (R/interventionalCRL.R) ==========")
message("")

# Small dims for fast tests
input_dim  <- 64L
latent_dim <- 3L
hidden_dim <- 32L
num_env    <- 3L
batch_size <- 8L

# --- 1. Model instantiation ---
message("---- 1. InterventionalCRL instantiation ----")
model <- InterventionalCRL(
  input_dim  = input_dim,
  latent_dim = latent_dim,
  hidden_dim = hidden_dim,
  num_env    = num_env
)$to(device = device)
stopifnot(inherits(model, "nn_module"))
message("  OK")
message("")

# --- 2. Forward pass: x [batch, input_dim], e [batch, num_env] one-hot ---
message("---- 2. InterventionalCRL forward ----")
set.seed(42)
x_batch <- (torch_rand(batch_size, input_dim) > 0.5)$to(dtype = torch_float())  # binary fingerprints
e_onehot <- torch_zeros(batch_size, num_env)
e_idx <- torch_randint(1L, num_env + 1L, batch_size)
for (i in 1:batch_size) e_onehot[i, e_idx[i]$item()] <- 1

out <- model(x_batch, e_onehot)
stopifnot(is.list(out))
stopifnot(c("enc_mu", "enc_logvar", "dec_mu", "dec_logvar", "prior_mu", "prior_logvar") %in% names(out))
stopifnot(out$enc_mu$size(1) == batch_size, out$enc_mu$size(2) == latent_dim)
stopifnot(out$dec_mu$size(1) == batch_size, out$dec_mu$size(2) == input_dim)
stopifnot(out$prior_mu$size(1) == batch_size, out$prior_mu$size(2) == latent_dim)
message("  OK")
message("")

# --- 3. interventional_elbo_loss ---
message("---- 3. interventional_elbo_loss ----")
loss <- interventional_elbo_loss(
  x            = x_batch,
  dec_mu       = out$dec_mu,
  enc_mu       = out$enc_mu,
  enc_logvar   = out$enc_logvar,
  prior_mu     = out$prior_mu,
  prior_logvar = out$prior_logvar
)
stopifnot(inherits(loss, "torch_tensor"), loss$numel() == 1L)
message("  Loss value: ", round(loss$item(), 4))
message("  OK")
message("")

# --- 4. encode / reparameterize / decode ---
message("---- 4. encode / reparameterize / decode ----")
enc <- model$encode(x_batch, e_onehot)
stopifnot(length(enc) == 2L, names(enc) == c("mu", "logvar"))
stopifnot(enc$mu$size(1) == batch_size, enc$mu$size(2) == latent_dim)

z <- model$reparameterize(enc$mu, enc$logvar)
stopifnot(z$size(1) == batch_size, z$size(2) == latent_dim)

dec <- model$decode(z)
stopifnot(dec$mu$size(1) == batch_size, dec$mu$size(2) == input_dim)
stopifnot(dec$mu$min()$item() >= 0, dec$mu$max()$item() <= 1)  # sigmoid in (0,1)
message("  OK")
message("")

# --- 5. Backward (gradient flow) ---
message("---- 5. backward ----")
optimizer <- optim_adam(model$parameters, lr = 1e-3)
out2 <- model(x_batch, e_onehot)
loss2 <- interventional_elbo_loss(
  x = x_batch,
  dec_mu = out2$dec_mu,
  enc_mu = out2$enc_mu, enc_logvar = out2$enc_logvar,
  prior_mu = out2$prior_mu, prior_logvar = out2$prior_logvar
)
optimizer$zero_grad()
loss2$backward()
optimizer$step()
message("  OK")
message("")

# --- 6. Demo setup ---
message("---- 6. .interventional_crl_demo_setup ----")
demo <- .interventional_crl_demo_setup()
stopifnot(is.list(demo), c("model", "optimizer", "epochs") %in% names(demo))
stopifnot(inherits(demo$model, "nn_module"), inherits(demo$optimizer, "torch_optimizer"))
stopifnot(demo$epochs == 200L)
message("  OK")
message("")

message("========== InterventionalCRL tests done ==========")
