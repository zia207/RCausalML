# Test TemporalCausalVAE (R/tempoCausalVAE.R)
# Run from package root: Rscript tests/test-tempoCausalVAE.R
# Requires: install.packages("torch")

pkg_root <- if (file.exists("R/tempoCausalVAE.R")) "." else
  if (file.exists("../R/tempoCausalVAE.R")) ".." else
    stop("Run from Causal_ML package root")

if (!requireNamespace("torch", quietly = TRUE)) {
  message("SKIP: Package 'torch' not installed — skipping TemporalCausalVAE tests.")
  message("Install with: install.packages('torch')")
  quit(status = 0, save = "no")
}
suppressPackageStartupMessages(library(torch))

# Source the module (defines TemporalCausalVAE, temporal_causal_loss, device)
source(file.path(pkg_root, "R/tempoCausalVAE.R"))

message("========== TemporalCausalVAE tests (R/tempoCausalVAE.R) ==========")
message("")

# --- 1. Model instantiation ---
message("---- 1. TemporalCausalVAE instantiation ----")
model <- TemporalCausalVAE(
  obs_dim = 3L, latent_dim = 3L, hidden_dim = 64L,
  beta = 4.0, gamma = 1.0, lambda_sparsity = 0.1
)$to(device = device)
stopifnot(inherits(model, "nn_module"))
message("  OK")
message("")

# --- 2. Forward pass: x [batch, T, obs_dim] ---
message("---- 2. TemporalCausalVAE forward ----")
batch_size <- 8L
t_steps   <- 10L
obs_dim   <- 3L
x_seq     <- torch_randn(batch_size, t_steps, obs_dim)$to(device = device)
out       <- model(x_seq)
stopifnot(is.list(out))
stopifnot(c("enc_mu", "enc_logvar", "dec_mu", "dec_logvar", "z_seq") %in% names(out))
stopifnot(out$enc_mu$size(1) == batch_size, out$enc_mu$size(2) == 3L)
stopifnot(out$dec_mu$size(1) == batch_size, out$dec_mu$size(2) == obs_dim)
stopifnot(out$z_seq$size(1) == batch_size, out$z_seq$size(2) == t_steps, out$z_seq$size(3) == 3L)
message("  OK")
message("")

# --- 3. temporal_causal_loss ---
message("---- 3. temporal_causal_loss ----")
loss <- temporal_causal_loss(
  x          = x_seq,
  dec_mu     = out$dec_mu, dec_logvar = out$dec_logvar,
  enc_mu     = out$enc_mu, enc_logvar = out$enc_logvar,
  model      = model
)
stopifnot(inherits(loss, "torch_tensor"), loss$numel() == 1L)
message("  Loss value: ", round(loss$item(), 4))
message("  OK")
message("")

# --- 4. encode, reparameterize, causal_dynamics, decode, decode_per_step ---
message("---- 4. encode / reparameterize / causal_dynamics / decode / decode_per_step ----")
enc <- model$encode(x_seq)
stopifnot(length(enc) == 2L, names(enc) == c("mu", "logvar"))
stopifnot(enc$mu$size(1) == batch_size, enc$mu$size(2) == 3L)

z_init <- model$reparameterize(enc$mu, enc$logvar)
stopifnot(z_init$size(1) == batch_size, z_init$size(2) == 3L)

z_seq <- model$causal_dynamics(z_init, t_steps)
stopifnot(z_seq$size(1) == batch_size, z_seq$size(2) == t_steps, z_seq$size(3) == 3L)

dec <- model$decode(z_seq)
stopifnot(dec$mu$size(1) == batch_size, dec$mu$size(2) == obs_dim)

recon_per_step <- model$decode_per_step(z_seq)
stopifnot(recon_per_step$size(1) == batch_size, recon_per_step$size(2) == t_steps, recon_per_step$size(3) == obs_dim)
message("  OK")
message("")

# --- 5. dag_penalty ---
message("---- 5. dag_penalty ----")
h <- model$dag_penalty()
stopifnot(inherits(h, "torch_tensor"), h$numel() == 1L)
message("  DAG penalty h(M): ", round(h$item(), 4))
message("  OK")
message("")

# --- 6. Backward (gradient flow) ---
message("---- 6. backward ----")
optimizer <- optim_adam(model$parameters, lr = 1e-3)
loss <- temporal_causal_loss(
  x          = x_seq,
  dec_mu     = out$dec_mu, dec_logvar = out$dec_logvar,
  enc_mu     = out$enc_mu, enc_logvar = out$enc_logvar,
  model      = model
)
optimizer$zero_grad()
loss$backward()
optimizer$step()
message("  OK")
message("")

# --- 7. Demo setup (optional) ---
message("---- 7. .temporal_causalvae_demo_setup ----")
demo <- .temporal_causalvae_demo_setup()
stopifnot(is.list(demo), c("model", "optimizer", "epochs") %in% names(demo))
stopifnot(inherits(demo$model, "nn_module"), inherits(demo$optimizer, "torch_optimizer"))
stopifnot(demo$epochs == 200L)
message("  OK")
message("")

message("========== TemporalCausalVAE tests done ==========")
