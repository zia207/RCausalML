#!/usr/bin/env Rscript
# Run prerequisite chunks + training-loop chunk from 30-Causal-Variational-Autoencoder-CausalVAE.qmd

setwd("/home/zia207/Dropbox/WebSites/R_Website/Causal_Inference_R/Cursor_R/Causal_ML")

# ---- Chunk: setup-packages ----
packages <- c("torch", "R6", "ranger", "progress", "expm", "coro",
              "networkD3", "igraph", "tidyverse", "igraph", "ggraph",
              "corrplot", "gridExtra", "patchwork", "bnlearn",
              "SEMgraph", "RCausalML")
invisible(lapply(packages, function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}))

# ---- Chunk: device-setup ----
device <- if (torch::cuda_is_available()) "cuda" else "cpu"
cat("Using device:", device, "\n")
set.seed(42)
torch::torch_manual_seed(42L)

# ---- Chunk: generate-synthetic-data ----
generate_data <- function(n_samples = 10000L, latent_dim = 3L) {
  epsilon_mat <- matrix(
    rnorm(n_samples * latent_dim),
    nrow = n_samples, ncol = latent_dim
  )
  z_mat <- matrix(0.0, nrow = n_samples, ncol = latent_dim)
  z_mat[, 1] <- epsilon_mat[, 1]
  z_mat[, 2] <- z_mat[, 1] + epsilon_mat[, 2]
  z_mat[, 3] <- z_mat[, 2]^2 + epsilon_mat[, 3]
  x_mat <- matrix(0.0, nrow = n_samples, ncol = latent_dim)
  x_mat[, 1] <- z_mat[, 1] * z_mat[, 3]
  x_mat[, 2] <- sin(z_mat[, 2]) + z_mat[, 1]
  x_mat[, 3] <- z_mat[, 3]^2 + rnorm(n_samples, mean = 0, sd = 0.1)
  list(
    x = torch_tensor(x_mat, dtype = torch_float()),
    z = torch_tensor(z_mat, dtype = torch_float()),
    epsilon = torch_tensor(epsilon_mat, dtype = torch_float())
  )
}
synthetic_data <- generate_data()
x <- synthetic_data$x
true_z <- synthetic_data$z
true_epsilon <- synthetic_data$epsilon
normalize_data <- function(x) {
  mean_val <- x$mean(dim = 1L)
  std_val <- x$std(dim = 1L)
  list(
    x_norm = (x - mean_val) / std_val,
    mean = mean_val,
    std = std_val
  )
}
norm_result <- normalize_data(x)
x_norm <- norm_result$x_norm
x_mean <- norm_result$mean
x_std <- norm_result$std
n_samples <- nrow(x_norm)
train_size <- as.integer(0.8 * n_samples)
val_size <- as.integer(0.1 * n_samples)
test_size <- n_samples - train_size - val_size
x_train <- x_norm[1:train_size, ]
x_val <- x_norm[(train_size + 1):(train_size + val_size), ]
x_test <- x_norm[(train_size + val_size + 1):n_samples, ]
true_z_train <- true_z[1:train_size, ]
true_z_val <- true_z[(train_size + 1):(train_size + val_size), ]
true_z_test <- true_z[(train_size + val_size + 1):n_samples, ]
cat(sprintf("Train: %d, Val: %d, Test: %d\n", nrow(x_train), nrow(x_val), nrow(x_test)))

# ---- Chunk: loss-function (model, optimizer, scheduler, epochs) ----
model <- CausalVAE()$to(device = device)
optimizer <- optim_adam(model$parameters, lr = 5e-4, weight_decay = 1e-5)
scheduler <- lr_reduce_on_plateau(optimizer, mode = "min", factor = 0.5, patience = 60)
epochs <- 800L

# ---- Chunk: training-loop ----
train_losses <- numeric(epochs)
val_losses <- numeric(epochs)
warmup_epochs <- 120L

train_dataset <- torch::tensor_dataset(x_train)
batch_size <- if (torch::cuda_is_available()) 256L else 128L
train_loader <- torch::dataloader(train_dataset, batch_size = batch_size, shuffle = TRUE, num_workers = 0L)
val_dataset <- torch::tensor_dataset(x_val, true_z_val)
val_loader <- torch::dataloader(val_dataset, batch_size = batch_size, shuffle = FALSE, num_workers = 0L)

model <- model$to(device = device)

for (epoch in seq_len(epochs)) {
  gamma_scale <- min(1.0, (epoch) / warmup_epochs)

  model$train()
  train_loss_epoch <- 0.0
  train_batches <- coro::collect(train_loader)

  for (x_batch in train_batches) {
    if (is.list(x_batch)) x_batch <- x_batch[[1]]
    x_batch_on_device <- x_batch$to(device = device)

    optimizer$zero_grad()
    forward_out <- model$forward(x_batch_on_device)
    loss <- loss_function(
      x_batch_on_device,
      forward_out$dec_mu,
      forward_out$dec_logvar,
      forward_out$enc_mu,
      forward_out$enc_logvar,
      model, gamma_scale = gamma_scale
    )
    loss$backward()
    torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
    optimizer$step()
    train_loss_epoch <- train_loss_epoch + loss$item()
  }
  train_losses[epoch] <- train_loss_epoch / length(train_batches)

  model$eval()
  val_loss_epoch <- 0.0
  val_batches <- coro::collect(val_loader)

  torch::with_no_grad({
    for (x_val_batch in val_batches) {
      if (is.list(x_val_batch)) x_val_batch <- x_val_batch[[1]]
      x_val_batch_on_device <- x_val_batch$to(device = device)
      forward_out <- model$forward(x_val_batch_on_device)
      loss <- loss_function(
        x_val_batch_on_device,
        forward_out$dec_mu,
        forward_out$dec_logvar,
        forward_out$enc_mu,
        forward_out$enc_logvar,
        model, gamma_scale = gamma_scale
      )
      val_loss_epoch <- val_loss_epoch + loss$item()
    }
  })
  val_losses[epoch] <- val_loss_epoch / length(val_batches)

  scheduler$step(val_losses[epoch])

  if (epoch %% 25 == 0 || epoch > (epochs - 10)) {
    lr <- optimizer$param_groups[[1]]$lr
    cat(sprintf("Epoch %d, Train: %.4f, Val: %.4f, lr: %.2e\n",
                epoch, train_losses[epoch], val_losses[epoch], lr))
  }
}

cat("Training loop finished.\n")
