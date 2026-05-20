# ==============================================================================
#  tempoCausalVAE.R
#  Temporal / Dynamical Causal Representation Learning (CRL) — R/torch
# ==============================================================================

# ---------------------------------------------------------------------------
# Device selection (lazy; only when torch is actually available at runtime)
# ---------------------------------------------------------------------------
.tempo_causal_vae_device <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Package 'torch' is required for TemporalCausalVAE. Install with install.packages('torch').")
  }
  if (torch::cuda_is_available()) torch::torch_device("cuda") else torch::torch_device("cpu")
}

# ==============================================================================
# TemporalCausalVAE Model
# ==============================================================================

#' Temporal Causal VAE (Variational Autoencoder with Causal Dynamics)
#'
#' A VAE for temporal sequences that learns a latent causal transition model.
#' Encoder (GRU) maps observed sequence to latent; dynamics evolve latent with
#' a learned adjacency; decoder (GRU) reconstructs from the latent trajectory.
#' Includes NOTEARS-style DAG penalty and sparsity on the adjacency.
#'
#' @param obs_dim Observation dimension (sequence feature size).
#' @param latent_dim Latent dimension (and adjacency size).
#' @param hidden_dim Hidden size for GRU and MLP layers.
#' @param beta KL weight in the loss.
#' @param gamma Weight for DAG + sparsity penalty.
#' @param lambda_sparsity Weight for L1 sparsity on adjacency.
#' @return An \code{nn_module} (R6) that can be trained with \code{temporal_causal_loss}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # function_name(...)
#' }
#' @export
TemporalCausalVAE <- if (requireNamespace("torch", quietly = TRUE)) torch::nn_module(
  classname = "TemporalCausalVAE",
  
  # ---------------------------------------------------------------------------
  initialize = function(obs_dim          = 3,
                        latent_dim       = 3,
                        hidden_dim       = 128,
                        beta             = 4.0,
                        gamma            = 1.0,
                        lambda_sparsity  = 0.1) {
    
    self$beta            <- beta
    self$gamma           <- gamma
    self$lambda_sparsity <- lambda_sparsity
    self$latent_dim      <- latent_dim
    
    # ===== Encoder: GRU maps observed sequence → final hidden state =====
    # GRU input: (batch, T, obs_dim)  →  hidden: (1, batch, hidden_dim)
    self$enc_gru    <- nn_gru(obs_dim, hidden_dim, batch_first = TRUE)
    self$enc_mu     <- nn_linear(hidden_dim, latent_dim)    # posterior mean
    self$enc_logvar <- nn_linear(hidden_dim, latent_dim)    # posterior log-variance
    
    # ===== Decoder: GRU maps latent sequence → final hidden state =====
    self$dec_gru    <- nn_gru(latent_dim, hidden_dim, batch_first = TRUE)
    self$dec_mu     <- nn_linear(hidden_dim, obs_dim)       # recon mean
    self$dec_logvar <- nn_linear(hidden_dim, obs_dim)       # recon log-variance
    
    # ===== Per-timestep decoder: direct latent → obs at each step =====
    self$dec_linear <- nn_linear(latent_dim, obs_dim)
    
    # ===== Causal adjacency logits: (latent_dim × latent_dim), trainable =====
    # nn_parameter() registers this tensor so the optimizer tracks gradients
    self$a_logits <- nn_parameter(
      torch_randn(latent_dim, latent_dim) * 0.01
    )
  },
  
  # ---------------------------------------------------------------------------
  encode = function(x) {
    # x      : [batch, T, obs_dim]
    # Returns: list(mu, logvar), each [batch, latent_dim]
    out    <- self$enc_gru(x)          # list(output=[batch,T,hidden], h=[1,batch,hidden])
    h      <- out[[2]]                 # [1, batch, hidden_dim]
    h_flat <- h$squeeze(1)$clone()     # [batch, hidden_dim]  (remove num_layers dim)
    list(
      mu     = self$enc_mu(h_flat),
      logvar = self$enc_logvar(h_flat)
    )
  },
  
  # ---------------------------------------------------------------------------
  reparameterize = function(mu, logvar) {
    # Reparameterization trick: z = mu + eps * std,  eps ~ N(0, I)
    std <- torch_exp(0.5 * logvar)     # [batch, latent_dim]
    eps <- torch_randn_like(std)
    mu + eps * std
  },
  
  # ---------------------------------------------------------------------------
  causal_dynamics = function(z_init, t_steps) {
    # Evolves the latent state using learned causal dynamics:
    #   z_t = z_{t-1} + A * z_{t-1}
    #
    # z_init  : [batch, latent_dim]
    # t_steps : integer
    # Returns : z_seq [batch, t_steps, latent_dim]
    
    A      <- nnf_sigmoid(self$a_logits)     # [latent_dim, latent_dim], in (0,1)
    z_list <- list(z_init)
    
    # seq_len(0) = integer(0) if t_steps == 1 → loop body is safely skipped
    for (t in seq_len(t_steps - 1)) {
      z_prev <- z_list[[length(z_list)]]     # [batch, latent_dim]
      # A @ z_prev^T  :  (latent_dim, latent_dim) × (batch, latent_dim, 1) → (batch, latent_dim, 1)
      z_next <- z_prev +
        torch_matmul(A, z_prev$unsqueeze(-1))$squeeze(-1)   # [batch, latent_dim]
      z_list <- c(z_list, list(z_next))
    }
    
    # Stack list of [batch, latent_dim] tensors along dim 2 → [batch, t_steps, latent_dim]
    torch_stack(z_list, dim = 2)
  },
  
  # ---------------------------------------------------------------------------
  decode = function(z_seq) {
    # z_seq  : [batch, t_steps, latent_dim]
    # Returns: list(mu, logvar), each [batch, obs_dim] — from final GRU hidden state
    out    <- self$dec_gru(z_seq)      # list(output, h)
    h      <- out[[2]]                 # [1, batch, hidden_dim]
    h_flat <- h$squeeze(1)$clone()     # [batch, hidden_dim]
    list(
      mu     = self$dec_mu(h_flat),
      logvar = self$dec_logvar(h_flat)
    )
  },
  
  # ---------------------------------------------------------------------------
  decode_per_step = function(z_seq) {
    # Projects every latent timestep directly to the observation space.
    # z_seq  : [batch, t_steps, latent_dim]
    # Returns: [batch, t_steps, obs_dim]
    self$dec_linear(z_seq)
  },
  
  # ---------------------------------------------------------------------------
  dag_penalty = function() {
    # NOTEARS acyclicity penalty: tr( exp(I + 0.1 * A ∘ A) ) − d
    # Equals zero iff A is a DAG (no directed cycles).
    A   <- nnf_sigmoid(self$a_logits)                      # [d, d]
    eye <- torch_eye(self$latent_dim, device = A$device)   # [d, d]
    M   <- eye + 0.1 * A * A                               # element-wise square of A
    torch_trace(torch_matrix_exp(M)) - self$latent_dim     # scalar
  },
  
  # ---------------------------------------------------------------------------
  forward = function(x) {
    # Full encode → sample → dynamics → decode pipeline.
    # x      : [batch, T, obs_dim]
    # Returns: named list of all quantities needed for the loss
    t_steps <- x$size(2)                                   # number of timesteps T
    
    enc    <- self$encode(x)
    z_init <- self$reparameterize(enc$mu, enc$logvar)
    z_seq  <- self$causal_dynamics(z_init, t_steps)        # [batch, T, latent_dim]
    dec    <- self$decode(z_seq)
    
    list(
      enc_mu     = enc$mu,
      enc_logvar = enc$logvar,
      dec_mu     = dec$mu,
      dec_logvar = dec$logvar,
      z_seq      = z_seq
    )
  }
) else NULL


# ==============================================================================
# Loss Function
# ==============================================================================

#' Total loss for the Temporal Causal VAE
#'
#' Combines four terms:
#'   1. Reconstruction loss — Gaussian log-likelihood at the *last* timestep
#'   2. KL loss             — KL( q(z|x) || N(0,1) ), analytical
#'   3. DAG loss            — NOTEARS acyclicity penalty
#'   4. Sparsity loss       — L1 on adjacency matrix entries
#'
#'   total = -recon + beta * KL + gamma * (DAG + lambda_sparsity * sparsity)
#'
#' @param x          Ground-truth sequence  [batch, T, obs_dim]
#' @param dec_mu     Decoder mean           [batch, obs_dim]
#' @param dec_logvar Decoder log-variance   [batch, obs_dim]
#' @param enc_mu     Encoder mean           [batch, latent_dim]
#' @param enc_logvar Encoder log-variance   [batch, latent_dim]
#' @param model      TemporalCausalVAE instance (hyperparams + penalty methods)
#'
#' @return Scalar tensor — total loss (lower is better)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # temporal_causal_loss(...)
#' }
#' @export
temporal_causal_loss <- function(x,
                                 dec_mu, dec_logvar,
                                 enc_mu, enc_logvar,
                                 model) {
  
  # --- 1. Reconstruction loss (last timestep only) ---
  # Clamp std > 0 for numerical stability
  dec_std    <- dec_logvar$exp()$sqrt()$clamp(min = 1e-6)   # [batch, obs_dim]
  x_last     <- x[, x$size(2), ]$contiguous()               # [batch, obs_dim]
  recon_loss <- distr_normal(dec_mu, dec_std)$
    log_prob(x_last)$sum(dim = -1)$mean()                    # scalar
  
  # --- 2. KL divergence: KL( N(mu, sigma^2) || N(0,1) ) — analytical ---
  #
  #  KL = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
  #
  # Equivalent to kl_divergence(Normal(mu, std), Normal(0,1)) in Python
  kl_loss <- (0.5 * (torch_exp(enc_logvar) + enc_mu^2 - 1 - enc_logvar))$
    sum(dim = -1)$mean()                                      # scalar
  
  # --- 3. DAG acyclicity penalty ---
  dag_loss      <- model$dag_penalty()                        # scalar
  
  # --- 4. Sparsity: L1 on sigmoid(a_logits) ---
  sparsity_loss <- nnf_sigmoid(model$a_logits)$abs()$sum()   # scalar
  
  # --- Combine all terms ---
  total_loss <- (
    -recon_loss
    + model$beta            * kl_loss
    + model$gamma           * (dag_loss + model$lambda_sparsity * sparsity_loss)
  )
  total_loss
}


# ==============================================================================
# Model & Optimizer Setup (optional; do not run at package load)
# ==============================================================================
#' @keywords internal
#' @noRd
.temporal_causalvae_demo_setup <- function() {
  model     <- TemporalCausalVAE()$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = 1e-3)
  list(model = model, optimizer = optimizer, epochs = 200L)
}
