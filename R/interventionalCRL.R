# ==============================================================================
#  interventionalCRL.R
#  Interventional Causal Representation Learning (CRL) — R/torch implementation
# ==============================================================================

# ---------------------------------------------------------------------------
# Device selection (lazy; only when torch is actually available at runtime)
# ---------------------------------------------------------------------------
.interventional_crl_device <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Package 'torch' is required for InterventionalCRL. Install with install.packages('torch').")
  }
  if (torch::cuda_is_available()) torch::torch_device("cuda") else torch::torch_device("cpu")
}


# ==============================================================================
# InterventionalCRL Model
# ==============================================================================

InterventionalCRL <- if (requireNamespace("torch", quietly = TRUE)) torch::nn_module(
  classname = "InterventionalCRL",
  
  # ---------------------------------------------------------------------------
  initialize = function(input_dim  = 2048,
                        latent_dim = 3,
                        hidden_dim = 512,
                        num_env    = 3) {
    
    # ===== Encoder: q(z | x, e) =====
    # Concatenates molecule features x with one-hot environment e
    # Two hidden layers → (mu, logvar) for posterior q(z|x,e)
    self$enc_fc1    <- nn_linear(input_dim + num_env, hidden_dim)  # input → hidden
    self$enc_fc2    <- nn_linear(hidden_dim, hidden_dim)           # hidden → hidden
    self$enc_mu     <- nn_linear(hidden_dim, latent_dim)           # → posterior mean
    self$enc_logvar <- nn_linear(hidden_dim, latent_dim)           # → posterior log-variance
    
    # ===== Decoder: p(x | z) =====
    # Maps latent z back to input space
    # dec_mu uses sigmoid activation → Bernoulli probs for binary fingerprints
    # dec_logvar is carried through for generality (unused in Bernoulli loss)
    self$dec_fc1    <- nn_linear(latent_dim, hidden_dim)           # latent → hidden
    self$dec_fc2    <- nn_linear(hidden_dim, hidden_dim)           # hidden → hidden
    self$dec_mu     <- nn_linear(hidden_dim, input_dim)            # → Bernoulli means
    self$dec_logvar <- nn_linear(hidden_dim, input_dim)            # → log-variance (auxiliary)
    
    # ===== Environment-conditioned Prior: p(z | e) = N(mu_e, I) =====
    # Linear map: one-hot(e) → prior mean; log-variance fixed at 0 (unit variance)
    self$prior_mu_fc <- nn_linear(num_env, latent_dim)
  },
  
  # ---------------------------------------------------------------------------
  encode = function(x, e) {
    # x : [batch, input_dim]   — molecule features (e.g. Morgan fingerprints)
    # e : [batch, num_env]     — one-hot encoded environment index
    # Returns: list(mu, logvar), each [batch, latent_dim]
    inp <- torch_cat(list(x, e), dim = 2)     # [batch, input_dim + num_env]
    h   <- nnf_relu(self$enc_fc1(inp))        # first hidden layer
    h   <- nnf_relu(self$enc_fc2(h))          # second hidden layer
    list(
      mu     = self$enc_mu(h),                # [batch, latent_dim]
      logvar = self$enc_logvar(h)             # [batch, latent_dim]
    )
  },
  
  # ---------------------------------------------------------------------------
  reparameterize = function(mu, logvar) {
    # Reparameterization trick: z = mu + eps * std,  eps ~ N(0, I)
    # Clamping std avoids vanishing gradients near zero variance.
    std <- torch_exp(0.5 * logvar)$clamp(min = 1e-5)  # [batch, latent_dim]
    eps <- torch_randn_like(std)
    mu + eps * std                                      # [batch, latent_dim]
  },
  
  # ---------------------------------------------------------------------------
  decode = function(z) {
    # z : [batch, latent_dim]
    # Returns: list(mu, logvar)
    #   mu     : sigmoid outputs → Bernoulli probabilities in (0,1)  [batch, input_dim]
    #   logvar : auxiliary head kept for interface consistency        [batch, input_dim]
    h      <- nnf_relu(self$dec_fc1(z))       # first hidden layer
    h      <- nnf_relu(self$dec_fc2(h))       # second hidden layer
    mu     <- torch_sigmoid(self$dec_mu(h))   # Bernoulli probs ∈ (0, 1)
    logvar <- self$dec_logvar(h)
    list(mu = mu, logvar = logvar)
  },
  
  # ---------------------------------------------------------------------------
  forward = function(x, e) {
    # Full forward pass through the InterventionalCRL model.
    # x : [batch, input_dim]
    # e : [batch, num_env]     — one-hot environment encoding
    #
    # Returns all quantities needed for ELBO loss computation.
    enc          <- self$encode(x, e)
    z            <- self$reparameterize(enc$mu, enc$logvar)
    dec          <- self$decode(z)
    prior_mu     <- self$prior_mu_fc(e)              # [batch, latent_dim]
    prior_logvar <- torch_zeros_like(prior_mu)       # logvar = 0  ⟹  variance = 1
    
    list(
      enc_mu       = enc$mu,
      enc_logvar   = enc$logvar,
      dec_mu       = dec$mu,
      dec_logvar   = dec$logvar,
      prior_mu     = prior_mu,
      prior_logvar = prior_logvar
    )
  }
) else NULL


# ==============================================================================
# ELBO Loss Function
# ==============================================================================

#' Negative ELBO loss for the Interventional CRL model
#'
#' Two-term loss:
#'   1. Reconstruction — Binary Cross-Entropy (Bernoulli likelihood for
#'      binary Morgan fingerprints).
#'   2. KL divergence  — KL( q(z|x,e) || p(z|e) ), analytical Gaussian form.
#'
#' total = BCE_recon + KL
#'
#' @param x            Observed binary input          [batch, input_dim]
#' @param dec_mu       Decoder sigmoid means          [batch, input_dim]
#' @param enc_mu       Encoder posterior mean         [batch, latent_dim]
#' @param enc_logvar   Encoder posterior log-variance [batch, latent_dim]
#' @param prior_mu     Prior mean  p(z|e)             [batch, latent_dim]
#' @param prior_logvar Prior log-variance p(z|e)      [batch, latent_dim]
#'
#' @return Scalar tensor — total loss (lower is better)
interventional_elbo_loss <- function(x,
                                     dec_mu,
                                     enc_mu,    enc_logvar,
                                     prior_mu,  prior_logvar) {
  
  # --- 1. Reconstruction: Bernoulli BCE ---
  # dec_mu are sigmoid probabilities; x must be binary (0/1).
  # nnf_binary_cross_entropy returns element-wise losses:
  #   BCE = -[ x * log(p) + (1-x) * log(1-p) ]
  # Sum over the input_dim features, then mean over the batch.
  recon_loss <- nnf_binary_cross_entropy(
    dec_mu, x, reduction = "none"
  )$sum(dim = -1)$mean()                              # scalar (≥ 0)
  
  # --- 2. KL divergence: KL( N(enc_mu, exp(enc_logvar)) || N(prior_mu, exp(prior_logvar)) ) ---
  #
  # Closed-form Gaussian KL:
  #   KL = 0.5 * sum[ prior_lv - enc_lv
  #                   + (exp(enc_lv) + (enc_mu - prior_mu)^2) / exp(prior_lv) - 1 ]
  #
  # With prior_logvar = 0 (unit variance) this simplifies to the standard
  # KL( N(mu, s²) || N(prior_mu, 1) ) = 0.5 * [ s² + (mu - prior_mu)² - 1 - log(s²) ]
  kl_elem <- 0.5 * (
    prior_logvar - enc_logvar +
      (torch_exp(enc_logvar) + (enc_mu - prior_mu)^2) / torch_exp(prior_logvar) - 1
  )
  kl_loss <- kl_elem$sum(dim = -1)$mean()             # scalar (≥ 0)
  
  # Total negative ELBO
  recon_loss + kl_loss
}


# ==============================================================================
# Model & Optimizer Setup (optional; do not run at package load)
# ==============================================================================
.interventional_crl_demo_setup <- function() {
  model     <- InterventionalCRL()$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = 1e-3)
  list(model = model, optimizer = optimizer, epochs = 200L)
}
