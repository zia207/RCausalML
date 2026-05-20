# causalDeepNet.R — Deep Causal Effect and Structure Learning Models for RCausalML
# ------------------------------------------------------------------------------
# Model summaries (grouped by category):
#
# ── Treatment Effect / ITE Estimators ─────────────────────────────────────────
#   • DragonNet            — Shared representation network with three heads:
#                            propensity score, Y(0), and Y(1). Targeted
#                            regularisation balances representation learning.
#                            Public API: dragonnet() / DragonNet().
#   • TARNet               — Treatment-Agnostic Representation Network: shared
#                            encoder, two separate outcome heads for treated /
#                            control; no representation-balancing penalty.
#                            Public API: tarnet() / TARNet().
#   • CFRNet               — CounterFactual Regression: TARNet + Maximum Mean
#                            Discrepancy (MMD) penalty to align treated and
#                            control latent distributions.
#                            Public API: cfrnet() / CFRNet().
#   • GANITE               — GAN-based ITE: generator produces counterfactual
#                            outcomes; discriminator enforces factual fidelity;
#                            inference network distils the generator.
#                            Public API: ganite() / GANITE().
#
# ── Generative Latent-Variable Causal Models ──────────────────────────────────
#   • CEVAE                — Causal Effect VAE: latent-confounder VAE that
#                            jointly models treatment assignment and potential
#                            outcomes; ELBO includes proxy-variable likelihood.
#                            Public API: cevae() / CEVAE().
#   • iVAE                 — Identifiable VAE: auxiliary-variable conditioned
#                            VAE with provably identifiable latent factors for
#                            causal structure recovery.
#   • CausalVAE            — VAE augmented with a learned causal graph over
#                            latent factors; acyclicity constraint (NOTEARS-
#                            style) and edge-sparsity penalty.
#   • CausalVAE-Opt        — Memory/speed-optimised variant of CausalVAE with
#                            the same architecture and loss.
#   • DSCM                 — Deep Structural Causal Model: fixed directed graph,
#                            per-variable SCM with abduction → action →
#                            prediction pipeline for counterfactuals.
#   • CausalDiscrepancyVAE — VAE with treatment-outcome prediction heads and
#                            MMD balancing in the latent space.
#   • CausalGAN            — Structural-equation GAN: X → T → Y causal chain;
#                            interventional samples obtained by do-calculus
#                            substitution inside the generator.
#                            Public API: causalGAN() / CausalGAN().
#   • CausalEGM            — Causal Encoding Generative Model: disentangled
#                            latent representation with identifiable causal
#                            factors and treatment/outcome decoder heads.
#                            Public API: causal_egm() / CausalEGM().
#
# ── Causal Structure Learning (DAG Discovery) ─────────────────────────────────
#   • CASTLE               — CAusal STructure LEarning Regularization: neural
#                            network with reconstruction loss + DAG acyclicity
#                            penalty + L1 edge-sparsity.
#   • NOTEARS-linear       — Sparse linear DAG via smooth acyclicity constraint
#                            h(W) = tr(e^{W ⊙ W}) − d = 0; L-BFGS-B solver.
#   • NOTEARS-nonlinear-MLP— Node-wise MLPs with NOTEARS acyclicity constraint;
#                            continuous optimisation over adjacency.
#   • NOTEARS-nonlinear-Sobolev — Sobolev basis nonlinear NOTEARS variant.
#   • DAGMA                — DAG learning via M-matrix log-det characterisation;
#                            linear and nonlinear (MLP) solvers.
#   • DAG-GNN              — Graph VAE for structure learning; encoder/decoder
#                            over node embeddings; augmented-Lagrangian DAG
#                            penalty on the learned adjacency.
#   • GraN-DAG             — Gradient-based Neural DAG learner: additive-noise
#                            model per variable, acyclicity via tr(exp(|A|)) − d.
#
# ── Time-Series Neural Granger Causality ──────────────────────────────────────
#   • cMLP                 — One MLP per target variable with group-lasso
#                            penalty over lagged predictor blocks.
#                            Public API: neural_granger_ml() / neuralGrangerML().
#   • cLSTM                — One LSTM per target with sparse input-weight
#                            structure encoding Granger relationships.
#   • EconomySRU           — Structured Recurrent Unit with a learnable binary
#                            causal mask matrix and group sparsity.
#   • NRI                  — Neural Relational Inference: encoder infers latent
#                            edge types; decoder propagates messages over the
#                            inferred graph to predict future states.
#
# ── Structural Causal Models (SCMs) with Deep Components ──────────────────────
#   • DeepSCM              — Fixed-graph SCM; per-variable variational noise
#                            encoders; ELBO training; do-calculus interventions
#                            via abduction-action-prediction.
#                            Public API: deep_scm() / deepSCM().
#   • DECI                 — Deep End-to-end Causal Inference: jointly learns
#                            the causal graph and structural equations under a
#                            NOTEARS acyclicity penalty; Monte-Carlo ATE.
#                            Public API: deci_model() / deciModel().
#   • DynoTEARS            — Lagged time-series DAG discovery using augmented-
#                            Lagrangian acyclicity on the lag-0 and lag-k
#                            adjacency matrices.
#                            Public API: dynotears() / dynoTEARS().
#   Graph utilities:       evaluate_graph_recovery(), plot_scm_dag()
#
# ── Attention-Based / Transformer Causal Models (v0.2.0) ──────────────────────
#   • TCDFNet              — Temporal Causal Discovery Framework (Nauta et al.,
#                            2019): stacked causal dilated convolutions with a
#                            per-variable attention head for variable importance.
#                            Public API: tcdf_model() / TCDFModel().
#   • CausalTransformer    — Transformer encoder with autoregressive causal
#                            masking and inter-variable cross-attention; the
#                            aggregated attention weights form the causal graph.
#                            Public API: causal_transformer_model() /
#                                        CausalTransformerModel().
#   • TFTNet               — Temporal Fusion Transformer (Lim et al., 2021):
#                            GRN-based variable-selection networks, LSTM encoder,
#                            interpretable multi-head temporal self-attention,
#                            outer-product variable causal matrix.
#                            Public API: tft_model() / TFTModel().
#   Unified entry point:   attn_causal_model() / attnCausalModel()
#   Causal graph accessor: causal_matrix_attn()
#   Prediction:            predict.attn_causal_model()
#
# ── RNN/LSTM-Based Causal Models (v0.3.0) ─────────────────────────────────────
#   • CausalLSTM           — Multi-layer stacked LSTM with a learnable soft
#                            causal-adjacency mask G ∈ [0,1]^{d×d} (sigmoid-
#                            gated logits); L1 sparsity penalty on off-diagonal
#                            entries prunes spurious edges; hard thresholding at
#                            inference yields a binary causal graph.
#                            Public API: causal_lstm_model() / CausalLSTMModel().
#   • RETAIN               — Reverse-time GRU with two-channel interpretable
#                            attention (Choi et al., 2016): temporal attention α
#                            weights visits by importance; variable attention β
#                            weights features per visit; outer product forms a
#                            d×d causal attribution matrix.
#                            Public API: retain_model() / RetainModel().
#   • InterventionRNN      — LSTM augmented with a soft GRU-based regime
#                            detector (n_regimes latent states) and an explicit
#                            intervention-indicator channel; regime embeddings
#                            concatenated to LSTM input enable intervention-
#                            conditioned and regime-specific causal matrices.
#                            Public API: intervention_rnn_model() /
#                                        InterventionRNNModel().
#   Unified entry point:   rnn_causal_model() / rnnCausalModel()
#   Causal graph accessor: causal_matrix_rnn()
#   Prediction:            predict.rnn_causal_model()
#
# ── Graph Neural Network (GNN) Causal Models (v0.3.0) ────────────────────────
#   • GVAR                 — Graph Vector Autoregression: lag-specific soft
#                            adjacency matrices A^(k) ∈ [0,1]^{d×d} (sigmoid-
#                            gated logits, no self-loops) jointly learned with
#                            two stacked GNN message-passing layers (per-lag
#                            linear transform + adjacency-weighted aggregation);
#                            L1 sparsity + NOTEARS DAG penalty on lag-averaged
#                            adjacency; GELU output head.
#                            Public API: gvar_model() / GVARModel().
#   • CausalGNN / CD-GNN   — Causal Discovery GNN: per-variable GRU temporal
#                            encoder produces node embeddings; bilinear-style
#                            graph learner infers a soft DAG; stacked
#                            EdgeConvLayers apply edge-conditioned message
#                            passing (concat neighbour pair + edge weight → MLP
#                            → single-step GRU update + LayerNorm); NOTEARS DAG
#                            + sparsity penalties.
#                            Public API: causal_gnn_model() / CausalGNNModel().
#   • CUTS+                — Causal discovery Under missing Time Series:
#                            variational Bernoulli graph posterior
#                            q(G) ≈ ∏ Bernoulli(π_ij); joint imputation network
#                            (observed values + binary missing mask → imputed
#                            sequence); GRU temporal encoder; EdgeConvLayer
#                            message passing; trained with
#                            MSE + KL(q ‖ sparse prior p=0.1) + NOTEARS penalty.
#                            Public API: cuts_model() / CUTSModel().
#   Unified entry point:   gnn_causal_model() / gnnCausalModel()
#   Causal graph accessor: causal_matrix_gnn()
#   Prediction:            predict.gnn_causal_model()
#   Location:              R/causalForest.R
#
# ── Counterfactual / Potential Outcomes Models (v0.4.0) ───────────────────────
#   Implements the Potential Outcomes (Rubin) framework for time-series:
#   ITE = Y(1) − Y(0);  ATE = E[ITE];  time-varying ATE = E[Y_t(ā) − Y_t(ā')]
#
#   • DeepSynth            — Neural Synthetic Control: two-layer GRU
#                            SharedEncoder maps covariate history to a latent
#                            query; donor variables (all except treatment/outcome
#                            columns) are encoded as keys/values; scaled dot-
#                            product attention computes soft donor weights; a
#                            factual head (query + T_{t+1}) and a counterfactual
#                            head (attended donor summary) yield ŷ_fact and
#                            ŷ_counter; ITE = ŷ_fact − ŷ_counter.
#                            Public API: deep_synth_model() / DeepSynthModel().
#   • CRN                  — Counterfactual Recurrent Network: two-layer GRU
#                            encodes [X, T_hist, 0] → LayerNorm + Tanh rep r;
#                            adversarial treatment discriminator minimises
#                            BCE(D(r), T) to balance representations; decoder
#                            conditioned on r and do(T ∈ {0,1}) produces
#                            Ŷ(0), Ŷ(1), and ITE; total loss =
#                            MSE − λ_adv · BCE(D(r), T).
#                            Public API: crn_model() / CRNModel().
#   • G-Net                — Deep G-Computation: two-layer GRU backbone encodes
#                            [X, T_hist]; covariate transition head predicts
#                            X̂_{t+1} | do(T); outcome head predicts
#                            Ŷ_{t+1}(ā) | X̂_{t+1}, do(T); counterfactuals
#                            Ŷ(0) and Ŷ(1) computed by substituting T=0/1 in
#                            both heads (sequential G-formula substitution).
#                            Public API: gnet_model() / GNetModel().
#   Unified entry point:   counterfactual_model() / CounterfactualModel()
#   ATE / ITE accessors:   ate_counterfactual(), ite_counterfactual()
#   Prediction:            predict.counterfactual_model()
#
# ── References ────────────────────────────────────────────────────────────────
#   Louizos et al. (2017). Causal Effect Inference with Deep Latent-Variable
#     Models. NeurIPS 30.
#     https://papers.nips.cc/paper/7223
#   Shalit et al. (2017). Estimating individual treatment effect: generalization
#     bounds and algorithms (TARNet/CFRNet). ICML.
#     https://arxiv.org/abs/1606.03976
#   Yoon et al. (2018). GANITE: Estimation of Individualized Treatment Effects
#     using Generative Adversarial Nets. ICLR.
#     https://openreview.net/forum?id=ByKWUeWA-
#   Nauta et al. (2019). Causal Discovery with Attention-Based Convolutional
#     Neural Networks. MDPI MAKE 1(1), 312–340.
#     https://doi.org/10.3390/make1010019
#   Choi et al. (2016). RETAIN: Interpretable Predictive Model for Healthcare
#     via Reverse Time Attention. NeurIPS 29.
#     https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html
#   Tank et al. (2021). Neural Granger Causality. IEEE TPAMI 44(8), 4267–4279.
#     https://doi.org/10.1109/TPAMI.2021.3065601
#   Lim et al. (2021). Temporal Fusion Transformers for Interpretable
#     Multi-Horizon Time Series Forecasting. IJF 37(4), 1748–1764.
#     https://doi.org/10.1016/j.ijforecast.2021.03.012
#   Zheng et al. (2018). DAGs with NOTEARS: Continuous Optimization for
#     Structure Learning. NeurIPS.
#     https://arxiv.org/abs/1803.01422
#   Brouwer et al. (2023). CUTS: Neural Causal Discovery from Irregular
#     Time-Series Data. ICLR 2023.
#     https://arxiv.org/abs/2302.05925
#   Bica et al. (2020). Estimating counterfactual treatment outcomes over time
#     through adversarially balanced representations (CRN). ICLR 2020.
#     https://arxiv.org/abs/2002.04083
#   Li & van der Schaar (2021). G-Net: a recurrent network approach to
#     G-computation for counterfactual prediction under a dynamic treatment
#     regime. ML4H @ NeurIPS 2021.
#     https://arxiv.org/abs/2110.10996
#   Abadie & Gardeazabal (2003). The economic costs of conflict (Synthetic
#     Control). Am. Econ. Rev. 93(1), 113–132.
#     https://doi.org/10.1257/000282803321455188

# --- CEVAE torch helpers (used when torch is available) ---
.cevae_fc <- function(sizes, final_activation = NULL) {
  layers <- list()
  n <- length(sizes)
  for (i in seq_len(n - 1L)) {
    layers[[length(layers) + 1L]] <- torch::nn_linear(sizes[i], sizes[i + 1L])
    if (i < n - 1L) layers[[length(layers) + 1L]] <- torch::nn_elu()
  }
  if (!is.null(final_activation)) layers[[length(layers) + 1L]] <- final_activation
  do.call(torch::nn_sequential, layers)
}

.cevae_diag_normal_net <- function(sizes) {
  dim_out <- sizes[length(sizes)]
  torch::nn_module(
    initialize = function() {
      self$fc <- .cevae_fc(c(sizes[-length(sizes)], as.integer(dim_out * 2L)))
    },
    forward = function(x) {
      loc_scale <- self$fc(x)
      loc <- loc_scale[ , 1:dim_out, drop = FALSE]$clamp(min = -1e2, max = 1e2)
      scale <- torch::nnf_softplus(loc_scale[ , (dim_out + 1L):(dim_out * 2L), drop = FALSE])$add(1e-3)$clamp(max = 1e2)
      list(loc = loc, scale = scale)
    }
  )
}

.cevae_bernoulli_net <- function(sizes) {
  torch::nn_module(
    initialize = function() {
      self$fc <- .cevae_fc(c(sizes, 1L))
    },
    forward = function(x) {
      logits <- self$fc(x)$squeeze(-1)$clamp(min = -10, max = 10)
      logits
    }
  )
}

.cevae_normal_outcome_net <- function(sizes) {
  torch::nn_module(
    initialize = function() {
      self$fc <- .cevae_fc(c(sizes, 2L))
    },
    forward = function(x) {
      loc_scale <- self$fc(x)
      loc <- loc_scale[ , 1, drop = FALSE]$clamp(min = -1e6, max = 1e6)
      scale <- torch::nnf_softplus(loc_scale[ , 2, drop = FALSE])$clamp(min = 1e-3, max = 1e6)
      list(loc = loc, scale = scale)
    }
  )
}

.cevae_model_torch <- function(config) {
  latent_dim <- config$latent_dim
  feature_dim <- config$feature_dim
  hidden_dim <- config$hidden_dim
  num_layers <- config$num_layers
  h_sizes <- c(latent_dim, rep(hidden_dim, num_layers))
  torch::nn_module(
    "CEVAEModel",
    initialize = function() {
      self$x_nn <- .cevae_diag_normal_net(c(h_sizes, feature_dim))()
      self$t_nn <- .cevae_bernoulli_net(latent_dim)()
      self$y0_nn <- .cevae_normal_outcome_net(h_sizes)()
      self$y1_nn <- .cevae_normal_outcome_net(h_sizes)()
    },
    z_prior = function(batch_size) {
      list(loc = torch::torch_zeros(c(batch_size, latent_dim)),
           scale = torch::torch_ones(c(batch_size, latent_dim)))
    },
    x_dist = function(z) self$x_nn(z),
    t_dist = function(z) self$t_nn(z),
    y_dist_params = function(t, z) {
      p0 <- self$y0_nn(z)
      p1 <- self$y1_nn(z)
      t_bool <- t$unsqueeze(-1)$expand_as(p0$loc)
      list(loc = torch::torch_where(t_bool$gt(0.5), p1$loc, p0$loc),
           scale = torch::torch_where(t_bool$gt(0.5), p1$scale, p0$scale))
    },
    y_mean = function(z, t) {
      params <- self$y_dist_params(t, z)
      params$loc
    }
  )
}

.cevae_guide_torch <- function(config) {
  latent_dim <- config$latent_dim
  feature_dim <- config$feature_dim
  hidden_dim <- config$hidden_dim
  num_layers <- config$num_layers
  torch::nn_module(
    "CEVAEGuide",
    initialize = function() {
      self$t_nn <- .cevae_bernoulli_net(feature_dim)()
      y_sizes <- c(feature_dim, rep(hidden_dim, max(1L, num_layers - 1L)))
      self$y_shared <- .cevae_fc(c(y_sizes, hidden_dim), torch::nn_elu())
      self$y0_nn <- .cevae_normal_outcome_net(hidden_dim)()
      self$y1_nn <- .cevae_normal_outcome_net(hidden_dim)()
      z_in <- 1L + feature_dim
      z_sizes <- c(z_in, rep(hidden_dim, max(1L, num_layers - 1L)))
      self$z_shared <- .cevae_fc(c(z_sizes, hidden_dim), torch::nn_elu())
      self$z0_nn <- .cevae_diag_normal_net(c(hidden_dim, latent_dim))()
      self$z1_nn <- .cevae_diag_normal_net(c(hidden_dim, latent_dim))()
    },
    t_dist = function(x) self$t_nn(x),
    y_dist_params = function(t, x) {
      hidden <- self$y_shared(x)
      p0 <- self$y0_nn(hidden)
      p1 <- self$y1_nn(hidden)
      t_bool <- t$unsqueeze(-1)$expand_as(p0$loc)
      list(loc = torch::torch_where(t_bool$gt(0.5), p1$loc, p0$loc),
           scale = torch::torch_where(t_bool$gt(0.5), p1$scale, p0$scale))
    },
    z_dist = function(y, t, x) {
      y_2d <- y$view(c(y$size(1), 1L))
      y_x <- torch::torch_cat(list(y_2d, x), dim = 2L)
      hidden <- self$z_shared(y_x)
      p0 <- self$z0_nn(hidden)
      p1 <- self$z1_nn(hidden)
      t_bool <- t$unsqueeze(-1)$expand_as(p0$loc)
      list(loc = torch::torch_where(t_bool$gt(0.5), p1$loc, p0$loc),
           scale = torch::torch_where(t_bool$gt(0.5), p1$scale, p0$scale))
    }
  )
}

.cevae_whiten <- function(x) {
  loc <- torch::torch_mean(x, dim = 1L, keepdim = TRUE)
  scale <- torch::torch_std(x, dim = 1L, unbiased = FALSE, keepdim = TRUE)
  scale <- torch::torch_where(scale > 0, scale, torch::torch_ones_like(scale))
  list(loc = loc$squeeze(1), inv_scale = (1 / scale)$squeeze(1))
}

.cevae_whiten_apply <- function(x, whiten) {
  (x - whiten$loc) * whiten$inv_scale
}

.cevae_loss_step <- function(model, guide, x, t, y, whiten) {
  x_w <- .cevae_whiten_apply(x, whiten)
  batch <- x$size(1)
  t_logits <- guide$t_dist(x_w)
  t_probs <- torch::nnf_sigmoid(t_logits)$clamp(1e-6, 1 - 1e-6)
  t_samp <- torch::torch_bernoulli(t_probs)
  y_params <- guide$y_dist_params(t_samp, x_w)
  y_samp <- y_params$loc + y_params$scale * torch::torch_randn_like(y_params$loc)
  z_params <- guide$z_dist(y_samp, t_samp, x_w)
  z_samp <- z_params$loc + z_params$scale * torch::torch_randn_like(z_params$loc)
  # z_prior must be on same device as z_samp (model$z_prior() creates on CPU by default)
  dev <- z_samp$device
  latent_dim <- z_samp$size(2)
  z_prior <- list(
    loc = torch::torch_zeros(c(batch, latent_dim), dtype = z_samp$dtype, device = dev),
    scale = torch::torch_ones(c(batch, latent_dim), dtype = z_samp$dtype, device = dev)
  )
  log_p_z <- torch::distr_normal(z_prior$loc, z_prior$scale)$log_prob(z_samp)$sum(2L)$mean()
  log_q_z <- torch::distr_normal(z_params$loc, z_params$scale)$log_prob(z_samp)$sum(2L)$mean()
  x_params <- model$x_dist(z_samp)
  log_p_x <- torch::distr_normal(x_params$loc, x_params$scale)$log_prob(x_w)$sum(2L)$mean()
  t_probs_p <- torch::nnf_sigmoid(model$t_dist(z_samp))$clamp(1e-6, 1 - 1e-6)
  log_p_t <- (t_samp * torch::torch_log(t_probs_p) + (1 - t_samp) * torch::torch_log(1 - t_probs_p))$mean()
  y_params_p <- model$y_dist_params(t_samp, z_samp)
  log_p_y <- torch::distr_normal(y_params_p$loc, y_params_p$scale)$log_prob(y)$squeeze(-1)$mean()
  log_q_t <- (t * torch::torch_log(t_probs) + (1 - t) * torch::torch_log(1 - t_probs))$mean()
  log_q_y <- torch::distr_normal(y_params$loc, y_params$scale)$log_prob(y)$squeeze(-1)$mean()
  elbo <- log_p_z + log_p_x + log_p_t + log_p_y - log_q_z
  (-elbo - log_q_t - log_q_y)
}

#' CEVAE (Counterfactual Variational Autoencoder)
#'
#' Generative model: z ~ p(z), x ~ p(x|z), w ~ p(w|z), y ~ p(y|t,z). Twin outcome nets allow imbalanced treatment.
#' Training: ELBO + log q(t|x) + log q(y|t,x). When \pkg{torch} is available the full VAE is fitted; else nnet/ranger placeholder.
#'
#' @param X Covariate matrix or data.frame.
#' @param treatment Binary treatment 0/1.
#' @param y Outcome vector.
#' @param outcome_dist Outcome distribution: \code{"normal"} (default) or \code{"bernoulli"}.
#' @param latent_dim Dimension of latent confounder (default 20).
#' @param hidden_dim Hidden layer size (default 200).
#' @param num_epochs Number of training epochs (default 50).
#' @param num_layers Number of hidden layers (default 3).
#' @param batch_size Batch size (default 100).
#' @param learning_rate Learning rate (default 1e-3).
#' @param learning_rate_decay Final lr = lr * this (default 0.1).
#' @param num_samples Number of samples for ITE prediction (default 1000).
#' @param weight_decay Weight decay (default 1e-4).
#' @param verbose If TRUE, print epoch loss.
#' @param device Device for torch: \code{"cuda"}, \code{"cpu"}, or \code{NULL} (default: use CUDA if available, else CPU).
#' @param ... Ignored.
#' @return Object of class \code{cevae} with \code{predict()} for ITE.
#' @references Louizos et al. (2017). Causal Effect Inference with Deep Latent-Variable Models. NeurIPS.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # cevae(...)
#' }
#' @export
cevae <- function(X, treatment, y,
                  outcome_dist = "normal",
                  latent_dim = 20L,
                  hidden_dim = 200L,
                  num_epochs = 50L,
                  num_layers = 3L,
                  batch_size = 100L,
                  learning_rate = 1e-3,
                  learning_rate_decay = 0.1,
                  num_samples = 1000L,
                  weight_decay = 1e-4,
                  verbose = TRUE,
                  device = NULL,
                  ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  X <- as.matrix(X)
  w <- as.integer(treatment)
  y <- as.numeric(y)
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(p))

  if (requireNamespace("torch", quietly = TRUE)) {
    if (is.null(device))
      device <- if (torch::cuda_is_available()) "cuda" else "cpu"
    config <- list(
      feature_dim = p,
      latent_dim = as.integer(latent_dim),
      hidden_dim = as.integer(hidden_dim),
      num_layers = as.integer(num_layers),
      outcome_dist = outcome_dist
    )
    model <- .cevae_model_torch(config)()
    guide <- .cevae_guide_torch(config)()
    model$to(device = device)
    guide$to(device = device)
    X_t <- torch::torch_tensor(X, dtype = torch::torch_float32(), device = device)
    t_t <- torch::torch_tensor(w, dtype = torch::torch_float32(), device = device)
    y_t <- torch::torch_tensor(matrix(y, ncol = 1L), dtype = torch::torch_float32(), device = device)
    whiten <- .cevae_whiten(X_t)
    params <- c(model$parameters, guide$parameters)
    num_steps <- num_epochs * max(1L, ceiling(n / batch_size))
    opt <- torch::optim_adam(params, lr = learning_rate, weight_decay = weight_decay)
    batch_size <- min(batch_size, n)
    n_batches <- max(1L, ceiling(n / batch_size))

    for (epoch in seq_len(num_epochs)) {
      model$train(TRUE)
      guide$train(TRUE)
      epoch_loss <- 0
      perm <- sample(n)
      for (start in seq(1L, n, by = batch_size)) {
        idx <- perm[start:min(start + batch_size - 1L, n)]
        opt$zero_grad()
        loss <- .cevae_loss_step(model, guide,
          X_t[idx, , drop = FALSE], t_t[idx], y_t[idx, , drop = FALSE], whiten)
        loss$backward()
        opt$step()
        epoch_loss <- epoch_loss + as.numeric(loss)
      }
      if (verbose && epoch %% 10L == 0L)
        message("CEVAE epoch ", epoch, " loss: ", round(epoch_loss / n_batches, 4))
    }

    return(structure(
      list(model = model, guide = guide, whiten = whiten,
           X_names = colnames(X), type = "cevae_torch",
           feature_dim = p, num_samples = num_samples, batch_size = batch_size,
           device = device),
      class = "cevae"
    ))
  }

  if (requireNamespace("nnet", quietly = TRUE)) {
    df0 <- as.data.frame(X[w == 0, , drop = FALSE]); df0$y <- y[w == 0]
    df1 <- as.data.frame(X[w == 1, , drop = FALSE]); df1$y <- y[w == 1]
    size <- max(5, min(20, ncol(X)))
    m0 <- nnet::nnet(y ~ ., data = df0, size = size, linout = TRUE, trace = FALSE, maxit = 200)
    m1 <- nnet::nnet(y ~ ., data = df1, size = size, linout = TRUE, trace = FALSE, maxit = 200)
    structure(list(model_0 = m0, model_1 = m1, X_names = colnames(X), type = "cevae_nnet_placeholder"),
              class = "cevae")
  } else {
    message("CEVAE: Install package 'torch' for full CEVAE. Install 'nnet' for placeholder. Using ranger.")
    df0 <- as.data.frame(X[w == 0, , drop = FALSE]); df0$y <- y[w == 0]
    df1 <- as.data.frame(X[w == 1, , drop = FALSE]); df1$y <- y[w == 1]
    m0 <- ranger::ranger(y ~ ., data = df0)
    m1 <- ranger::ranger(y ~ ., data = df1)
    structure(list(model_0 = m0, model_1 = m1, X_names = colnames(X), type = "cevae_ranger_fallback"),
              class = "cevae")
  }
}

#' Predict ITE from CEVAE (or placeholder)
#'
#' For torch CEVAE: samples z from guide given x, then ITE = mean over samples of E[y|z,t=1] - E[y|z,t=0].
#' When \code{num_samples = 1}, a fast deterministic path is used (posterior mean for z, single forward pass).
#' @param object Fitted \code{cevae} object.
#' @param newdata Covariate matrix or data.frame.
#' @param num_samples Number of MC samples for ITE (torch only). Use \code{1} for fast deterministic prediction.
#' @param ... Ignored.
#' @return Vector of predicted ITE.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.cevae(...)
#' }
#' @export
predict.cevae <- function(object, newdata, num_samples = NULL, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names

  if (identical(object$type, "cevae_torch")) {
    n_samp <- if (!is.null(num_samples)) num_samples else object$num_samples
    n_samp <- max(1L, as.integer(n_samp))
    batch_size <- object$batch_size
    model <- object$model
    guide <- object$guide
    whiten <- object$whiten
    dev <- if (!is.null(object$device)) object$device else "cpu"
    model$eval()
    guide$eval()
    torch::with_no_grad({
      x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = dev)
      x_w <- .cevae_whiten_apply(x_t, whiten)
      n_obs <- x_t$size(1)
      ite_list <- list()
      for (start in seq(1L, n_obs, by = batch_size)) {
        end <- min(start + batch_size - 1L, n_obs)
        x_b <- x_w[start:end, , drop = FALSE]
        batch_len <- x_b$size(1)
        y0_sum <- torch::torch_zeros(batch_len, dtype = torch::torch_float32(), device = dev)
        y1_sum <- torch::torch_zeros(batch_len, dtype = torch::torch_float32(), device = dev)
        if (n_samp == 1L) {
          # Fast deterministic path: use posterior mean for z (no sampling), single forward pass
          t_logits <- guide$t_dist(x_b)
          t_probs <- torch::nnf_sigmoid(t_logits)$clamp(1e-6, 1 - 1e-6)
          t_samp <- torch::torch_bernoulli(t_probs)
          y_params <- guide$y_dist_params(t_samp, x_b)
          y_samp <- y_params$loc
          z_params <- guide$z_dist(y_samp, t_samp, x_b)
          z_samp <- z_params$loc
          t0 <- torch::torch_zeros(batch_len, dtype = torch::torch_float32(), device = dev)
          t1 <- torch::torch_ones(batch_len, dtype = torch::torch_float32(), device = dev)
          y0_sum <- model$y_mean(z_samp, t0)$squeeze(2)
          y1_sum <- model$y_mean(z_samp, t1)$squeeze(2)
          ite_b <- y1_sum - y0_sum
        } else {
          for (k in seq_len(n_samp)) {
            t_logits <- guide$t_dist(x_b)
            t_probs <- torch::nnf_sigmoid(t_logits)$clamp(1e-6, 1 - 1e-6)
            t_samp <- torch::torch_bernoulli(t_probs)
            y_params <- guide$y_dist_params(t_samp, x_b)
            y_samp <- y_params$loc + y_params$scale * torch::torch_randn_like(y_params$loc)
            z_params <- guide$z_dist(y_samp, t_samp, x_b)
            z_samp <- z_params$loc + z_params$scale * torch::torch_randn_like(z_params$loc)
            t0 <- torch::torch_zeros(batch_len, dtype = torch::torch_float32(), device = dev)
            t1 <- torch::torch_ones(batch_len, dtype = torch::torch_float32(), device = dev)
            y0_sum <- y0_sum + model$y_mean(z_samp, t0)$squeeze(2)
            y1_sum <- y1_sum + model$y_mean(z_samp, t1)$squeeze(2)
          }
          ite_b <- (y1_sum - y0_sum) / n_samp
        }
        ite_list[[length(ite_list) + 1L]] <- ite_b
      }
      ite <- torch::torch_cat(ite_list)
    })
    return(as.numeric(ite))
  }

  df <- as.data.frame(newdata)
  if (inherits(object$model_0, "nnet")) {
    p1 <- predict(object$model_1, newdata = df)
    p0 <- predict(object$model_0, newdata = df)
  } else {
    p1 <- predict(object$model_1, data = df)$predictions
    p0 <- predict(object$model_0, data = df)$predictions
  }
  as.vector(p1) - as.vector(p0)
}

#' Fit CEVAE and predict ITE (fit_predict)
#'
#' Convenience to fit and predict in one call, matching Python \code{CEVAE.fit_predict(X, treatment, y)}.
#' @param X Covariate matrix or data.frame.
#' @param treatment Binary treatment 0/1.
#' @param y Outcome vector.
#' @param ... Passed to \code{\link{cevae}}.
#' @return Vector of predicted ITE (same length as \code{nrow(X)}).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit_predict_cevae(...)
#' }
#' @export
fit_predict_cevae <- function(X, treatment, y, ...) {
  obj <- cevae(X, treatment, y, ...)
  predict(obj, X)
}

# --- DragonNet (torch) helpers: module, loss, training ---
# Only used when torch is available; see dragonnet() below.

.dragonnet_module_torch <- function(input_dim, neurons = 200L, reg_l2 = 0.01) {
  half <- as.integer(neurons / 2)
  torch::nn_module(
    "DragonNet",
    initialize = function() {
      self$repr_l1 <- torch::nn_linear(input_dim, neurons)
      self$repr_l2 <- torch::nn_linear(neurons, neurons)
      self$repr_l3 <- torch::nn_linear(neurons, neurons)
      self$propensity_head <- torch::nn_linear(neurons, 1L)
      self$y0_h1 <- torch::nn_linear(neurons, half)
      self$y0_h2 <- torch::nn_linear(half, half)
      self$y0_out <- torch::nn_linear(half, 1L)
      self$y1_h1 <- torch::nn_linear(neurons, half)
      self$y1_h2 <- torch::nn_linear(half, half)
      self$y1_out <- torch::nn_linear(half, 1L)
      self$epsilon <- torch::nn_parameter(torch::torch_zeros(1L))
    },
    forward = function(x) {
      phi <- torch::nnf_elu(self$repr_l3(torch::nnf_elu(self$repr_l2(torch::nnf_elu(self$repr_l1(x))))))
      propensity_logits <- self$propensity_head(phi)$squeeze(2L)
      e <- torch::nnf_sigmoid(propensity_logits)
      y0_h <- torch::nnf_elu(self$y0_h2(torch::nnf_elu(self$y0_h1(phi))))
      y1_h <- torch::nnf_elu(self$y1_h2(torch::nnf_elu(self$y1_h1(phi))))
      y0 <- self$y0_out(y0_h)$squeeze(2L)
      y1 <- self$y1_out(y1_h)$squeeze(2L)
      list(y0 = y0, y1 = y1, propensity = e, propensity_logits = propensity_logits, epsilon = self$epsilon)
    }
  )
}

.dragonnet_loss_torch <- function(y0, y1, propensity, propensity_logits, epsilon, t, y, ratio_tar = 1, clip_eps = 1e-5) {
  propensity_logits <- propensity_logits$squeeze()
  t_ <- t$squeeze()
  e <- propensity$clamp(min = clip_eps, max = 1 - clip_eps)
  y_hat <- (1 - t_) * y0 + t_ * y1
  y_ <- y$squeeze()
  reg_loss <- torch::nnf_mse_loss(y_hat, y_)
  bce_loss <- torch::nnf_binary_cross_entropy_with_logits(propensity_logits, t_)
  base_loss <- reg_loss + bce_loss
  inv_ps_weight <- t_ / e - (1 - t_) / (1 - e)
  y_corrected <- y_hat + epsilon * inv_ps_weight
  tar_loss <- torch::nnf_mse_loss(y_corrected, y_)
  base_loss + ratio_tar * tar_loss
}

#' DragonNet — shared representation and three heads (torch)
#'
#' DragonNet uses a shared representation and three heads: propensity \eqn{\hat{e}(X)}, \eqn{\hat{Y}(0)}, and \eqn{\hat{Y}(1)},
#' with optional targeted regularization (learnable \eqn{\varepsilon}). Architecture follows
#' [causalml dragonnet.py](https://github.com/uber/causalml/blob/master/causalml/inference/tf/dragonnet.py):
#' 3 shared layers (ELU), propensity head (sigmoid), and two outcome branches (2 hidden layers each, then output).
#'
#' When \pkg{torch} is installed, the full DragonNet is fitted (Adam then SGD). Otherwise falls back to an nnet or ranger placeholder.
#'
#' @param X covariate matrix or data.frame
#' @param treatment binary treatment 0/1
#' @param y outcome vector
#' @param neurons units per shared layer (default 200)
#' @param reg_l2 L2 regularization for outcome heads (default 0.01)
#' @param targeted_reg if TRUE (default), use targeted regularization in the loss
#' @param ratio_tar weight for targeted regularization term (default 1)
#' @param batch_size batch size for training (default 64)
#' @param val_split fraction for validation (default 0.2)
#' @param adam_epochs epochs for Adam phase (default 30)
#' @param adam_lr Adam learning rate (default 1e-3)
#' @param sgd_epochs epochs for SGD phase (default 100)
#' @param sgd_lr SGD learning rate (default 1e-5)
#' @param sgd_momentum SGD momentum (default 0.9)
#' @param verbose if TRUE, print epoch loss
#' @param ... ignored
#' @return Object of class \code{dragonnet} (torch model when available, else placeholder) with \code{predict()} for ITE and optional \code{propensity}.
#' @references Shi, C., Blei, D., Veitch, V. (2019). Adapting Neural Networks for the Estimation of Treatment Effects. \url{https://arxiv.org/abs/1906.02120}.
#' @references \url{https://github.com/uber/causalml/blob/master/causalml/inference/tf/dragonnet.py}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # dragonnet(...)
#' }
#' @export
dragonnet <- function(X, treatment, y,
                      neurons = 200L,
                      reg_l2 = 0.01,
                      targeted_reg = TRUE,
                      ratio_tar = 1,
                      batch_size = 64L,
                      val_split = 0.2,
                      adam_epochs = 30L,
                      adam_lr = 1e-3,
                      sgd_epochs = 100L,
                      sgd_lr = 1e-5,
                      sgd_momentum = 0.9,
                      verbose = TRUE,
                      ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  X <- as.matrix(X)
  w <- as.integer(treatment)
  y <- as.numeric(y)
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(p))

  if (requireNamespace("torch", quietly = TRUE)) {
    # Build module and tensors
    Module <- .dragonnet_module_torch(as.integer(p), as.integer(neurons), reg_l2)
    model <- Module()
    X_t <- torch::torch_tensor(X, dtype = torch::torch_float32())
    t_t <- torch::torch_tensor(matrix(w, ncol = 1L), dtype = torch::torch_float32())
    y_t <- torch::torch_tensor(matrix(y, ncol = 1L), dtype = torch::torch_float32())

    # Train/val split
    n_val <- max(1L, round(n * val_split))
    n_train <- n - n_val
    perm <- sample(n)
    idx_train <- perm[seq_len(n_train)]
    idx_val <- perm[seq(n_train + 1L, n)]

    # Adam
    opt_adam <- torch::optim_adam(model$parameters, lr = adam_lr)
    batch_size <- min(batch_size, n_train)
    n_batches <- max(1L, ceiling(n_train / batch_size))

    for (epoch in seq_len(adam_epochs)) {
      model$train(TRUE)
      batch_loss <- 0
      shuffle <- sample(idx_train)
      for (start in seq(1L, n_train, by = batch_size)) {
        idx <- shuffle[start:min(start + batch_size - 1L, n_train)]
        opt_adam$zero_grad()
        out <- model(X_t[idx, , drop = FALSE])
        loss <- .dragonnet_loss_torch(
          out$y0, out$y1, out$propensity, out$propensity_logits, out$epsilon,
          t_t[idx, , drop = FALSE], y_t[idx, , drop = FALSE],
          ratio_tar = if (targeted_reg) ratio_tar else 0
        )
        loss$backward()
        torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 5)
        opt_adam$step()
        batch_loss <- batch_loss + as.numeric(loss)
      }
      if (verbose && epoch %% 10L == 0L)
        message("DragonNet Adam epoch ", epoch, " loss: ", round(batch_loss / n_batches, 4))
    }

    # SGD with momentum
    opt_sgd <- torch::optim_sgd(model$parameters, lr = sgd_lr, momentum = sgd_momentum)
    for (epoch in seq_len(sgd_epochs)) {
      model$train(TRUE)
      batch_loss <- 0
      shuffle <- sample(idx_train)
      for (start in seq(1L, n_train, by = batch_size)) {
        idx <- shuffle[start:min(start + batch_size - 1L, n_train)]
        opt_sgd$zero_grad()
        out <- model(X_t[idx, , drop = FALSE])
        loss <- .dragonnet_loss_torch(
          out$y0, out$y1, out$propensity, out$propensity_logits, out$epsilon,
          t_t[idx, , drop = FALSE], y_t[idx, , drop = FALSE],
          ratio_tar = if (targeted_reg) ratio_tar else 0
        )
        loss$backward()
        torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 5)
        opt_sgd$step()
        batch_loss <- batch_loss + as.numeric(loss)
      }
      if (verbose && epoch %% 25L == 0L)
        message("DragonNet SGD epoch ", epoch, " loss: ", round(batch_loss / n_batches, 4))
    }

    return(structure(
      list(model = model, X_names = colnames(X), type = "dragonnet_torch",
           input_dim = p, neurons = neurons),
      class = "dragonnet"
    ))
  }

  # Fallback: nnet or ranger placeholder
  if (requireNamespace("nnet", quietly = TRUE)) {
    df <- as.data.frame(X)
    df$w <- w
    df$y <- y
    size <- max(5, min(20, ncol(X)))
    m <- nnet::nnet(y ~ ., data = df, size = size, linout = TRUE, trace = FALSE, maxit = 200)
    return(structure(list(model = m, X_names = colnames(X), type = "dragonnet_nnet_placeholder"),
                    class = "dragonnet"))
  }
  message("DragonNet: Install package 'torch' for the full DragonNet. Using ranger fallback.")
  df0 <- as.data.frame(X[w == 0, , drop = FALSE]); df0$y <- y[w == 0]
  df1 <- as.data.frame(X[w == 1, , drop = FALSE]); df1$y <- y[w == 1]
  m0 <- ranger::ranger(y ~ ., data = df0)
  m1 <- ranger::ranger(y ~ ., data = df1)
  structure(list(model_0 = m0, model_1 = m1, X_names = colnames(X), type = "dragonnet_ranger_fallback"),
            class = "dragonnet")
}

#' Predict CATE (and optionally propensity) from DragonNet
#'
#' For torch DragonNet: \eqn{\widehat{ITE}(x) = \hat{Y}(1) - \hat{Y}(0)}. For placeholder models, returns difference of predicted outcomes.
#'
#' @param object fitted \code{dragonnet} object
#' @param newdata covariate matrix or data.frame
#' @param propensity if TRUE and the model supports it, include propensity in the result (torch only)
#' @param ... ignored
#' @return Vector of predicted ITE. If \code{propensity = TRUE} and model is torch, a list with \code{ite} and \code{propensity}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.dragonnet(...)
#' }
#' @export
predict.dragonnet <- function(object, newdata, propensity = FALSE, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names

  if (identical(object$type, "dragonnet_torch")) {
    object$model$eval()
    torch::with_no_grad({
      x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32())
      out <- object$model(x_t)
      y0 <- as.numeric(out$y0)
      y1 <- as.numeric(out$y1)
      ite <- y1 - y0
      if (propensity) {
        prop <- as.numeric(out$propensity)
        return(list(ite = ite, propensity = prop))
      }
      return(ite)
    })
  }

  df <- as.data.frame(newdata)
  if (!is.null(object$model) && inherits(object$model, "nnet")) {
    df0 <- df; df0$w <- 0
    df1 <- df; df1$w <- 1
    p0 <- predict(object$model, newdata = df0)
    p1 <- predict(object$model, newdata = df1)
    return(as.vector(p1) - as.vector(p0))
  }
  p0 <- predict(object$model_0, data = df)$predictions
  p1 <- predict(object$model_1, data = df)$predictions
  as.vector(p1) - as.vector(p0)
}

# --- TARNet (Treatment-Agnostic Representation Network) ---
# Shared representation Phi(X) and two heads for Y(0) and Y(1). Ref: https://github.com/arnaudscott/TARNet

.tarnet_module_torch <- function(input_dim, rep_dim = 100L, hidden = c(200L, 200L, 100L)) {
  hidden <- as.integer(hidden)
  torch::nn_module(
    "TARNet",
    initialize = function() {
      layers <- list()
      prev <- input_dim
      for (h in hidden) {
        layers[[length(layers) + 1L]] <- torch::nn_linear(prev, h)
        layers[[length(layers) + 1L]] <- torch::nn_elu()
        prev <- h
      }
      self$repr <- do.call(torch::nn_sequential, layers)
      self$head0 <- torch::nn_linear(prev, 1L)
      self$head1 <- torch::nn_linear(prev, 1L)
    },
    forward = function(x) {
      phi <- self$repr(x)
      list(y0 = self$head0(phi)$squeeze(2L), y1 = self$head1(phi)$squeeze(2L), phi = phi)
    }
  )
}

.tarnet_loss_torch <- function(y0, y1, t, y) {
  t_ <- t$squeeze()
  y_ <- y$squeeze()
  y_hat <- (1 - t_) * y0 + t_ * y1
  torch::nnf_mse_loss(y_hat, y_)
}

#' TARNet — Treatment-Agnostic Representation Network
#'
#' TARNet uses a shared representation \eqn{\Phi(X)} and two heads for \eqn{\hat{Y}(0)} and \eqn{\hat{Y}(1)}.
#' Architecture follows \href{https://github.com/arnaudscott/TARNet}{TARNet}: MLP representation (ELU), then treatment-specific outputs.
#'
#' When \pkg{torch} is installed, the full TARNet is fitted. Otherwise falls back to an nnet or ranger placeholder.
#'
#' @param X covariate matrix or data.frame
#' @param treatment binary treatment 0/1
#' @param y outcome vector
#' @param rep_dim deprecated; use \code{hidden} for layer sizes
#' @param hidden vector of hidden layer sizes (default \code{c(200, 200, 100)})
#' @param batch_size batch size for training (default 64)
#' @param val_split fraction for validation (default 0.2)
#' @param epochs number of training epochs (default 100)
#' @param lr learning rate (default 1e-3)
#' @param verbose if TRUE, print epoch loss
#' @param ... ignored
#' @return Object of class \code{tarnet} with \code{predict()} for ITE.
#' @references \url{https://github.com/arnaudscott/TARNet}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # tarnet(...)
#' }
#' @export
tarnet <- function(X, treatment, y,
                   rep_dim = 100L,
                   hidden = c(200L, 200L, 100L),
                   batch_size = 64L,
                   val_split = 0.2,
                   epochs = 100L,
                   lr = 1e-3,
                   verbose = TRUE,
                   ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  X <- as.matrix(X)
  w <- as.integer(treatment)
  y <- as.numeric(y)
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(p))

  if (requireNamespace("torch", quietly = TRUE)) {
    Module <- .tarnet_module_torch(as.integer(p), rep_dim = rep_dim, hidden = hidden)
    model <- Module()
    X_t <- torch::torch_tensor(X, dtype = torch::torch_float32())
    t_t <- torch::torch_tensor(matrix(w, ncol = 1L), dtype = torch::torch_float32())
    y_t <- torch::torch_tensor(matrix(y, ncol = 1L), dtype = torch::torch_float32())

    n_val <- max(1L, round(n * val_split))
    n_train <- n - n_val
    perm <- sample(n)
    idx_train <- perm[seq_len(n_train)]

    opt <- torch::optim_adam(model$parameters, lr = lr)
    batch_size <- min(batch_size, n_train)
    n_batches <- max(1L, ceiling(n_train / batch_size))

    for (epoch in seq_len(epochs)) {
      model$train(TRUE)
      batch_loss <- 0
      shuffle <- sample(idx_train)
      for (start in seq(1L, n_train, by = batch_size)) {
        idx <- shuffle[start:min(start + batch_size - 1L, n_train)]
        opt$zero_grad()
        out <- model(X_t[idx, , drop = FALSE])
        loss <- .tarnet_loss_torch(out$y0, out$y1, t_t[idx, , drop = FALSE], y_t[idx, , drop = FALSE])
        loss$backward()
        opt$step()
        batch_loss <- batch_loss + as.numeric(loss)
      }
      if (verbose && epoch %% 10L == 0L)
        message("TARNet epoch ", epoch, " loss: ", round(batch_loss / n_batches, 4))
    }

    return(structure(
      list(model = model, X_names = colnames(X), type = "tarnet_torch", input_dim = p),
      class = "tarnet"
    ))
  }

  if (requireNamespace("nnet", quietly = TRUE)) {
    df <- as.data.frame(X)
    df$w <- w
    df$y <- y
    size <- max(5, min(20, ncol(X)))
    m <- nnet::nnet(y ~ ., data = df, size = size, linout = TRUE, trace = FALSE, maxit = 200)
    return(structure(list(model = m, X_names = colnames(X), type = "tarnet_nnet_placeholder"), class = "tarnet"))
  }
  message("TARNet: Install package 'torch' for the full TARNet. Using ranger fallback.")
  df0 <- as.data.frame(X[w == 0, , drop = FALSE]); df0$y <- y[w == 0]
  df1 <- as.data.frame(X[w == 1, , drop = FALSE]); df1$y <- y[w == 1]
  m0 <- ranger::ranger(y ~ ., data = df0)
  m1 <- ranger::ranger(y ~ ., data = df1)
  structure(list(model_0 = m0, model_1 = m1, X_names = colnames(X), type = "tarnet_ranger_fallback"), class = "tarnet")
}

#' Predict CATE from TARNet
#' @param object fitted \code{tarnet} object
#' @param newdata covariate matrix or data.frame
#' @param ... ignored
#' @return Vector of predicted ITE.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.tarnet(...)
#' }
#' @export
predict.tarnet <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names

  if (identical(object$type, "tarnet_torch")) {
    object$model$eval()
    torch::with_no_grad({
      x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32())
      out <- object$model(x_t)
      as.numeric(out$y1 - out$y0)
    })
  } else if (!is.null(object$model) && inherits(object$model, "nnet")) {
    df <- as.data.frame(newdata)
    df0 <- df; df0$w <- 0
    df1 <- df; df1$w <- 1
    as.vector(predict(object$model, newdata = df1)) - as.vector(predict(object$model, newdata = df0))
  } else {
    df <- as.data.frame(newdata)
    as.vector(predict(object$model_1, data = df)$predictions) - as.vector(predict(object$model_0, data = df)$predictions)
  }
}

# --- CFRNet (Counterfactual Regression Network): TARNet + MMD balancing ---
# MMD^2 with RBF kernel between Phi(X) for treated and control. Ref: https://github.com/clinicalml/cfrnet

.mmd2_rbf_torch <- function(phi0, phi1, sigma = 1) {
  n0 <- phi0$size(1)
  n1 <- phi1$size(1)
  if (n0 == 0L || n1 == 0L)
    return(torch::torch_tensor(0, device = phi0$device, dtype = phi0$dtype))
  d00 <- torch::torch_cdist(phi0, phi0)$pow(2)
  d11 <- torch::torch_cdist(phi1, phi1)$pow(2)
  d01 <- torch::torch_cdist(phi0, phi1)$pow(2)
  g <- 1 / (2 * sigma^2)
  k00 <- torch::torch_exp(-g * d00)$mean()
  k11 <- torch::torch_exp(-g * d11)$mean()
  k01 <- torch::torch_exp(-g * d01)$mean()
  k00 + k11 - 2 * k01
}

.cfrnet_module_torch <- function(input_dim, rep_dim = 100L, hidden = c(200L, 200L, 100L)) {
  hidden <- as.integer(hidden)
  torch::nn_module(
    "CFRNet",
    initialize = function() {
      layers <- list()
      prev <- input_dim
      for (h in hidden) {
        layers[[length(layers) + 1L]] <- torch::nn_linear(prev, h)
        layers[[length(layers) + 1L]] <- torch::nn_elu()
        prev <- h
      }
      self$repr <- do.call(torch::nn_sequential, layers)
      self$head0 <- torch::nn_linear(prev, 1L)
      self$head1 <- torch::nn_linear(prev, 1L)
    },
    forward = function(x) {
      phi <- self$repr(x)
      list(y0 = self$head0(phi)$squeeze(2L), y1 = self$head1(phi)$squeeze(2L), phi = phi)
    }
  )
}

.cfrnet_loss_torch <- function(y0, y1, phi, t, y, mmd_weight, sigma_mmd = 1) {
  loss_mse <- .tarnet_loss_torch(y0, y1, t, y)
  t_ <- t$squeeze()
  t_numeric <- as.numeric(t_)
  idx0 <- which(t_numeric == 0)
  idx1 <- which(t_numeric == 1)
  if (length(idx0) > 0L && length(idx1) > 0L) {
    phi0 <- phi[idx0, , drop = FALSE]
    phi1 <- phi[idx1, , drop = FALSE]
    mmd <- .mmd2_rbf_torch(phi0, phi1, sigma = sigma_mmd)
    return(loss_mse + mmd_weight * mmd)
  }
  loss_mse
}

#' CFRNet — Counterfactual Regression Network (TARNet + MMD balancing)
#'
#' CFRNet adds explicit balancing via MMD^2 (RBF kernel) on the representation \eqn{\Phi(X)} between treated and control,
#' as in \href{https://github.com/clinicalml/cfrnet}{cfrnet}. Same architecture as TARNet (shared representation + two heads).
#'
#' When \pkg{torch} is installed, the full CFRNet is fitted. Otherwise falls back to an nnet or ranger placeholder.
#'
#' @param X covariate matrix or data.frame
#' @param treatment binary treatment 0/1
#' @param y outcome vector
#' @param rep_dim deprecated; use \code{hidden} for layer sizes
#' @param hidden vector of hidden layer sizes (default \code{c(200, 200, 100)})
#' @param mmd_weight weight for MMD balancing term (default 0.1)
#' @param sigma_mmd RBF kernel bandwidth for MMD (default 1)
#' @param batch_size batch size for training (default 64)
#' @param val_split fraction for validation (default 0.2)
#' @param epochs number of training epochs (default 100)
#' @param lr learning rate (default 1e-3)
#' @param verbose if TRUE, print epoch loss
#' @param ... ignored
#' @return Object of class \code{cfrnet} with \code{predict()} for ITE.
#' @references \url{https://github.com/clinicalml/cfrnet}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # cfrnet(...)
#' }
#' @export
cfrnet <- function(X, treatment, y,
                   rep_dim = 100L,
                   hidden = c(200L, 200L, 100L),
                   mmd_weight = 0.1,
                   sigma_mmd = 1,
                   batch_size = 64L,
                   val_split = 0.2,
                   epochs = 100L,
                   lr = 1e-3,
                   verbose = TRUE,
                   ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  X <- as.matrix(X)
  w <- as.integer(treatment)
  y <- as.numeric(y)
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(p))

  if (requireNamespace("torch", quietly = TRUE)) {
    Module <- .cfrnet_module_torch(as.integer(p), rep_dim = rep_dim, hidden = hidden)
    model <- Module()
    X_t <- torch::torch_tensor(X, dtype = torch::torch_float32())
    t_t <- torch::torch_tensor(matrix(w, ncol = 1L), dtype = torch::torch_float32())
    y_t <- torch::torch_tensor(matrix(y, ncol = 1L), dtype = torch::torch_float32())

    n_val <- max(1L, round(n * val_split))
    n_train <- n - n_val
    perm <- sample(n)
    idx_train <- perm[seq_len(n_train)]

    opt <- torch::optim_adam(model$parameters, lr = lr)
    batch_size <- min(batch_size, n_train)
    n_batches <- max(1L, ceiling(n_train / batch_size))

    for (epoch in seq_len(epochs)) {
      model$train(TRUE)
      batch_loss <- 0
      shuffle <- sample(idx_train)
      for (start in seq(1L, n_train, by = batch_size)) {
        idx <- shuffle[start:min(start + batch_size - 1L, n_train)]
        opt$zero_grad()
        out <- model(X_t[idx, , drop = FALSE])
        loss <- .cfrnet_loss_torch(
          out$y0, out$y1, out$phi,
          t_t[idx, , drop = FALSE], y_t[idx, , drop = FALSE],
          mmd_weight = mmd_weight, sigma_mmd = sigma_mmd
        )
        loss$backward()
        opt$step()
        batch_loss <- batch_loss + as.numeric(loss)
      }
      if (verbose && epoch %% 10L == 0L)
        message("CFRNet epoch ", epoch, " loss: ", round(batch_loss / n_batches, 4))
    }

    return(structure(
      list(model = model, X_names = colnames(X), type = "cfrnet_torch", input_dim = p),
      class = "cfrnet"
    ))
  }

  if (requireNamespace("nnet", quietly = TRUE)) {
    df <- as.data.frame(X)
    df$w <- w
    df$y <- y
    size <- max(5, min(20, ncol(X)))
    m <- nnet::nnet(y ~ ., data = df, size = size, linout = TRUE, trace = FALSE, maxit = 200)
    return(structure(list(model = m, X_names = colnames(X), type = "cfrnet_nnet_placeholder"), class = "cfrnet"))
  }
  message("CFRNet: Install package 'torch' for the full CFRNet. Using ranger fallback.")
  df0 <- as.data.frame(X[w == 0, , drop = FALSE]); df0$y <- y[w == 0]
  df1 <- as.data.frame(X[w == 1, , drop = FALSE]); df1$y <- y[w == 1]
  m0 <- ranger::ranger(y ~ ., data = df0)
  m1 <- ranger::ranger(y ~ ., data = df1)
  structure(list(model_0 = m0, model_1 = m1, X_names = colnames(X), type = "cfrnet_ranger_fallback"), class = "cfrnet")
}

#' Predict CATE from CFRNet
#' @param object fitted \code{cfrnet} object
#' @param newdata covariate matrix or data.frame
#' @param ... ignored
#' @return Vector of predicted ITE.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.cfrnet(...)
#' }
#' @export
predict.cfrnet <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names

  if (identical(object$type, "cfrnet_torch")) {
    object$model$eval()
    torch::with_no_grad({
      x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32())
      out <- object$model(x_t)
      as.numeric(out$y1 - out$y0)
    })
  } else if (!is.null(object$model) && inherits(object$model, "nnet")) {
    df <- as.data.frame(newdata)
    df0 <- df; df0$w <- 0
    df1 <- df; df1$w <- 1
    as.vector(predict(object$model, newdata = df1)) - as.vector(predict(object$model, newdata = df0))
  } else {
    df <- as.data.frame(newdata)
    as.vector(predict(object$model_1, data = df)$predictions) - as.vector(predict(object$model_0, data = df)$predictions)
  }
}

# --- GANITE: Causal Inference with Generative Adversarial Nets ---
# Generator generates counterfactual outcomes; Discriminator tries to identify treatment from (x, y0, y1);
# Inference net maps x -> (y0, y1) for ITE. Ref: Yoon et al. (2018) GANITE - https://arxiv.org/abs/1809.00916

.ganite_set_discriminator <- function(generator, discriminator, inference_net) {
  for (p in generator$parameters) p$requires_grad <- FALSE
  for (p in discriminator$parameters) p$requires_grad <- TRUE
  for (p in inference_net$parameters) p$requires_grad <- FALSE
  generator$eval()
  discriminator$train()
  inference_net$eval()
}

.ganite_set_generator <- function(generator, discriminator, inference_net) {
  for (p in generator$parameters) p$requires_grad <- TRUE
  for (p in discriminator$parameters) p$requires_grad <- FALSE
  for (p in inference_net$parameters) p$requires_grad <- FALSE
  generator$train()
  discriminator$eval()
  inference_net$eval()
}

.ganite_set_inference <- function(generator, discriminator, inference_net) {
  for (p in generator$parameters) p$requires_grad <- FALSE
  for (p in discriminator$parameters) p$requires_grad <- FALSE
  for (p in inference_net$parameters) p$requires_grad <- TRUE
  generator$eval()
  discriminator$eval()
  inference_net$train()
}

.ganite_set_test <- function(generator, discriminator, inference_net) {
  for (p in generator$parameters) p$requires_grad <- FALSE
  for (p in discriminator$parameters) p$requires_grad <- FALSE
  for (p in inference_net$parameters) p$requires_grad <- FALSE
  generator$eval()
  discriminator$eval()
  inference_net$eval()
}

.ganite_generator_torch <- function(input_dim, h_dim, flag_dropout) {
  torch::nn_module(
    "GANITEGenerator",
    initialize = function() {
      self$fc1 <- torch::nn_linear(input_dim + 2L, h_dim)
      self$dp1 <- torch::nn_dropout(p = 0.2)
      self$fc2_1 <- torch::nn_linear(h_dim, h_dim)
      self$dp2_1 <- torch::nn_dropout(p = 0.2)
      self$fc2_2 <- torch::nn_linear(h_dim, h_dim)
      self$dp2_2 <- torch::nn_dropout(p = 0.2)
      self$fc2 <- torch::nn_linear(h_dim, h_dim)
      self$dp2 <- torch::nn_dropout(p = 0.2)
      self$fc31 <- torch::nn_linear(h_dim, h_dim)
      self$fc32 <- torch::nn_linear(h_dim, 1L)
      self$fc41 <- torch::nn_linear(h_dim, h_dim)
      self$fc42 <- torch::nn_linear(h_dim, 1L)
      self$flag_dropout <- flag_dropout
    },
    forward = function(x, t, y) {
      inputs <- torch::torch_cat(list(x, t, y), dim = 2L)
      if (self$flag_dropout) {
        h1 <- self$dp1(torch::nnf_relu(self$fc1(inputs)))
        h2_1 <- self$dp2_1(torch::nnf_relu(self$fc2_1(h1)))
        h2_2 <- self$dp2_2(torch::nnf_relu(self$fc2_2(h2_1)))
        h2 <- self$dp2(torch::nnf_relu(self$fc2(h2_2)))
      } else {
        h1 <- torch::nnf_relu(self$fc1(inputs))
        h2_1 <- torch::nnf_relu(self$fc2_1(h1))
        h2_2 <- torch::nnf_relu(self$fc2_2(h2_1))
        h2 <- torch::nnf_relu(self$fc2(h2_2))
      }
      y_hat_0 <- self$fc32(torch::nnf_relu(self$fc31(h2)))
      y_hat_1 <- self$fc42(torch::nnf_relu(self$fc41(h2)))
      torch::torch_cat(list(y_hat_0, y_hat_1), dim = 2L)
    }
  )
}

.ganite_discriminator_torch <- function(input_dim, h_dim, flag_dropout) {
  torch::nn_module(
    "GANITEDiscriminator",
    initialize = function() {
      self$fc1 <- torch::nn_linear(input_dim + 2L, h_dim)
      self$dp1 <- torch::nn_dropout(p = 0.2)
      self$fc2_1 <- torch::nn_linear(h_dim, h_dim)
      self$dp2_1 <- torch::nn_dropout(p = 0.2)
      self$fc2_2 <- torch::nn_linear(h_dim, h_dim)
      self$dp2_2 <- torch::nn_dropout(p = 0.2)
      self$fc2 <- torch::nn_linear(h_dim, h_dim)
      self$dp2 <- torch::nn_dropout(p = 0.2)
      self$fc3 <- torch::nn_linear(h_dim, 1L)
      self$flag_dropout <- flag_dropout
    },
    forward = function(x, t, y, hat_y) {
      input0 <- (1 - t) * y + t * hat_y[, 1, drop = FALSE]
      input1 <- t * y + (1 - t) * hat_y[, 2, drop = FALSE]
      inputs <- torch::torch_cat(list(x, input0, input1), dim = 2L)
      if (self$flag_dropout) {
        h1 <- self$dp1(torch::nnf_relu(self$fc1(inputs)))
        h2_1 <- self$dp2_1(torch::nnf_relu(self$fc2_1(h1)))
        h2_2 <- self$dp2_2(torch::nnf_relu(self$fc2_2(h2_1)))
        h2 <- self$dp2(torch::nnf_relu(self$fc2(h2_2)))
      } else {
        h1 <- torch::nnf_relu(self$fc1(inputs))
        h2_1 <- torch::nnf_relu(self$fc2_1(h1))
        h2_2 <- torch::nnf_relu(self$fc2_2(h2_1))
        h2 <- torch::nnf_relu(self$fc2(h2_2))
      }
      self$fc3(h2)
    }
  )
}

.ganite_inference_torch <- function(input_dim, h_dim, flag_dropout) {
  torch::nn_module(
    "GANITEInference",
    initialize = function() {
      self$fc1 <- torch::nn_linear(input_dim, h_dim)
      self$dp1 <- torch::nn_dropout(p = 0.2)
      self$fc2_1 <- torch::nn_linear(h_dim, h_dim)
      self$dp2_1 <- torch::nn_dropout(p = 0.2)
      self$fc2_2 <- torch::nn_linear(h_dim, h_dim)
      self$dp2_2 <- torch::nn_dropout(p = 0.2)
      self$fc2 <- torch::nn_linear(h_dim, h_dim)
      self$dp2 <- torch::nn_dropout(p = 0.2)
      self$fc31 <- torch::nn_linear(h_dim, h_dim)
      self$fc32 <- torch::nn_linear(h_dim, 1L)
      self$fc41 <- torch::nn_linear(h_dim, h_dim)
      self$fc42 <- torch::nn_linear(h_dim, 1L)
      self$flag_dropout <- flag_dropout
    },
    forward = function(x) {
      if (self$flag_dropout) {
        h1 <- self$dp1(torch::nnf_relu(self$fc1(x)))
        h2_1 <- self$dp2_1(torch::nnf_relu(self$fc2_1(h1)))
        h2_2 <- self$dp2_2(torch::nnf_relu(self$fc2_2(h2_1)))
        h2 <- self$dp2(torch::nnf_relu(self$fc2(h2_2)))
      } else {
        h1 <- torch::nnf_relu(self$fc1(x))
        h2_1 <- torch::nnf_relu(self$fc2_1(h1))
        h2_2 <- torch::nnf_relu(self$fc2_2(h2_1))
        h2 <- torch::nnf_relu(self$fc2(h2_2))
      }
      y_hat_0 <- self$fc32(torch::nnf_relu(self$fc31(h2)))
      y_hat_1 <- self$fc42(torch::nnf_relu(self$fc41(h2)))
      torch::torch_cat(list(y_hat_0, y_hat_1), dim = 2L)
    }
  )
}

#' GANITE — Causal Inference with Generative Adversarial Nets
#'
#' GANITE uses a Generator to produce counterfactual outcomes, a Discriminator to encourage
#' balanced representations, and an Inference network that maps covariates to potential outcomes
#' \eqn{\hat{Y}(0)}, \eqn{\hat{Y}(1)} for ITE. Requires \pkg{torch}.
#'
#' @param X Covariate matrix or data.frame.
#' @param treatment Binary treatment 0/1.
#' @param y Outcome vector.
#' @param h_dim Hidden layer size for Generator, Discriminator, and Inference net (default 50).
#' @param iterations Number of training iterations (default 1000).
#' @param batch_size Batch size (default 64).
#' @param alpha Weight for GAN loss in Generator (default 1).
#' @param beta Weight for ATE-supervised loss in Inference net (default 1).
#' @param lr Learning rate for all nets (default 1e-4).
#' @param dropout If TRUE, use dropout 0.2 in all nets (default FALSE).
#' @param use_adamw If TRUE, use AdamW optimizer (default FALSE).
#' @param verbose If TRUE, print progress every 100 iterations.
#' @param device Device for torch: \code{"cuda"}, \code{"cpu"}, or \code{NULL} (default: use CUDA if available, else CPU).
#' @param ... Ignored.
#' @return Object of class \code{ganite} with \code{predict()} for ITE. When \pkg{torch} is not available, falls back to nnet or ranger placeholder.
#' @references Yoon J., Jordon J., van der Schaar M. (2018). GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets. ICLR. \url{https://arxiv.org/abs/1809.00916}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # ganite(...)
#' }
#' @export
ganite <- function(X, treatment, y,
                   h_dim = 50L,
                   iterations = 1000L,
                   batch_size = 64L,
                   alpha = 1,
                   beta = 1,
                   lr = 1e-4,
                   dropout = FALSE,
                   use_adamw = FALSE,
                   verbose = TRUE,
                   device = NULL,
                   ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  X <- as.matrix(X)
  w <- as.integer(treatment)
  y <- as.numeric(y)
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(p))

  if (requireNamespace("torch", quietly = TRUE)) {
    if (is.null(device))
      device <- if (torch::cuda_is_available()) "cuda" else "cpu"
    h_dim <- as.integer(h_dim)
    batch_size <- min(batch_size, n)

    X_t <- torch::torch_tensor(X, dtype = torch::torch_float32(), device = device)
    t_t <- torch::torch_tensor(matrix(as.numeric(w), ncol = 1L), dtype = torch::torch_float32(), device = device)
    y_t <- torch::torch_tensor(matrix(y, ncol = 1L), dtype = torch::torch_float32(), device = device)

    Generator <- .ganite_generator_torch(p, h_dim, dropout)
    Discriminator <- .ganite_discriminator_torch(p, h_dim, dropout)
    InferenceNet <- .ganite_inference_torch(p, h_dim, dropout)
    generator <- Generator()
    discriminator <- Discriminator()
    inference_net <- InferenceNet()
    generator$to(device = device)
    discriminator$to(device = device)
    inference_net$to(device = device)

    if (use_adamw && exists("optim_adamw", mode = "function", where = asNamespace("torch"))) {
      opt_g <- torch::optim_adamw(generator$parameters, lr = lr, betas = c(0.9, 0.999))
      opt_d <- torch::optim_adamw(discriminator$parameters, lr = lr, betas = c(0.9, 0.999))
      opt_i <- torch::optim_adamw(inference_net$parameters, lr = lr, betas = c(0.9, 0.999))
    } else {
      opt_g <- torch::optim_adam(generator$parameters, lr = lr, betas = c(0.9, 0.999))
      opt_d <- torch::optim_adam(discriminator$parameters, lr = lr, betas = c(0.9, 0.999))
      opt_i <- torch::optim_adam(inference_net$parameters, lr = lr, betas = c(0.9, 0.999))
    }

    for (iter in seq_len(iterations)) {
      perm <- sample(n)
      for (start in seq(1L, n, by = batch_size)) {
        idx <- perm[start:min(start + batch_size - 1L, n)]
        x <- X_t[idx, , drop = FALSE]
        t <- t_t[idx, , drop = FALSE]
        y_b <- y_t[idx, , drop = FALSE]

        .ganite_set_discriminator(generator, discriminator, inference_net)
        for (. in 1:2) {
          y_tilde <- generator(x, t, y_b)
          d_logit <- discriminator(x, t, y_b, y_tilde)
          D_loss <- torch::nnf_binary_cross_entropy_with_logits(d_logit, t)
          opt_d$zero_grad()
          D_loss$backward(retain_graph = TRUE)
          opt_d$step()
        }

        .ganite_set_generator(generator, discriminator, inference_net)
        y_tilde <- generator(x, t, y_b)
        d_logit <- discriminator(x, t, y_b, y_tilde)
        D_loss <- torch::nnf_binary_cross_entropy_with_logits(d_logit, t)
        G_loss_gan <- -D_loss
        y_est <- t * y_tilde[, 2, drop = FALSE] + (1 - t) * y_tilde[, 1, drop = FALSE]
        G_loss_factual <- torch::nnf_mse_loss(y_est, y_b)
        G_loss <- G_loss_factual + alpha * G_loss_gan
        opt_g$zero_grad()
        G_loss$backward(retain_graph = TRUE)
        opt_g$step()

        .ganite_set_inference(generator, discriminator, inference_net)
        y_hat <- inference_net(x)
        y_tilde <- generator(x, t, y_b)
        y_t0 <- t * y_b + (1 - t) * y_tilde[, 2, drop = FALSE]
        I_loss1 <- torch::nnf_mse_loss(y_hat[, 2, drop = FALSE], y_t0)
        y_t1 <- (1 - t) * y_b + t * y_tilde[, 1, drop = FALSE]
        I_loss2 <- torch::nnf_mse_loss(y_hat[, 1, drop = FALSE], y_t1)
        y_ate <- torch::torch_mean(t * y_b - (1 - t) * y_b)
        y_hat_ate <- torch::torch_mean(y_hat[, 2] - y_hat[, 1])
        supervised_loss <- torch::nnf_mse_loss(y_hat_ate$view(1L), y_ate$detach()$view(1L))
        I_loss <- I_loss1 + I_loss2 + beta * supervised_loss
        opt_i$zero_grad()
        I_loss$backward()
        opt_i$step()
      }
      if (verbose && iter %% 100L == 0L)
        message("GANITE iteration ", iter, " / ", iterations)
    }

    .ganite_set_test(generator, discriminator, inference_net)
    return(structure(
      list(generator = generator, discriminator = discriminator, inference_net = inference_net,
           X_names = colnames(X), type = "ganite_torch", input_dim = p, device = device),
      class = "ganite"
    ))
  }

  if (requireNamespace("nnet", quietly = TRUE)) {
    df <- as.data.frame(X)
    df$w <- w
    df$y <- y
    size <- max(5L, min(20L, ncol(X)))
    m <- nnet::nnet(y ~ ., data = df, size = size, linout = TRUE, trace = FALSE, maxit = 200L)
    return(structure(list(model = m, X_names = colnames(X), type = "ganite_nnet_placeholder"),
                    class = "ganite"))
  }
  message("GANITE: Install package 'torch' for the full GANITE. Using ranger fallback.")
  df0 <- as.data.frame(X[w == 0, , drop = FALSE])
  df0$y <- y[w == 0]
  df1 <- as.data.frame(X[w == 1, , drop = FALSE])
  df1$y <- y[w == 1]
  m0 <- ranger::ranger(y ~ ., data = df0)
  m1 <- ranger::ranger(y ~ ., data = df1)
  structure(list(model_0 = m0, model_1 = m1, X_names = colnames(X), type = "ganite_ranger_fallback"),
            class = "ganite")
}

#' Predict ITE from GANITE
#'
#' For torch GANITE, uses the Inference network to predict \eqn{\hat{Y}(0)} and \eqn{\hat{Y}(1)}, then ITE = \eqn{\hat{Y}(1) - \hat{Y}(0)}.
#'
#' @param object Fitted \code{ganite} object.
#' @param newdata Covariate matrix or data.frame.
#' @param ... Ignored.
#' @return Vector of predicted ITE.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.ganite(...)
#' }
#' @export
predict.ganite <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names

  if (identical(object$type, "ganite_torch")) {
    .ganite_set_test(object$generator, object$discriminator, object$inference_net)
    object$inference_net$eval()
    dev <- if (!is.null(object$device)) object$device else "cpu"
    torch::with_no_grad({
      x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = dev)
      y_hat <- object$inference_net(x_t)
      ite <- as.numeric(y_hat[, 2] - y_hat[, 1])
    })
    return(ite)
  }

  df <- as.data.frame(newdata)
  if (!is.null(object$model) && inherits(object$model, "nnet")) {
    df0 <- df
    df0$w <- 0
    df1 <- df
    df1$w <- 1
    p0 <- predict(object$model, newdata = df0)
    p1 <- predict(object$model, newdata = df1)
    return(as.vector(p1) - as.vector(p0))
  }
  as.vector(predict(object$model_1, data = df)$predictions) - as.vector(predict(object$model_0, data = df)$predictions)
}

# --- CausalGAN: Structural GAN with node-wise generators ---

.causalgan_make_mlp <- function(input_dim, output_dim, hidden_dim, layers = 3L, out_activation = NULL) {
  layers <- as.integer(max(1L, layers))
  hidden_dim <- as.integer(hidden_dim)
  mods <- list()
  if (layers <= 1L) {
    mods[[1L]] <- torch::nn_linear(as.integer(input_dim), as.integer(output_dim))
  } else {
    mods[[1L]] <- torch::nn_linear(as.integer(input_dim), hidden_dim)
    mods[[2L]] <- torch::nn_relu()
    for (i in 2:(layers - 1L)) {
      mods[[length(mods) + 1L]] <- torch::nn_linear(hidden_dim, hidden_dim)
      mods[[length(mods) + 1L]] <- torch::nn_relu()
    }
    mods[[length(mods) + 1L]] <- torch::nn_linear(hidden_dim, as.integer(output_dim))
  }
  if (!is.null(out_activation))
    mods[[length(mods) + 1L]] <- out_activation
  do.call(torch::nn_sequential, mods)
}

#' CausalGAN (Structural Equation GAN)
#'
#' CausalGAN models \eqn{X \rightarrow T \rightarrow Y} with node-wise generators:
#' \eqn{G_X(\epsilon_X)}, \eqn{G_T(X, \epsilon_T)}, \eqn{G_Y(X, T, \epsilon_Y)}.
#' A global discriminator scores the full joint \eqn{(X, T, Y)} while auxiliary labellers
#' regularize each causal component to improve structural equation fidelity and
#' interventional calibration.
#'
#' @param X Covariate matrix or data.frame.
#' @param treatment Binary treatment 0/1.
#' @param y Outcome vector.
#' @param hidden_dim Hidden dimension for all MLPs (default 128).
#' @param noise_x Noise dimension for \eqn{G_X} (default ncol(X)).
#' @param noise_t Noise dimension for \eqn{G_T} (default 4).
#' @param noise_y Noise dimension for \eqn{G_Y} (default 4).
#' @param epochs Number of training epochs (default 300).
#' @param batch_size Batch size (default 128).
#' @param lr_g Learning rate for generator (default 2e-4).
#' @param lr_d Learning rate for discriminator/labellers (default 2e-4).
#' @param lambda_label Weight of labeller loss relative to GAN loss (default 0.5).
#' @param label_smooth Smoothed real label for BCE (default 0.9).
#' @param verbose If TRUE, print progress every 50 epochs.
#' @param device Device for torch: \code{"cuda"}, \code{"cpu"}, or \code{NULL} (default: CUDA if available, else CPU).
#' @param ... Ignored.
#' @return Object of class \code{causalGAN}. Use \code{predict()} for interventional samples and ITE.
#' @references CausalGAN notebook in this package (\code{causal_gan.ipynb}).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # causalGAN(...)
#' }
#' @export
causalGAN <- function(X, treatment, y,
                      hidden_dim = 128L,
                      noise_x = NULL,
                      noise_t = 4L,
                      noise_y = 4L,
                      epochs = 300L,
                      batch_size = 128L,
                      lr_g = 2e-4,
                      lr_d = 2e-4,
                      lambda_label = 0.5,
                      label_smooth = 0.9,
                      verbose = TRUE,
                      device = NULL,
                      ...) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("causalGAN requires package 'torch'. Please install torch.", call. = FALSE)

  if (inherits(X, "data.frame")) X <- as.matrix(X)
  X <- as.matrix(X)
  w <- as.numeric(treatment)
  y <- as.numeric(y)
  n <- nrow(X)
  p <- ncol(X)
  if (length(w) != n || length(y) != n)
    stop("X, treatment, and y must have compatible lengths.", call. = FALSE)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(p))
  if (is.null(device))
    device <- if (torch::cuda_is_available()) "cuda" else "cpu"
  if (is.null(noise_x)) noise_x <- p

  dim_t <- 1L
  dim_y <- 1L
  hidden_dim <- as.integer(hidden_dim)
  noise_x <- as.integer(noise_x)
  noise_t <- as.integer(noise_t)
  noise_y <- as.integer(noise_y)
  batch_size <- max(2L, min(as.integer(batch_size), n))

  GeneratorX <- torch::nn_module(
    initialize = function() {
      self$net <- .causalgan_make_mlp(noise_x, p, hidden_dim, layers = 3L)
    },
    forward = function(noise) self$net(noise)
  )
  GeneratorT <- torch::nn_module(
    initialize = function() {
      self$net <- .causalgan_make_mlp(p + noise_t, dim_t, hidden_dim, layers = 2L, out_activation = torch::nn_sigmoid())
    },
    forward = function(x_hat, noise) {
      inp <- torch::torch_cat(list(x_hat, noise), dim = 2L)
      prob <- self$net(inp)
      if (self$training) prob else (prob > 0.5)$to(dtype = torch::torch_float32())
    }
  )
  GeneratorY <- torch::nn_module(
    initialize = function() {
      self$net <- .causalgan_make_mlp(p + dim_t + noise_y, dim_y, hidden_dim, layers = 3L)
    },
    forward = function(x_hat, t_hat, noise) {
      inp <- torch::torch_cat(list(x_hat, t_hat, noise), dim = 2L)
      self$net(inp)
    }
  )
  CausalGenerator <- torch::nn_module(
    initialize = function() {
      self$g_x <- GeneratorX()
      self$g_t <- GeneratorT()
      self$g_y <- GeneratorY()
    },
    sample_noise = function(m) {
      list(
        nx = torch::torch_randn(c(m, noise_x), device = device),
        nt = torch::torch_randn(c(m, noise_t), device = device),
        ny = torch::torch_randn(c(m, noise_y), device = device)
      )
    },
    forward = function(m, do_t = NULL) {
      nz <- self$sample_noise(m)
      x_hat <- self$g_x(nz$nx)
      if (is.null(do_t)) {
        t_hat <- self$g_t(x_hat, nz$nt)
      } else {
        t_hat <- torch::torch_full(c(m, dim_t), as.numeric(do_t), dtype = torch::torch_float32(), device = device)
      }
      y_hat <- self$g_y(x_hat, t_hat, nz$ny)
      joint <- torch::torch_cat(list(x_hat, t_hat, y_hat), dim = 2L)
      list(x = x_hat, t = t_hat, y = y_hat, joint = joint)
    }
  )
  Discriminator <- torch::nn_module(
    initialize = function() {
      self$net <- .causalgan_make_mlp(p + dim_t + dim_y, 1L, hidden_dim, layers = 3L, out_activation = torch::nn_sigmoid())
    },
    forward = function(joint) self$net(joint)
  )
  LabellerX <- torch::nn_module(
    initialize = function() {
      self$net <- .causalgan_make_mlp(p, 1L, max(2L, hidden_dim %/% 2L), layers = 2L, out_activation = torch::nn_sigmoid())
    },
    forward = function(x) self$net(x)
  )
  LabellerT <- torch::nn_module(
    initialize = function() {
      self$net <- .causalgan_make_mlp(p + dim_t, 1L, max(2L, hidden_dim %/% 2L), layers = 2L, out_activation = torch::nn_sigmoid())
    },
    forward = function(x, t) self$net(torch::torch_cat(list(x, t), dim = 2L))
  )
  LabellerY <- torch::nn_module(
    initialize = function() {
      self$net <- .causalgan_make_mlp(p + dim_t + dim_y, 1L, max(2L, hidden_dim %/% 2L), layers = 2L, out_activation = torch::nn_sigmoid())
    },
    forward = function(x, t, y_) self$net(torch::torch_cat(list(x, t, y_), dim = 2L))
  )

  gen <- CausalGenerator()
  disc <- Discriminator()
  lab_x <- LabellerX()
  lab_t <- LabellerT()
  lab_y <- LabellerY()
  gen$to(device = device)
  disc$to(device = device)
  lab_x$to(device = device)
  lab_t$to(device = device)
  lab_y$to(device = device)

  opt_G <- torch::optim_adam(gen$parameters, lr = lr_g, betas = c(0.5, 0.999))
  opt_D <- torch::optim_adam(disc$parameters, lr = lr_d, betas = c(0.5, 0.999))
  opt_L <- torch::optim_adam(c(lab_x$parameters, lab_t$parameters, lab_y$parameters), lr = lr_d, betas = c(0.5, 0.999))

  real_joint <- cbind(X, w, y)
  real_joint_t <- torch::torch_tensor(real_joint, dtype = torch::torch_float32(), device = device)
  real_label <- function(m) torch::torch_full(c(m, 1L), label_smooth, dtype = torch::torch_float32(), device = device)
  fake_label <- function(m) torch::torch_zeros(c(m, 1L), dtype = torch::torch_float32(), device = device)
  bce <- function(pred, target) torch::nnf_binary_cross_entropy(pred, target)

  disc_step <- function(real_batch) {
    B <- as.integer(real_batch$size(1))
    rx <- real_batch[, 1:p, drop = FALSE]
    rt <- real_batch[, (p + 1L):(p + 1L), drop = FALSE]
    ry <- real_batch[, (p + 2L):(p + 2L), drop = FALSE]

    opt_D$zero_grad()
    opt_L$zero_grad()
    loss_d_real <- bce(disc(real_batch), real_label(B))
    loss_lx_real <- bce(lab_x(rx), real_label(B))
    loss_lt_real <- bce(lab_t(rx, rt), real_label(B))
    loss_ly_real <- bce(lab_y(rx, rt, ry), real_label(B))

    fake_joint <- torch::with_no_grad({
      gen(B)$joint
    })
    fx <- fake_joint[, 1:p, drop = FALSE]
    ft <- fake_joint[, (p + 1L):(p + 1L), drop = FALSE]
    fy <- fake_joint[, (p + 2L):(p + 2L), drop = FALSE]
    loss_d_fake <- bce(disc(fake_joint), fake_label(B))
    loss_lx_fake <- bce(lab_x(fx), fake_label(B))
    loss_lt_fake <- bce(lab_t(fx, ft), fake_label(B))
    loss_ly_fake <- bce(lab_y(fx, ft, fy), fake_label(B))

    loss_d <- loss_d_real + loss_d_fake
    loss_l <- loss_lx_real + loss_lx_fake + loss_lt_real + loss_lt_fake + loss_ly_real + loss_ly_fake
    total <- loss_d + lambda_label * loss_l
    total$backward()
    opt_D$step()
    opt_L$step()
    as.numeric(total$item())
  }

  gen_step <- function(B) {
    B <- as.integer(B)
    opt_G$zero_grad()
    fake <- gen(B)
    loss_g <- bce(disc(fake$joint), real_label(B))
    loss_lx <- bce(lab_x(fake$x), real_label(B))
    loss_lt <- bce(lab_t(fake$x, fake$t), real_label(B))
    loss_ly <- bce(lab_y(fake$x, fake$t, fake$y), real_label(B))
    total <- loss_g + lambda_label * (loss_lx + loss_lt + loss_ly)
    total$backward()
    opt_G$step()
    as.numeric(total$item())
  }

  history_d <- numeric(epochs)
  history_g <- numeric(epochs)
  if (verbose) {
    message(strrep("=", 60))
    message("Starting CausalGAN training ...")
    message(strrep("=", 60))
  }
  for (epoch in seq_len(epochs)) {
    perm <- sample.int(n)
    epoch_d <- numeric(0L)
    epoch_g <- numeric(0L)
    for (start in seq.int(1L, n, by = batch_size)) {
      end <- min(start + batch_size - 1L, n)
      idx <- perm[start:end]
      if (length(idx) < batch_size) next
      real_batch <- real_joint_t[idx, , drop = FALSE]
      d_loss <- disc_step(real_batch)
      g_loss <- gen_step(length(idx))
      epoch_d <- c(epoch_d, d_loss)
      epoch_g <- c(epoch_g, g_loss)
    }
    history_d[epoch] <- if (length(epoch_d)) mean(epoch_d) else NA_real_
    history_g[epoch] <- if (length(epoch_g)) mean(epoch_g) else NA_real_
    if (verbose && (epoch == 1L || epoch %% 50L == 0L)) {
      message(
        sprintf(
          "  Epoch %4d/%d  |  D+L loss: %.4f  |  G loss: %.4f",
          epoch, epochs, history_d[epoch], history_g[epoch]
        )
      )
    }
  }
  if (verbose) message("Training complete.")

  gen$eval()
  disc$eval()
  lab_x$eval()
  lab_t$eval()
  lab_y$eval()
  structure(
    list(
      generator = gen,
      discriminator = disc,
      labeller_x = lab_x,
      labeller_t = lab_t,
      labeller_y = lab_y,
      X_names = colnames(X),
      input_dim = p,
      history_d = history_d,
      history_g = history_g,
      noise_t = noise_t,
      noise_y = noise_y,
      type = "causalgan_torch",
      device = device
    ),
    class = "causalGAN"
  )
}

#' Predict from CausalGAN
#'
#' Produces interventional potential outcomes \eqn{\hat{Y}(0)}, \eqn{\hat{Y}(1)} and ITE.
#'
#' @param object Fitted \code{causalGAN} object.
#' @param newdata Optional matrix/data.frame of covariates. If NULL, samples from learned SCM.
#' @param n_samples Number of generated draws when \code{newdata = NULL} (default 1000).
#' @param ... Ignored.
#' @return Data frame with columns \code{y0}, \code{y1}, and \code{ite}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.causalGAN(...)
#' }
#' @export
predict.causalGAN <- function(object, newdata = NULL, n_samples = 1000L, ...) {
  if (!identical(object$type, "causalgan_torch"))
    stop("Unsupported causalGAN object type.", call. = FALSE)

  gen <- object$generator
  p <- object$input_dim
  device <- if (!is.null(object$device)) object$device else "cpu"
  noise_y <- if (!is.null(object$noise_y)) as.integer(object$noise_y) else 4L
  gen$eval()

  if (is.null(newdata)) {
    n_samples <- as.integer(max(1L, n_samples))
    out <- torch::with_no_grad({
      y1 <- gen(n_samples, do_t = 1)$y$squeeze(2L)
      y0 <- gen(n_samples, do_t = 0)$y$squeeze(2L)
      list(y0 = as.numeric(y0$to(device = torch::torch_device("cpu"))),
           y1 = as.numeric(y1$to(device = torch::torch_device("cpu"))))
    })
    return(data.frame(y0 = out$y0, y1 = out$y1, ite = out$y1 - out$y0))
  }

  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  newdata <- as.matrix(newdata)
  if (ncol(newdata) != p)
    stop("newdata must have the same number of columns as training X.", call. = FALSE)

  x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = device)
  n <- nrow(newdata)
  y_pair <- torch::with_no_grad({
    nz_y <- torch::torch_randn(c(n, noise_y), device = device)
    t0 <- torch::torch_zeros(c(n, 1L), dtype = torch::torch_float32(), device = device)
    t1 <- torch::torch_ones(c(n, 1L), dtype = torch::torch_float32(), device = device)
    y0 <- gen$g_y(x_t, t0, nz_y)$squeeze(2L)
    y1 <- gen$g_y(x_t, t1, nz_y)$squeeze(2L)
    list(y0 = as.numeric(y0$to(device = torch::torch_device("cpu"))),
         y1 = as.numeric(y1$to(device = torch::torch_device("cpu"))))
  })
  data.frame(y0 = y_pair$y0, y1 = y_pair$y1, ite = y_pair$y1 - y_pair$y0)
}

# --- CASTLE: CAusal STructure LEarning Regularization ---

.castle_module_torch <- function(d, hidden_dim = 64L, num_layers = 3L,
                                 lambda_reg = 1.0, beta_sparsity = 0.1,
                                 acyc_weight = 1.0, recon_weight = 0.5,
                                 neighbor_temp = 10.0, y_index = NULL) {
  y_index <- if (is.null(y_index)) as.integer(d) else as.integer(y_index)
  if (y_index < 1L || y_index > d) stop("y_index must be in [1, d].", call. = FALSE)
  torch::nn_module(
    "CASTLE",
    initialize = function() {
      self$d <- as.integer(d)
      self$lambda_reg <- lambda_reg
      self$beta_sparsity <- beta_sparsity
      self$acyc_weight <- acyc_weight
      self$recon_weight <- recon_weight
      self$neighbor_temp <- neighbor_temp
      self$y_index <- y_index
      self$A <- torch::nn_parameter(torch::torch_randn(self$d, self$d) * 0.03)
      self$register_buffer(
        "loop_mask",
        1 - torch::torch_eye(self$d, dtype = torch::torch_float32())
      )

      layers <- list()
      inp <- self$d
      for (i in seq_len(max(1L, as.integer(num_layers) - 1L))) {
        layers[[length(layers) + 1L]] <- torch::nn_linear(inp, as.integer(hidden_dim))
        layers[[length(layers) + 1L]] <- torch::nn_relu()
        inp <- as.integer(hidden_dim)
      }
      layers[[length(layers) + 1L]] <- torch::nn_linear(inp, 1L)
      self$predictor <- do.call(torch::nn_sequential, layers)
      self$decoder <- torch::nn_sequential(
        torch::nn_linear(self$d, as.integer(hidden_dim)),
        torch::nn_relu(),
        torch::nn_linear(as.integer(hidden_dim), self$d)
      )
    },
    get_adj = function() {
      self$A * self$loop_mask
    },
    h_dag = function(A) {
      # Evaluate matrix exponential in CPU float64 for numerical stability.
      dev <- A$device
      dt <- A$dtype
      M <- (A * A)$to(device = torch::torch_device("cpu"), dtype = torch::torch_float64())
      h <- torch::torch_trace(torch::torch_matrix_exp(M)) - self$d
      h$to(device = dev, dtype = dt)
    },
    forward = function(Z) {
      A <- self$get_adj()
      Z_in <- Z$clone()
      Z_in[, self$y_index] <- 0

      importance <- torch::torch_sigmoid(torch::torch_abs(A)$sum(dim = 1L))
      h <- Z_in * importance
      pred <- self$predictor(h)$squeeze(2L)
      z_hat <- self$decoder(h)

      neighbor_score <- torch::torch_abs(A)$sum(dim = 1L) + torch::torch_abs(A)$sum(dim = 2L)
      soft_mask <- torch::torch_sigmoid(self$neighbor_temp * neighbor_score)

      list(pred = pred, A = A, z_hat = z_hat, soft_mask = soft_mask)
    },
    loss = function(Z, y_true) {
      out <- self$forward(Z)
      mse <- torch::nnf_mse_loss(out$pred, y_true)
      sparsity <- self$beta_sparsity * torch::torch_abs(out$A)$sum()
      acyc <- self$acyc_weight * self$h_dag(out$A)
      err <- (out$z_hat - Z)^2
      recon <- self$recon_weight * (out$soft_mask * err)$mean()
      total <- self$lambda_reg * mse + sparsity + acyc + recon
      list(
        total = total, mse = mse$detach(), sparsity = sparsity$detach(),
        acyc = acyc$detach(), recon = recon$detach()
      )
    }
  )
}

#' CASTLE (CAusal STructure LEarning Regularization)
#'
#' Torch implementation of CASTLE translated from the Python tutorial module:
#' learn weighted adjacency \eqn{A}, optimize prediction + sparsity + acyclicity
#' (NOTEARS-style) + neighborhood reconstruction regularization.
#'
#' @param X Numeric matrix or data.frame of predictors.
#' @param y Numeric outcome.
#' @param hidden_dim Hidden layer width (default 64).
#' @param num_layers Number of predictor layers (default 3).
#' @param lambda_reg Weight on prediction loss (default 1).
#' @param beta_sparsity L1 weight on adjacency (default 0.1).
#' @param acyc_weight Acyclicity penalty weight (default 1).
#' @param recon_weight Neighborhood reconstruction weight (default 0.5).
#' @param neighbor_temp Temperature for neighborhood soft mask (default 10).
#' @param y_index 1-based index of outcome column in \code{Z=[X,y]}. Default last column.
#' @param epochs Number of training epochs (default 200).
#' @param batch_size Batch size (default 128).
#' @param learning_rate Adam learning rate (default 1e-3).
#' @param weight_decay Adam weight decay (default 1e-5).
#' @param verbose If TRUE, print progress every 20 epochs.
#' @param device Device string: \code{"cuda"}, \code{"cpu"}, or \code{NULL}.
#' @param threshold Threshold used by summary/plot for edge filtering (default 0.05).
#' @param ... Ignored.
#' @return Object of class \code{castle}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # castle(...)
#' }
#' @export
castle <- function(X, y,
                   hidden_dim = 64L,
                   num_layers = 3L,
                   lambda_reg = 1.0,
                   beta_sparsity = 0.1,
                   acyc_weight = 1.0,
                   recon_weight = 0.5,
                   neighbor_temp = 10.0,
                   y_index = NULL,
                   epochs = 200L,
                   batch_size = 128L,
                   learning_rate = 1e-3,
                   weight_decay = 1e-5,
                   verbose = TRUE,
                   device = NULL,
                   threshold = 0.05,
                   ...) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("castle() requires package 'torch'.", call. = FALSE)
  X <- as.matrix(X)
  y <- as.numeric(y)
  if (nrow(X) != length(y)) stop("X and y must have compatible dimensions.", call. = FALSE)
  n <- nrow(X)
  p <- ncol(X)
  X_names <- if (!is.null(colnames(X))) colnames(X) else paste0("X", seq_len(p))
  Z <- cbind(X, y = y)
  Z_names <- c(X_names, "y")
  d <- ncol(Z)
  if (is.null(y_index)) y_index <- d
  y_index <- as.integer(y_index)
  if (y_index < 1L || y_index > d) stop("y_index must be in [1, ncol(Z)].", call. = FALSE)

  if (is.null(device)) device <- if (torch::cuda_is_available()) "cuda" else "cpu"
  Module <- .castle_module_torch(
    d = as.integer(d),
    hidden_dim = as.integer(hidden_dim),
    num_layers = as.integer(num_layers),
    lambda_reg = lambda_reg,
    beta_sparsity = beta_sparsity,
    acyc_weight = acyc_weight,
    recon_weight = recon_weight,
    neighbor_temp = neighbor_temp,
    y_index = y_index
  )
  model <- Module()
  model$to(device = device)

  Z_t <- torch::torch_tensor(Z, dtype = torch::torch_float32(), device = device)
  y_t <- torch::torch_tensor(y, dtype = torch::torch_float32(), device = device)
  opt <- torch::optim_adam(model$parameters, lr = learning_rate, weight_decay = weight_decay)
  batch_size <- min(as.integer(batch_size), n)
  n_batches <- max(1L, ceiling(n / batch_size))

  history <- data.frame(
    epoch = integer(0), total = numeric(0), mse = numeric(0),
    sparsity = numeric(0), acyc = numeric(0), recon = numeric(0)
  )

  for (epoch in seq_len(as.integer(epochs))) {
    model$train(TRUE)
    ep_total <- ep_mse <- ep_sparse <- ep_acyc <- ep_recon <- 0
    perm <- sample.int(n)
    for (start in seq(1L, n, by = batch_size)) {
      idx <- perm[start:min(start + batch_size - 1L, n)]
      opt$zero_grad()
      losses <- model$loss(Z_t[idx, , drop = FALSE], y_t[idx])
      losses$total$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 5.0)
      opt$step()
      ep_total <- ep_total + as.numeric(losses$total)
      ep_mse <- ep_mse + as.numeric(losses$mse)
      ep_sparse <- ep_sparse + as.numeric(losses$sparsity)
      ep_acyc <- ep_acyc + as.numeric(losses$acyc)
      ep_recon <- ep_recon + as.numeric(losses$recon)
    }
    history[nrow(history) + 1L, ] <- list(
      epoch, ep_total / n_batches, ep_mse / n_batches, ep_sparse / n_batches,
      ep_acyc / n_batches, ep_recon / n_batches
    )
    if (verbose && (epoch %% 20L == 0L || epoch == 1L || epoch == as.integer(epochs))) {
      message(
        sprintf(
          "CASTLE epoch %d | total %.4f | mse %.4f | sparse %.4f | acyc %.4f | recon %.4f",
          epoch, history$total[nrow(history)], history$mse[nrow(history)],
          history$sparsity[nrow(history)], history$acyc[nrow(history)], history$recon[nrow(history)]
        )
      )
    }
  }

  model$eval()
  torch::with_no_grad({
    A <- as.matrix(model$get_adj()$to(device = "cpu"))
  })
  colnames(A) <- Z_names
  rownames(A) <- Z_names

  structure(
    list(
      model = model,
      adjacency = A,
      threshold = threshold,
      y_index = y_index,
      var_names = Z_names,
      X_names = X_names,
      type = "castle_torch",
      history = history,
      device = device
    ),
    class = "castle"
  )
}

#' Predict outcome from CASTLE
#'
#' @param object Fitted \code{castle} object.
#' @param newdata Matrix or data.frame containing predictors \code{X}.
#' @param ... Ignored.
#' @return Numeric vector of predicted outcomes.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.castle(...)
#' }
#' @export
predict.castle <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  newdata <- as.matrix(newdata)
  if (ncol(newdata) != length(object$X_names))
    stop("newdata must have same number of columns as training X.", call. = FALSE)
  if (!is.null(object$X_names)) colnames(newdata) <- object$X_names
  Z_new <- cbind(newdata, y = rep(0, nrow(newdata)))
  object$model$eval()
  dev <- if (!is.null(object$device)) object$device else "cpu"
  torch::with_no_grad({
    z_t <- torch::torch_tensor(Z_new, dtype = torch::torch_float32(), device = dev)
    out <- object$model$forward(z_t)
    as.numeric(out$pred)
  })
}

#' Summarize CASTLE structure
#'
#' @param object Fitted \code{castle} object.
#' @param threshold Edge threshold (default \code{object$threshold}).
#' @param top_n Number of strongest edges to print (default 10).
#' @param ... Ignored.
#' @return Invisibly returns a data.frame with edge strengths.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # summary.castle(...)
#' }
#' @export
summary.castle <- function(object, threshold = object$threshold, top_n = 10L, ...) {
  A <- object$adjacency
  diag(A) <- 0
  sel <- abs(A) >= threshold
  idx <- which(sel, arr.ind = TRUE)
  if (nrow(idx) == 0L) {
    cat("CASTLE summary: no edges above threshold =", threshold, "\n")
    return(invisible(data.frame(from = character(0), to = character(0), weight = numeric(0))))
  }
  edges <- data.frame(
    from = rownames(A)[idx[, 1]],
    to = colnames(A)[idx[, 2]],
    weight = A[idx],
    abs_weight = abs(A[idx]),
    stringsAsFactors = FALSE
  )
  edges <- edges[order(edges$abs_weight, decreasing = TRUE), , drop = FALSE]
  cat("CASTLE summary: ", nrow(edges), " edges above threshold ", threshold, "\n", sep = "")
  print(utils::head(edges[, c("from", "to", "weight"), drop = FALSE], as.integer(top_n)))
  invisible(edges)
}

#' Plot CASTLE graph
#'
#' Uses \pkg{igraph} for graph construction and either \pkg{ggraph} (if installed)
#' or base \pkg{igraph} plotting.
#'
#' @param x Fitted \code{castle} object.
#' @param threshold Edge threshold (default \code{x$threshold}).
#' @param layout Layout name for \code{igraph::layout_with_*} (default \code{"fr"}).
#' @param edge_alpha Edge alpha for ggraph (default 0.8).
#' @param ... Additional parameters passed to plotting backend.
#' @return Graph object invisibly.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # plot.castle(...)
#' }
#' @export
plot.castle <- function(x, threshold = x$threshold, layout = "fr", edge_alpha = 0.8, ...) {
  if (!requireNamespace("igraph", quietly = TRUE))
    stop("plot.castle() requires package 'igraph'.", call. = FALSE)
  A <- x$adjacency
  diag(A) <- 0
  A[abs(A) < threshold] <- 0
  g <- igraph::graph_from_adjacency_matrix(A, mode = "directed", weighted = TRUE, diag = FALSE)

  if (requireNamespace("ggraph", quietly = TRUE) && requireNamespace("ggplot2", quietly = TRUE)) {
    return(
      ggraph::ggraph(g, layout = layout) +
        ggraph::geom_edge_link(
          ggplot2::aes(width = abs(weight), alpha = abs(weight), color = weight > 0),
          show.legend = TRUE
        ) +
        ggraph::geom_node_point(size = 4, color = "#2c3e50") +
        ggraph::geom_node_text(ggplot2::aes(label = name), repel = TRUE) +
        ggraph::scale_edge_color_manual(values = c("TRUE" = "#1b9e77", "FALSE" = "#d95f02")) +
        ggraph::scale_edge_alpha(range = c(0.2, edge_alpha)) +
        ggplot2::labs(
          title = "CASTLE learned causal graph",
          edge_width = "|weight|",
          edge_alpha = "|weight|",
          edge_color = "sign"
        ) +
        ggplot2::theme_void()
    )
  }

  e_w <- igraph::E(g)$weight
  igraph::plot.igraph(
    g,
    edge.width = 1 + 4 * abs(e_w) / max(1e-8, max(abs(e_w))),
    edge.color = ifelse(e_w >= 0, "#1b9e77", "#d95f02"),
    vertex.color = "#2c3e50",
    vertex.label.color = "black",
    ...
  )
  invisible(g)
}

# --- causalStructureML integration (NOTEARS, DAG-GNN, GraN-DAG) ---

.deepnet_or <- function(x, y) if (is.null(x)) y else x

# --- DAGMA (Directed Acyclic Graphs via M-matrices) ---

.dagma_logistic_loss <- function(R, X) {
  # Stable log(1 + exp(R)) - X * R elementwise.
  a <- pmax(R, 0)
  mean(a + log1p(exp(-abs(R))) - X * R)
}

.dagma_make_s_sequence <- function(s, T) {
  if (is.list(s)) s <- unlist(s, use.names = FALSE)
  if (length(s) == 1L) return(rep(as.numeric(s), as.integer(T)))
  s <- as.numeric(s)
  if (length(s) < T) {
    s <- c(s, rep(s[length(s)], T - length(s)))
  }
  s
}

#' DagmaMLP module constructor
#'
#' Builds a torch module that parameterizes one structural equation MLP per node,
#' mirroring DAGMA's nonlinear architecture.
#'
#' @param dims Integer vector of layer sizes. Must satisfy \code{dims[1] = d} and \code{dims[length(dims)] = 1}.
#' @param bias Logical; include biases in all layers.
#' @param dtype torch dtype, default \code{torch::torch_float64()}.
#' @return A torch \code{nn_module} generator.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # DagmaMLP(...)
#' }
#' @export
DagmaMLP <- function(dims = c(10L, 10L, 1L), bias = TRUE, dtype = torch::torch_float64()) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("DagmaMLP() requires package 'torch'.", call. = FALSE)
  dims <- as.integer(dims)
  if (length(dims) < 2L || dims[length(dims)] != 1L)
    stop("dims must have length >= 2 and end with 1.", call. = FALSE)
  d <- dims[1]

  .dagma_locally_connected <- torch::nn_module(
    "DagmaLocallyConnected",
    initialize = function(num_linear, input_features, output_features, bias = TRUE) {
      self$num_linear <- as.integer(num_linear)
      self$input_features <- as.integer(input_features)
      self$output_features <- as.integer(output_features)
      k <- 1 / self$input_features
      bound <- sqrt(k)
      self$weight <- torch::nn_parameter(
        torch::torch_empty(c(self$num_linear, self$input_features, self$output_features), dtype = dtype)
      )
      torch::nn_init_uniform_(self$weight, -bound, bound)
      if (isTRUE(bias)) {
        self$bias <- torch::nn_parameter(
          torch::torch_empty(c(self$num_linear, self$output_features), dtype = dtype)
        )
        torch::nn_init_uniform_(self$bias, -bound, bound)
      } else {
        self$bias <- NULL
      }
    },
    forward = function(input) {
      out <- torch::torch_matmul(input$unsqueeze(3L), self$weight$unsqueeze(1L))$squeeze(3L)
      if (!is.null(self$bias)) out <- out + self$bias
      out
    }
  )

  torch::nn_module(
    "DagmaMLP",
    initialize = function() {
      self$dims <- dims
      self$d <- as.integer(d)
      self$I <- torch::torch_eye(self$d, dtype = dtype)
      self$fc1 <- torch::nn_linear(self$d, self$d * self$dims[2], bias = bias, dtype = dtype)
      torch::nn_init_zeros_(self$fc1$weight)
      if (isTRUE(bias) && !is.null(self$fc1$bias)) torch::nn_init_zeros_(self$fc1$bias)
      layers <- list()
      if (length(self$dims) > 2L) {
        for (l in seq_len(length(self$dims) - 2L)) {
          layers[[length(layers) + 1L]] <- .dagma_locally_connected(
            self$d, self$dims[l + 1L], self$dims[l + 2L], bias = bias
          )
        }
      }
      self$fc2 <- torch::nn_module_list(layers)
    },
    forward = function(x) {
      x <- self$fc1(x)
      x <- x$view(c(-1L, self$dims[1], self$dims[2]))
      if (length(self$fc2) > 0L) {
        for (i in seq_along(self$fc2)) {
          x <- torch::torch_sigmoid(x)
          x <- self$fc2[[i]](x)
        }
      }
      x$squeeze(3L)
    },
    h_func = function(s = 1.0) {
      fc1_weight <- self$fc1$weight$view(c(self$d, -1L, self$d))
      A <- torch::torch_sum(fc1_weight^2, dim = 2L)$transpose(1L, 2L)
      M <- s * self$I - A
      -torch::torch_logdet(M) + self$d * log(s)
    },
    fc1_l1_reg = function() {
      torch::torch_sum(torch::torch_abs(self$fc1$weight))
    },
    fc1_to_adj = function() {
      torch::with_no_grad({
        fc1_weight <- self$fc1$weight$view(c(self$d, -1L, self$d))
        A <- torch::torch_sum(fc1_weight^2, dim = 2L)$transpose(1L, 2L)
        as.matrix(torch::torch_sqrt(A)$to(device = "cpu"))
      })
    }
  )
}

#' DAGMA for linear SEMs
#'
#' R translation of DAGMA's linear optimizer (\code{DagmaLinear}) for \code{"l2"} and \code{"logistic"} losses.
#'
#' @param X Numeric matrix/data.frame of shape \eqn{n \times d}.
#' @param loss_type One of \code{"l2"} or \code{"logistic"}.
#' @param lambda1 L1 penalty on adjacency.
#' @param w_threshold Threshold for pruning small edges.
#' @param T Number of outer DAGMA iterations.
#' @param mu_init Initial \code{mu}.
#' @param mu_factor Multiplicative decay for \code{mu}.
#' @param s Scalar or vector controlling M-matrix domain.
#' @param warm_iter Inner iterations for \eqn{t < T}.
#' @param max_iter Inner iterations for \eqn{t = T}.
#' @param lr Adam learning rate.
#' @param checkpoint Convergence check interval.
#' @param beta_1 Adam beta1.
#' @param beta_2 Adam beta2.
#' @param exclude_edges Optional two-column matrix/data.frame of 1-based edges to force zero.
#' @param include_edges Optional two-column matrix/data.frame of 1-based edges to encourage inclusion.
#' @param tol Relative objective tolerance.
#' @param verbose Print optimization diagnostics.
#' @return Object of class \code{dagma} with weighted adjacency in \code{$adjacency}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # dagmaLinear(...)
#' }
#' @export
dagmaLinear <- function(X,
                        loss_type = c("l2", "logistic"),
                        lambda1 = 0.03,
                        w_threshold = 0.3,
                        T = 5L,
                        mu_init = 1.0,
                        mu_factor = 0.1,
                        s = c(1.0, 0.9, 0.8, 0.7, 0.6),
                        warm_iter = 3e4,
                        max_iter = 6e4,
                        lr = 3e-4,
                        checkpoint = 1000L,
                        beta_1 = 0.99,
                        beta_2 = 0.999,
                        exclude_edges = NULL,
                        include_edges = NULL,
                        tol = 1e-6,
                        verbose = FALSE) {
  X <- as.matrix(X)
  if (!is.numeric(X)) stop("X must be numeric.", call. = FALSE)
  n <- nrow(X)
  d <- ncol(X)
  loss_type <- match.arg(loss_type)
  Id <- diag(d)
  X_work <- X
  if (loss_type == "l2") {
    X_work <- scale(X_work, center = TRUE, scale = FALSE)
  }
  cov <- crossprod(X_work) / n
  W_est <- matrix(0, d, d)
  s_seq <- .dagma_make_s_sequence(s, as.integer(T))

  mask_exc <- matrix(1, d, d)
  diag(mask_exc) <- 0
  if (!is.null(exclude_edges)) {
    ex <- as.matrix(exclude_edges)
    if (ncol(ex) != 2L) stop("exclude_edges must have 2 columns (from, to).", call. = FALSE)
    mask_exc[cbind(ex[, 1], ex[, 2])] <- 0
  }

  score_and_grad <- function(W, mu) {
    if (loss_type == "l2") {
      G <- -mu * cov %*% (Id - W)
      dif <- Id - W
      rhs <- cov %*% dif
      loss <- 0.5 * sum(diag(t(dif) %*% rhs))
      return(list(loss = loss, grad = G))
    }
    R <- X_work %*% W
    loss <- .dagma_logistic_loss(R, X_work)
    S <- 1 / (1 + exp(-R))
    G <- (mu / n) * crossprod(X_work, S) - mu * cov
    list(loss = loss, grad = G)
  }

  h_fun <- function(W, s_cur) {
    M <- s_cur * Id - W * W
    detM <- determinant(M, logarithm = TRUE)
    if (!isTRUE(detM$sign > 0)) return(list(h = Inf, invMt = matrix(NA_real_, d, d)))
    invM <- tryCatch(solve(M), error = function(e) NULL)
    if (is.null(invM)) return(list(h = Inf, invMt = matrix(NA_real_, d, d)))
    list(h = -as.numeric(detM$modulus) + d * log(s_cur), invMt = t(invM))
  }

  objective <- function(W, mu, s_cur) {
    sc <- score_and_grad(W, mu)
    hh <- h_fun(W, s_cur)
    mu * (sc$loss + lambda1 * sum(abs(W))) + hh$h
  }

  minimize_one <- function(W, mu, inner_iter, s_cur, lr_cur) {
    obj_prev <- 1e16
    m <- matrix(0, d, d)
    v <- matrix(0, d, d)
    mask_inc <- matrix(0, d, d)
    if (!is.null(include_edges)) {
      inc <- as.matrix(include_edges)
      if (ncol(inc) != 2L) stop("include_edges must have 2 columns (from, to).", call. = FALSE)
      mask_inc[cbind(inc[, 1], inc[, 2])] <- -2 * mu * lambda1
    }

    grad_prev <- matrix(0, d, d)
    for (iter in seq_len(inner_iter)) {
      M_inv <- tryCatch(solve(s_cur * Id - W * W), error = function(e) NULL)
      if (is.null(M_inv) || any(M_inv < 0)) {
        if (iter == 1L || s_cur <= 0.9) return(list(W = W, success = FALSE, lr = lr_cur))
        W <- W + lr_cur * grad_prev
        lr_cur <- lr_cur * 0.5
        if (lr_cur <= 1e-16) return(list(W = W, success = TRUE, lr = lr_cur))
        W <- W - lr_cur * grad_prev
        M_inv <- tryCatch(solve(s_cur * Id - W * W), error = function(e) NULL)
        if (is.null(M_inv) || any(M_inv < 0)) return(list(W = W, success = FALSE, lr = lr_cur))
      }

      sc <- score_and_grad(W, mu)
      Gobj <- sc$grad + mu * lambda1 * sign(W) + 2 * W * t(M_inv) + mask_inc * sign(W)
      m <- beta_1 * m + (1 - beta_1) * Gobj
      v <- beta_2 * v + (1 - beta_2) * (Gobj^2)
      m_hat <- m / (1 - beta_1^iter)
      v_hat <- v / (1 - beta_2^iter)
      grad <- m_hat / (sqrt(v_hat) + 1e-8)
      grad_prev <- grad
      W <- W - lr_cur * grad
      W <- W * mask_exc

      if (iter %% checkpoint == 0L || iter == inner_iter) {
        obj_new <- objective(W, mu, s_cur)
        if (verbose) {
          message("DAGMA linear iter ", iter, " | obj=", signif(obj_new, 5))
        }
        if (is.finite(obj_new) && abs((obj_prev - obj_new) / obj_prev) <= tol) break
        obj_prev <- obj_new
      }
    }
    list(W = W, success = TRUE, lr = lr_cur)
  }

  mu <- mu_init
  for (i in seq_len(as.integer(T))) {
    inner_iters <- if (i == as.integer(T)) as.integer(max_iter) else as.integer(warm_iter)
    s_cur <- s_seq[i]
    lr_cur <- lr
    success <- FALSE
    while (!success) {
      out <- minimize_one(W_est, mu, inner_iters, s_cur, lr_cur)
      W_est <- out$W
      success <- isTRUE(out$success)
      if (!success) {
        lr_cur <- out$lr * 0.5
        s_cur <- s_cur + 0.1
        if (lr_cur < 1e-12) break
      }
    }
    mu <- mu * mu_factor
  }

  W_est[abs(W_est) < w_threshold] <- 0
  diag(W_est) <- 0
  structure(
    list(
      method = "dagma_linear",
      adjacency = W_est,
      loss_type = loss_type,
      lambda1 = lambda1
    ),
    class = c("dagma", "list")
  )
}

.dagma_fit_nonlinear <- function(X,
                                 model,
                                 lambda1 = 0.02,
                                 lambda2 = 0.005,
                                 T = 4L,
                                 mu_init = 0.1,
                                 mu_factor = 0.1,
                                 s = 1.0,
                                 warm_iter = 5e4,
                                 max_iter = 8e4,
                                 lr = 2e-4,
                                 w_threshold = 0.3,
                                 checkpoint = 1000L,
                                 tol = 1e-6,
                                 verbose = FALSE) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("Nonlinear dagma() requires package 'torch'.", call. = FALSE)
  X <- as.matrix(X)
  n <- nrow(X)
  d <- ncol(X)
  X_t <- torch::torch_tensor(X, dtype = torch::torch_float64())
  s_seq <- .dagma_make_s_sequence(s, as.integer(T))
  mu <- mu_init

  minimize_one <- function(inner_iter, lr_cur, mu_cur, s_cur, lr_decay = FALSE) {
    optimizer <- torch::optim_adam(
      model$parameters,
      lr = lr_cur,
      betas = c(0.99, 0.999),
      weight_decay = mu_cur * lambda2
    )
    scheduler <- NULL
    if (isTRUE(lr_decay)) {
      # torch scheduler APIs differ by version; skip scheduler when unavailable.
      scheduler <- NULL
    }
    obj_prev <- 1e16
    for (iter in seq_len(as.integer(inner_iter))) {
      optimizer$zero_grad()
      h_val <- model$h_func(s_cur)
      h_num <- as.numeric(h_val)
      if (!is.finite(h_num) || h_num < 0) return(FALSE)
      X_hat <- model(X_t)
      mse <- torch::torch_mean((X_hat - X_t)^2)
      score <- 0.5 * d * torch::torch_log(mse + 1e-16)
      l1_reg <- lambda1 * model$fc1_l1_reg()
      obj <- mu_cur * (score + l1_reg) + h_val
      obj$backward()
      optimizer$step()
      if (!is.null(scheduler) && iter %% 1000L == 0L) scheduler$step()

      if (iter %% checkpoint == 0L || iter == as.integer(inner_iter)) {
        obj_new <- as.numeric(obj)
        if (verbose) {
          message("DAGMA nonlinear iter ", iter, " | obj=", signif(obj_new, 5))
        }
        if (is.finite(obj_new) && abs((obj_prev - obj_new) / obj_prev) <= tol) break
        obj_prev <- obj_new
      }
    }
    TRUE
  }

  for (i in seq_len(as.integer(T))) {
    if (verbose) message("DAGMA nonlinear outer iter ", i, " | mu=", signif(mu, 4))
    inner_iter <- if (i == as.integer(T)) as.integer(max_iter) else as.integer(warm_iter)
    s_cur <- s_seq[i]
    success <- FALSE
    state <- model$state_dict()
    state_copy <- lapply(state, function(t) t$clone())
    lr_cur <- lr
    lr_decay <- FALSE
    while (!success) {
      success <- minimize_one(inner_iter, lr_cur, mu, s_cur, lr_decay = lr_decay)
      if (!success) {
        model$load_state_dict(state_copy)
        lr_cur <- lr_cur * 0.5
        lr_decay <- TRUE
        s_cur <- 1.0
        if (lr_cur < 1e-10) break
      }
    }
    mu <- mu * mu_factor
  }

  W_est <- model$fc1_to_adj()
  W_est[abs(W_est) < w_threshold] <- 0
  diag(W_est) <- 0
  structure(
    list(
      method = "dagma_nonlinear_mlp",
      adjacency = W_est,
      model = model
    ),
    class = c("dagma", "list")
  )
}

#' DAGMA unified interface
#'
#' Wrapper for DAGMA linear and nonlinear-MLP causal discovery translated from
#' \code{dagma-main/src/dagma} Python implementation.
#'
#' @param X Numeric matrix/data.frame (\eqn{n \times d}).
#' @param method \code{"linear"} or \code{"nonlinear_mlp"}.
#' @param ... Additional arguments passed to \code{dagmaLinear()} or nonlinear optimizer.
#' @return Object of class \code{dagma} with \code{$adjacency}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # dagma(...)
#' }
#' @export
dagma <- function(X, method = c("linear", "nonlinear_mlp"), ...) {
  method <- match.arg(method)
  dots <- list(...)
  X <- as.matrix(X)
  if (!is.numeric(X)) stop("X must be numeric.", call. = FALSE)
  if (method == "linear") {
    return(do.call(dagmaLinear, c(list(X = X), dots)))
  }
  if (!requireNamespace("torch", quietly = TRUE))
    stop("method='nonlinear_mlp' requires package 'torch'.", call. = FALSE)
  model <- dots$model
  dots$model <- NULL
  if (is.null(model)) {
    hidden <- if (!is.null(dots$hidden)) as.integer(dots$hidden) else 10L
    dots$hidden <- NULL
    bias <- if (!is.null(dots$bias)) isTRUE(dots$bias) else TRUE
    dots$bias <- NULL
    Model <- DagmaMLP(dims = c(ncol(X), hidden, 1L), bias = bias)
    model <- Model()
  }
  do.call(.dagma_fit_nonlinear, c(list(X = X, model = model), dots))
}

#' Short descriptions of causal structure learning methods
#'
#' Returns one-line summaries of each \code{\link{causalStructureML}} \code{method}.
#'
#' @return Named character vector.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # causal_structure_ml_model_descriptions(...)
#' }
#' @export
causal_structure_ml_model_descriptions <- function() {
  c(
    notears_linear = paste0(
      "Linear NOTEARS (Zheng et al.): L1-regularized linear SEM fit with smooth ",
      "acyclicity constraint trace(exp(W*W)) - d, augmented Lagrangian + L-BFGS-B; ",
      "uses expm and igraph (no torch)."
    ),
    notears_nonlinear_mlp = paste0(
      "Nonlinear NOTEARS: per-node MLP, same trace(exp(A))-d constraint with ",
      "custom autograd; Adam then L-BFGS. Requires torch."
    ),
    notears_nonlinear_sobolev = paste0(
      "Nonlinear NOTEARS (Sobolev basis features instead of MLP); ",
      "Adam + L-BFGS. Requires torch."
    ),
    dag_gnn = paste0(
      "DAG-GNN (Yu et al., ICML 2019): VAE-style encoder-decoder, learnable A, ",
      "matrix-power DAG penalty h(A), augmented Lagrangian. Requires torch."
    ),
    grandag = paste0(
      "GraN-DAG: neural additive-noise likelihood, trace(exp|A|)-d constraint, ",
      "optional PNS via ranger; R6 learner. Requires torch, R6, expm."
    )
  )
}

#' @keywords internal
.dag_gnn_fit <- function(X, n_epochs = 400L, lr = 1e-3, hidden_dim = 64L,
                         adj_threshold = 0.10, verbose = FALSE, seed = NULL,
                         device = NULL) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("Package 'torch' is required for method 'dag_gnn'.", call. = FALSE)

  if (!is.null(seed)) {
    set.seed(seed)
    torch::torch_manual_seed(as.integer(seed))
  }

  d <- ncol(X)
  if (is.null(device)) {
    device <- torch::torch_device(
      if (torch::cuda_is_available()) "cuda" else "cpu"
    )
    if (verbose)
      message("DAG-GNN device: ", device$type)
  }

  X_t <- torch::torch_tensor(
    X,
    dtype = torch::torch_float32(),
    device = device
  )

  model <- make_daggnn(
    n_nodes = as.integer(d),
    hidden_dim = as.integer(hidden_dim),
    device = device
  )

  optimizer <- torch::optim_adam(model$parameters, lr = lr)

  lambda_h <- 0.0
  c_pen <- 1.0
  eta <- 10.0
  gamma <- 0.25
  max_c <- 1e6
  c_update_freq <- 25L
  h_prev <- NULL
  final_loss <- NA_real_

  for (epoch in seq_len(as.integer(n_epochs))) {
    model$train()
    optimizer$zero_grad()

    out <- tryCatch(model$forward(X_t), error = function(e) NULL)
    if (is.null(out)) next

    loss <- model$elbo_loss(X_t, out$MX, out$MZ)
    h_val <- model$h_func()

    loss_num <- as.numeric(loss)
    h_num <- as.numeric(h_val)
    if (!is.finite(loss_num)) next

    huber_reg <- torch::torch_mean(
      torch::torch_sqrt((model$A)^2 + 1e-4) - 1e-2
    )

    if (is.finite(h_num)) {
      aug <- loss + lambda_h * h_val + 0.5 * c_pen * (h_val^2) + 1e-4 * huber_reg
    } else {
      aug <- loss + 1e-4 * huber_reg
    }

    aug$backward()
    torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
    optimizer$step()

    if (is.finite(h_num)) lambda_h <- lambda_h + c_pen * h_num

    if (epoch %% c_update_freq == 0L && is.finite(h_num)) {
      if (!is.null(h_prev) && is.finite(h_prev) &&
          abs(h_num) > gamma * abs(h_prev))
        c_pen <- min(eta * c_pen, max_c)
      h_prev <- h_num
    }

    final_loss <- loss_num
    if (verbose && epoch %% max(1L, n_epochs %/% 10L) == 0L) {
      message(sprintf(
        "DAG-GNN epoch %d | loss %.4f | h %.6f | c %.2e",
        epoch, loss_num, if (is.finite(h_num)) h_num else NA_real_, c_pen
      ))
    }
  }

  A_hat <- daggnn_adj(model, threshold = adj_threshold)
  list(model = model, adjacency = A_hat, final_loss = final_loss)
}

#' Causal structure learning (unified API)
#'
#' Single entry point for \strong{NOTEARS} (linear and nonlinear), \strong{DAG-GNN},
#' and \strong{GraN-DAG} implemented in this package. See
#' \code{\link{causal_structure_ml_model_descriptions}} for concise model summaries.
#'
#' @param data Numeric matrix or data.frame (\eqn{n \times d}).
#' @param method Character: \code{"notears_linear"}, \code{"notears_nonlinear_mlp"},
#'   \code{"notears_nonlinear_sobolev"}, \code{"dag_gnn"}, or \code{"grandag"}.
#' @param ... Method-specific arguments (see below).
#'
#' @section Method-specific \code{...} arguments:
#' \describe{
#'   \item{\code{notears_linear}}{Passed to \code{\link{notears_linear}} (e.g.
#'     \code{lambda1}, \code{loss_type}, \code{max_iter}, \code{w_threshold}).}
#'   \item{\code{notears_nonlinear_mlp}, \code{notears_nonlinear_sobolev}}{Passed to
#'     \code{\link{notears_nonlinear}} after building the model. Use \code{notears_hidden}
#'     (integer, default 10) for MLP hidden units; use \code{sobolev_k} (integer, default 5)
#'     for Sobolev \code{k}.}
#'   \item{\code{dag_gnn}}{\code{n_epochs} (default 400), \code{lr}, \code{hidden_dim},
#'     \code{adj_threshold} (passed to \code{\link{daggnn_adj}}), \code{verbose},
#'     \code{seed}, \code{device} (a \code{torch_device} or NULL for auto).}
#'   \item{\code{grandag}}{Passed to \code{GraNDAG$new()} after \code{input_dim} is set
#'     (e.g. \code{iterations}, \code{lr}, \code{batch_size}, \code{model_name},
#'     \code{device_type}, \code{use_pns}). Then \code{learn()} is called on \code{data}.}
#' }
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{method}}{Character method name.}
#'   \item{\code{description}}{Short text from \code{\link{causal_structure_ml_model_descriptions}}.}
#'   \item{\code{adjacency}}{Estimated weighted adjacency (\eqn{d \times d}). Orientation matches
#'     the underlying implementation (for linear NOTEARS vs ground-truth DAGs from this package,
#'     you may compare with \code{t(adjacency)}; see NOTEARS demos).}
#'   \item{\code{binary_adjacency}}{0/1 matrix, nonzero where \code{abs(adjacency)} is positive.}
#'   \item{\code{fit}}{Method-specific object: \code{NULL} (linear NOTEARS), torch model
#'     (nonlinear NOTEARS / DAG-GNN), or \code{\link{GraNDAG}} R6 instance (\code{grandag}).}
#'   \item{\code{extra}}{Named list of extras (e.g. \code{final_loss} for DAG-GNN,
#'     \code{train_losses} attribute copied for nonlinear NOTEARS when present).}
#' }
#'
#' @seealso \code{\link{notears_linear}}, \code{\link{notears_nonlinear}},
#'   \code{\link{make_daggnn}}, \code{\link{daggnn_adj}}, \code{\link{GraNDAG}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # causalStructureML(...)
#' }
#' @export
causalStructureML <- function(
    data,
    method = c(
      "notears_linear",
      "notears_nonlinear_mlp",
      "notears_nonlinear_sobolev",
      "dag_gnn",
      "grandag"
    ),
    ...) {

  method <- match.arg(method)
  desc <- causal_structure_ml_model_descriptions()

  X <- as.matrix(data)
  if (!is.numeric(X)) stop("'data' must be numeric.", call. = FALSE)
  cn <- colnames(X)
  if (is.null(cn)) {
    cn <- paste0("V", seq_len(ncol(X)))
    colnames(X) <- cn
  }

  dots <- list(...)
  extra <- list()
  fit <- NULL

  block <- switch(
    method,
    notears_linear = {
      W <- do.call(notears_linear, c(list(X = X), dots))
      list(adj = W, fit = NULL, extra = list())
    },
    notears_nonlinear_mlp = {
      if (!requireNamespace("torch", quietly = TRUE))
        stop("Package 'torch' is required for nonlinear NOTEARS.", call. = FALSE)
      nh <- if (!is.null(dots$notears_hidden)) dots$notears_hidden else 10L
      dots$notears_hidden <- NULL
      Mod <- get0(
        "NotearsMLP",
        envir = asNamespace("RCausalML"),
        inherits = FALSE,
        ifnotfound = NULL
      )
      if (is.null(Mod))
        stop(
          "NotearsMLP not available (is torch installed and notears.R loaded?).",
          call. = FALSE
        )
      model <- Mod(d = ncol(X), hidden = as.integer(nh))
      W <- do.call(notears_nonlinear, c(list(model = model, X = X), dots))
      ex <- list()
      tl <- attr(W, "train_losses")
      if (!is.null(tl)) ex$train_losses <- tl
      list(adj = W, fit = model, extra = ex)
    },
    notears_nonlinear_sobolev = {
      if (!requireNamespace("torch", quietly = TRUE))
        stop("Package 'torch' is required for nonlinear NOTEARS.", call. = FALSE)
      sk <- if (!is.null(dots$sobolev_k)) dots$sobolev_k else 5L
      dots$sobolev_k <- NULL
      Mod <- get0(
        "NotearsSobolev",
        envir = asNamespace("RCausalML"),
        inherits = FALSE,
        ifnotfound = NULL
      )
      if (is.null(Mod))
        stop("NotearsSobolev not available.", call. = FALSE)
      model <- Mod(d = ncol(X), k = as.integer(sk))
      W <- do.call(notears_nonlinear, c(list(model = model, X = X), dots))
      ex <- list()
      tl <- attr(W, "train_losses")
      if (!is.null(tl)) ex$train_losses <- tl
      list(adj = W, fit = model, extra = ex)
    },
    dag_gnn = {
      n_epochs <- .deepnet_or(dots$n_epochs, 400L)
      lr <- .deepnet_or(dots$lr, 1e-3)
      hidden_dim <- .deepnet_or(dots$hidden_dim, 64L)
      adj_threshold <- .deepnet_or(dots$adj_threshold, 0.10)
      verbose <- isTRUE(dots$verbose)
      seed <- dots$seed
      device <- dots$device
      dots$n_epochs <- dots$lr <- dots$hidden_dim <- dots$adj_threshold <-
        dots$verbose <- dots$seed <- dots$device <- NULL
      if (length(dots))
        warning("Unused arguments for dag_gnn: ", paste(names(dots), collapse = ", "),
                call. = FALSE)

      dg <- .dag_gnn_fit(
        X,
        n_epochs = n_epochs,
        lr = lr,
        hidden_dim = hidden_dim,
        adj_threshold = adj_threshold,
        verbose = verbose,
        seed = seed,
        device = device
      )
      list(
        adj = dg$adjacency,
        fit = dg$model,
        extra = list(final_loss = dg$final_loss)
      )
    },
    grandag = {
      dots$input_dim <- NULL
      learner <- do.call(
        GraNDAG$new,
        c(list(input_dim = ncol(X)), dots)
      )
      learner$learn(X, columns = cn)
      list(adj = learner$get_causal_matrix(), fit = learner, extra = list())
    }
  )

  adj <- block$adj
  fit <- block$fit
  extra <- modifyList(extra, block$extra)

  if (!is.null(cn)) {
    if (is.null(rownames(adj))) rownames(adj) <- cn
    if (is.null(colnames(adj))) colnames(adj) <- cn
  }

  bin <- (abs(adj) > 0) + 0L
  dimnames(bin) <- dimnames(adj)

  structure(
    list(
      method = method,
      description = unname(desc[[method]]),
      adjacency = adj,
      binary_adjacency = bin,
      fit = fit,
      extra = extra
    ),
    class = c("causal_structure_ml", "list")
  )
}

# --- iVAE and CausalVAE family (integrated from standalone scripts) ---

.deepnet_select_device <- function(device = NULL) {
  if (!requireNamespace("torch", quietly = TRUE)) return("cpu")
  if (is.null(device)) return(if (torch::cuda_is_available()) "cuda" else "cpu")
  as.character(device)
}

.deepnet_ivae_module <- function(input_dim, latent_dim, hidden_dim, n_aux, dropout) {
  torch::nn_module(
    "iVAE",
    initialize = function() {
      self$enc_fc1 <- torch::nn_linear(input_dim + n_aux, hidden_dim)
      self$enc_fc2 <- torch::nn_linear(hidden_dim, hidden_dim)
      self$enc_mu <- torch::nn_linear(hidden_dim, latent_dim)
      self$enc_logvar <- torch::nn_linear(hidden_dim, latent_dim)
      self$dec_fc1 <- torch::nn_linear(latent_dim, hidden_dim)
      self$dec_fc2 <- torch::nn_linear(hidden_dim, hidden_dim)
      self$dec_mu <- torch::nn_linear(hidden_dim, input_dim)
      self$dec_logvar <- torch::nn_linear(hidden_dim, input_dim)
      self$prior_fc <- torch::nn_linear(n_aux, latent_dim)
      self$n_aux <- as.integer(n_aux)
      self$dropout <- torch::nn_dropout(p = dropout)
    },
    .onehot = function(u, dev) {
      torch::nnf_one_hot(u$to(dtype = torch::torch_long()), num_classes = self$n_aux)$to(
        dtype = torch::torch_float32(),
        device = dev
      )
    },
    encode = function(x, u) {
      u_onehot <- self$.onehot(u, x$device)
      h <- torch::nnf_relu(self$enc_fc1(torch::torch_cat(list(x, u_onehot), dim = 2L)))
      h <- self$dropout(torch::nnf_relu(self$enc_fc2(h)))
      list(mu = self$enc_mu(h), logvar = self$enc_logvar(h)$clamp(min = -10, max = 10))
    },
    reparameterize = function(mu, logvar) mu + torch::torch_exp(0.5 * logvar) * torch::torch_randn_like(mu),
    decode = function(z) {
      h <- self$dropout(torch::nnf_relu(self$dec_fc1(z)))
      h <- self$dropout(torch::nnf_relu(self$dec_fc2(h)))
      list(mu = self$dec_mu(h), logvar = self$dec_logvar(h)$clamp(min = -10, max = 10))
    },
    prior = function(u, dev) {
      u_onehot <- self$.onehot(u, dev)
      mu <- self$prior_fc(u_onehot)
      list(mu = mu, logvar = torch::torch_zeros_like(mu))
    },
    forward = function(x, u) {
      enc <- self$encode(x, u)
      z <- self$reparameterize(enc$mu, enc$logvar)
      dec <- self$decode(z)
      pri <- self$prior(u, x$device)
      list(enc_mu = enc$mu, enc_logvar = enc$logvar, dec_mu = dec$mu, dec_logvar = dec$logvar,
           prior_mu = pri$mu, prior_logvar = pri$logvar, z = z)
    }
  )
}

.deepnet_ivae_loss <- function(x, out) {
  dec_std <- torch::torch_exp(0.5 * out$dec_logvar)$clamp(min = 1e-4)
  recon <- torch::distr_normal(out$dec_mu, dec_std)$log_prob(x)$sum(dim = 2L)$mean()
  kl_elem <- 0.5 * (
    out$prior_logvar - out$enc_logvar +
      (torch::torch_exp(out$enc_logvar) + (out$enc_mu - out$prior_mu)^2) / torch::torch_exp(out$prior_logvar) - 1
  )
  kl <- kl_elem$sum(dim = 2L)$mean()
  -(recon - kl)
}

#' iVAE (Identifiable Variational Autoencoder)
#'
#' Fits an iVAE model with auxiliary variable \code{u} and predicts latent representation \eqn{z}.
#' This is a package-safe integration of \code{causal_iVAE.R} without load-time side effects.
#'
#' @param X Covariate matrix or data.frame.
#' @param u Auxiliary variable (integer labels from 0 to \code{n_aux - 1}).
#' @param latent_dim Latent dimension (default 2).
#' @param hidden_dim Hidden dimension (default 128).
#' @param n_aux Number of auxiliary classes (default inferred from \code{u}).
#' @param dropout Dropout probability (default 0.15).
#' @param num_epochs Number of epochs (default 100).
#' @param batch_size Batch size (default 128).
#' @param learning_rate Learning rate (default 1e-3).
#' @param verbose If TRUE, prints training loss every 10 epochs.
#' @param device Device string: \code{"cuda"}, \code{"cpu"}, or \code{NULL}.
#' @param ... Ignored.
#' @return Object of class \code{ivae}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # ivae(...)
#' }
#' @export
ivae <- function(X, u,
                 latent_dim = 2L,
                 hidden_dim = 128L,
                 n_aux = NULL,
                 dropout = 0.15,
                 num_epochs = 100L,
                 batch_size = 128L,
                 learning_rate = 1e-3,
                 verbose = TRUE,
                 device = NULL,
                 ...) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("ivae() requires package 'torch'.")
  X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  u <- as.integer(u)
  if (is.null(n_aux)) n_aux <- max(u, na.rm = TRUE) + 1L
  if (any(u < 0L) || any(u >= n_aux))
    stop("u must be integer labels in [0, n_aux - 1].")
  dev <- .deepnet_select_device(device)
  Module <- .deepnet_ivae_module(as.integer(p), as.integer(latent_dim), as.integer(hidden_dim), as.integer(n_aux), dropout)
  model <- Module()
  model$to(device = dev)
  x_t <- torch::torch_tensor(X, dtype = torch::torch_float32(), device = dev)
  u_t <- torch::torch_tensor(u, dtype = torch::torch_long(), device = dev)
  opt <- torch::optim_adam(model$parameters, lr = learning_rate)
  batch_size <- min(as.integer(batch_size), n)
  n_batches <- max(1L, ceiling(n / batch_size))
  for (epoch in seq_len(as.integer(num_epochs))) {
    model$train(TRUE)
    epoch_loss <- 0
    perm <- sample.int(n)
    for (start in seq(1L, n, by = batch_size)) {
      idx <- perm[start:min(start + batch_size - 1L, n)]
      opt$zero_grad()
      out <- model(x_t[idx, , drop = FALSE], u_t[idx])
      loss <- .deepnet_ivae_loss(x_t[idx, , drop = FALSE], out)
      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      opt$step()
      epoch_loss <- epoch_loss + as.numeric(loss)
    }
    if (verbose && epoch %% 10L == 0L) {
      message("iVAE epoch ", epoch, " loss: ", round(epoch_loss / n_batches, 4))
    }
  }
  structure(list(model = model, X_names = colnames(X), type = "ivae_torch", device = dev, n_aux = n_aux),
            class = "ivae")
}

#' Predict latent representation from iVAE
#'
#' @param object Fitted \code{ivae} object.
#' @param newdata Covariate matrix or data.frame.
#' @param u Auxiliary labels in \code{[0, n_aux - 1]}.
#' @param ... Ignored.
#' @return Matrix of latent means.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.ivae(...)
#' }
#' @export
predict.ivae <- function(object, newdata, u, ...) {
  newdata <- as.matrix(newdata)
  u <- as.integer(u)
  if (nrow(newdata) != length(u)) stop("newdata and u must have matching length.")
  if (any(u < 0L) || any(u >= object$n_aux)) stop("u must be in [0, n_aux - 1].")
  model <- object$model
  model$eval()
  torch::with_no_grad({
    x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = object$device)
    u_t <- torch::torch_tensor(u, dtype = torch::torch_long(), device = object$device)
    out <- model(x_t, u_t)
    as.matrix(out$enc_mu$to(device = "cpu"))
  })
}

.deepnet_causalvae_module <- function(input_dim, latent_dim, hidden_dim, beta, gamma, lambda_sparsity, alpha) {
  torch::nn_module(
    "CausalVAE",
    initialize = function() {
      self$beta <- beta
      self$gamma <- gamma
      self$lambda_sparsity <- lambda_sparsity
      self$alpha <- alpha
      self$latent_dim <- as.integer(latent_dim)
      self$enc_fc1 <- torch::nn_linear(input_dim, hidden_dim)
      self$enc_fc2 <- torch::nn_linear(hidden_dim, hidden_dim)
      self$enc_mu <- torch::nn_linear(hidden_dim, latent_dim)
      self$enc_logvar <- torch::nn_linear(hidden_dim, latent_dim)
      self$dec_fc1 <- torch::nn_linear(latent_dim, hidden_dim)
      self$dec_fc2 <- torch::nn_linear(hidden_dim, hidden_dim)
      self$dec_mu <- torch::nn_linear(hidden_dim, input_dim)
      self$dec_logvar <- torch::nn_linear(hidden_dim, input_dim)
      self$a_weights <- torch::nn_parameter(torch::torch_randn(latent_dim, latent_dim) * 0.01)
      self$m_logits <- torch::nn_parameter(torch::torch_randn(latent_dim, latent_dim) * 0.01)
    },
    encode = function(x) {
      h <- torch::nnf_relu(self$enc_fc2(torch::nnf_relu(self$enc_fc1(x))))
      list(mu = self$enc_mu(h), logvar = self$enc_logvar(h)$clamp(min = -10, max = 10))
    },
    reparameterize = function(mu, logvar) mu + torch::torch_exp(0.5 * logvar) * torch::torch_randn_like(mu),
    causal_layer = function(epsilon) {
      AM <- torch::torch_tanh(self$a_weights) * torch::torch_sigmoid(self$m_logits)
      I <- torch::torch_eye(self$latent_dim, device = epsilon$device, dtype = epsilon$dtype)
      tryCatch(torch::linalg_solve(I - AM$t(), epsilon$t())$t(),
               error = function(e) epsilon + torch::torch_matmul(epsilon, AM))
    },
    decode = function(z) {
      h <- torch::nnf_relu(self$dec_fc2(torch::nnf_relu(self$dec_fc1(z))))
      list(mu = self$dec_mu(h), logvar = self$dec_logvar(h)$clamp(min = -10, max = 10))
    },
    dag_penalty = function() {
      M <- torch::torch_sigmoid(self$m_logits)
      eye <- torch::torch_eye(self$latent_dim, device = M$device, dtype = M$dtype)
      torch::torch_trace(torch::torch_matrix_exp(eye + self$alpha * M * M)) - self$latent_dim
    },
    forward = function(x) {
      enc <- self$encode(x)
      eps <- self$reparameterize(enc$mu, enc$logvar)
      z <- self$causal_layer(eps)
      dec <- self$decode(z)
      list(enc_mu = enc$mu, enc_logvar = enc$logvar, dec_mu = dec$mu, dec_logvar = dec$logvar, z = z)
    }
  )
}

.deepnet_causalvae_loss <- function(x, out, model, gamma_scale = 1) {
  dec_std <- torch::torch_exp(0.5 * out$dec_logvar)$clamp(min = 1e-4)
  recon <- (-0.5 * ((x - out$dec_mu) / dec_std)^2 - torch::torch_log(dec_std) - 0.5 * log(2 * pi))$sum(dim = 2L)$mean()
  enc_std <- torch::torch_exp(0.5 * out$enc_logvar)$clamp(min = 1e-4)
  kl <- (0.5 * (enc_std^2 + out$enc_mu^2 - 1 - 2 * torch::torch_log(enc_std)))$sum(dim = 2L)$mean()
  dag <- model$dag_penalty()
  sparse <- torch::torch_sum(torch::torch_abs(torch::torch_sigmoid(model$m_logits)))
  causal <- gamma_scale * model$gamma * (dag + model$lambda_sparsity * sparse)
  -recon + model$beta * kl + causal
}

.deepnet_causalvae_ate_module <- function(input_dim, latent_dim, hidden_dim, beta, gamma, lambda_sparsity, alpha) {
  Base <- .deepnet_causalvae_module(input_dim, latent_dim, hidden_dim, beta, gamma, lambda_sparsity, alpha)
  torch::nn_module(
    "CausalVAE_ATE",
    inherit = Base,
    initialize = function() {
      super$initialize()
      self$outcome_head <- torch::nn_linear(as.integer(latent_dim) + 1L, 1L)
    },
    forward = function(x, T = NULL) {
      out <- super$forward(x)
      if (!is.null(T)) {
        out$y_pred <- self$outcome_head(torch::torch_cat(list(out$z, T), dim = 2L))
      }
      out
    }
  )
}

.deepnet_fit_causalvae <- function(X, latent_dim, hidden_dim, beta, gamma, lambda_sparsity, alpha,
                                   num_epochs, batch_size, learning_rate, weight_decay, warmup_epochs,
                                   verbose, device) {
  X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  dev <- .deepnet_select_device(device)
  Module <- .deepnet_causalvae_module(as.integer(p), as.integer(latent_dim), as.integer(hidden_dim),
                                      beta, gamma, lambda_sparsity, alpha)
  model <- Module()
  model$to(device = dev)
  x_t <- torch::torch_tensor(X, dtype = torch::torch_float32(), device = dev)
  opt <- torch::optim_adam(model$parameters, lr = learning_rate, weight_decay = weight_decay)
  batch_size <- min(as.integer(batch_size), n)
  n_batches <- max(1L, ceiling(n / batch_size))
  for (epoch in seq_len(as.integer(num_epochs))) {
    model$train(TRUE)
    epoch_loss <- 0
    gamma_scale <- min(1.0, epoch / max(1L, as.integer(warmup_epochs)))
    perm <- sample.int(n)
    for (start in seq(1L, n, by = batch_size)) {
      idx <- perm[start:min(start + batch_size - 1L, n)]
      x_b <- x_t[idx, , drop = FALSE]
      opt$zero_grad()
      out <- model(x_b)
      loss <- .deepnet_causalvae_loss(x_b, out, model, gamma_scale = gamma_scale)
      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      opt$step()
      epoch_loss <- epoch_loss + as.numeric(loss)
    }
    if (verbose && epoch %% 10L == 0L)
      message("CausalVAE epoch ", epoch, " loss: ", round(epoch_loss / n_batches, 4))
  }
  structure(list(model = model, X_names = colnames(X), type = "causal_vae_torch",
                 device = dev, latent_dim = latent_dim),
            class = "causal_vae")
}

#' Generate synthetic data for CausalVAE examples
#'
#' @param n_samples Number of observations.
#' @param latent_dim Latent dimension. Current generator supports 3.
#' @param device Device string.
#' @return List with tensors \code{x}, \code{z}, and \code{epsilon}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # generate_data(...)
#' }
#' @export
generate_data <- function(n_samples = 5000L, latent_dim = 3L, device = NULL) {
  latent_dim <- as.integer(latent_dim)
  if (latent_dim != 3L) stop("generate_data() currently supports latent_dim = 3.")
  dev <- .deepnet_select_device(device)
  epsilon_mat <- matrix(stats::rnorm(n_samples * latent_dim), nrow = n_samples, ncol = latent_dim)
  z_mat <- matrix(0.0, nrow = n_samples, ncol = latent_dim)
  x_mat <- matrix(0.0, nrow = n_samples, ncol = latent_dim)
  z_mat[, 1] <- epsilon_mat[, 1]
  z_mat[, 2] <- z_mat[, 1] + epsilon_mat[, 2]
  z_mat[, 3] <- z_mat[, 2]^2 + epsilon_mat[, 3]
  x_mat[, 1] <- z_mat[, 1] * z_mat[, 3]
  x_mat[, 2] <- sin(z_mat[, 2]) + z_mat[, 1]
  x_mat[, 3] <- z_mat[, 3]^2 + stats::rnorm(n_samples, 0, 0.1)
  list(
    x = torch::torch_tensor(x_mat, dtype = torch::torch_float32(), device = dev),
    z = torch::torch_tensor(z_mat, dtype = torch::torch_float32(), device = dev),
    epsilon = torch::torch_tensor(epsilon_mat, dtype = torch::torch_float32(), device = dev)
  )
}

#' Build a CausalVAE nn_module instance
#'
#' @inheritParams causal_vae
#' @return A torch \code{nn_module} instance.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # CausalVAE(...)
#' }
#' @export
CausalVAE <- function(input_dim = 3L, latent_dim = 3L, hidden_dim = 64L, beta = 1.0, gamma = 0.01,
                      lambda_sparsity = 0.001, alpha = 0.001, device = NULL) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("CausalVAE() requires package 'torch'.")
  Module <- .deepnet_causalvae_module(
    as.integer(input_dim), as.integer(latent_dim), as.integer(hidden_dim),
    beta, gamma, lambda_sparsity, alpha
  )
  dev <- .deepnet_select_device(device)
  model <- Module()
  model$to(device = dev)
  model
}

#' Build a CausalVAE model with ATE head
#'
#' @inheritParams CausalVAE
#' @return A torch \code{nn_module} instance with outcome head.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # CausalVAE_ATE(...)
#' }
#' @export
CausalVAE_ATE <- function(input_dim = 3L, latent_dim = 3L, hidden_dim = 64L, beta = 1.0, gamma = 0.01,
                          lambda_sparsity = 0.001, alpha = 0.001, device = NULL) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("CausalVAE_ATE() requires package 'torch'.")
  Module <- .deepnet_causalvae_ate_module(
    as.integer(input_dim), as.integer(latent_dim), as.integer(hidden_dim),
    beta, gamma, lambda_sparsity, alpha
  )
  dev <- .deepnet_select_device(device)
  model <- Module()
  model$to(device = dev)
  model
}

#' CausalVAE loss function
#'
#' @param x Input tensor.
#' @param dec_mu Decoder mean.
#' @param dec_logvar Decoder log variance.
#' @param enc_mu Encoder mean.
#' @param enc_logvar Encoder log variance.
#' @param model CausalVAE model instance.
#' @param gamma_scale Warmup scaling for structural loss term.
#' @return Scalar tensor loss.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # loss_function(...)
#' }
#' @export
loss_function <- function(x, dec_mu, dec_logvar, enc_mu, enc_logvar, model, gamma_scale = 1.0) {
  out <- list(
    dec_mu = dec_mu, dec_logvar = dec_logvar,
    enc_mu = enc_mu, enc_logvar = enc_logvar
  )
  .deepnet_causalvae_loss(x = x, out = out, model = model, gamma_scale = gamma_scale)
}

#' Train CausalVAE from dataloaders
#'
#' @param model CausalVAE model instance.
#' @param train_loader Training dataloader.
#' @param val_loader Validation dataloader.
#' @param epochs Number of epochs.
#' @param warmup_epochs Number of warmup epochs.
#' @param learning_rate Learning rate.
#' @param weight_decay Weight decay.
#' @param device Device string.
#' @param verbose Print progress every 10 epochs.
#' @return List with train losses, validation losses and best validation loss.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # train_causalvae(...)
#' }
#' @export
train_causalvae <- function(model, train_loader, val_loader, epochs = 200L, warmup_epochs = 40L,
                            learning_rate = 1e-4, weight_decay = 1e-5, device = NULL, verbose = TRUE) {
  if (!requireNamespace("coro", quietly = TRUE))
    stop("train_causalvae() requires package 'coro'.")
  dev <- .deepnet_select_device(device)
  optimizer <- torch::optim_adam(model$parameters, lr = learning_rate, weight_decay = weight_decay)
  scheduler <- torch::lr_reduce_on_plateau(optimizer, mode = "min", factor = 0.5, patience = 15)
  train_losses <- numeric(as.integer(epochs))
  val_losses <- numeric(as.integer(epochs))
  best_val_loss <- Inf

  for (epoch in seq_len(as.integer(epochs))) {
    model$train(TRUE)
    gamma_scale <- min(1.0, epoch / max(1L, as.integer(warmup_epochs)))
    epoch_train_loss <- 0
    train_batches <- coro::collect(train_loader)
    train_n <- 0L
    for (batch in train_batches) {
      x_b <- if (is.list(batch)) batch[[1]] else batch
      x_b <- x_b$to(device = dev)
      optimizer$zero_grad()
      out <- model$forward(x_b)
      loss <- .deepnet_causalvae_loss(x_b, out, model, gamma_scale = gamma_scale)
      lv <- loss$item()
      if (!is.finite(lv)) next
      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      optimizer$step()
      epoch_train_loss <- epoch_train_loss + lv
      train_n <- train_n + 1L
    }
    train_losses[epoch] <- if (train_n > 0L) epoch_train_loss / train_n else Inf

    model$eval()
    epoch_val_loss <- 0
    val_n <- 0L
    torch::with_no_grad({
      val_batches <- coro::collect(val_loader)
      for (batch in val_batches) {
        x_b <- if (is.list(batch)) batch[[1]] else batch
        x_b <- x_b$to(device = dev)
        out <- model$forward(x_b)
        lv <- .deepnet_causalvae_loss(x_b, out, model, gamma_scale = gamma_scale)$item()
        if (is.finite(lv)) {
          epoch_val_loss <- epoch_val_loss + lv
          val_n <- val_n + 1L
        }
      }
    })
    val_losses[epoch] <- if (val_n > 0L) epoch_val_loss / val_n else Inf
    if (is.finite(val_losses[epoch])) scheduler$step(val_losses[epoch])
    if (val_losses[epoch] < best_val_loss) best_val_loss <- val_losses[epoch]

    if (verbose && epoch %% 10L == 0L) {
      message("CausalVAE epoch ", epoch, " train=", round(train_losses[epoch], 4),
              " val=", round(val_losses[epoch], 4))
    }
  }
  list(train_losses = train_losses, val_losses = val_losses, best_val_loss = best_val_loss)
}

#' Tune CausalVAE hyperparameters
#'
#' @param train_loader Training dataloader.
#' @param val_loader Validation dataloader.
#' @param n_trials Number of random trials.
#' @param n_epochs_per_trial Number of epochs per trial.
#' @param input_dim Input dimension.
#' @param latent_dim Latent dimension.
#' @param device Device string.
#' @return List of best parameters.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # tune_hyperparameters(...)
#' }
#' @export
tune_hyperparameters <- function(train_loader, val_loader, n_trials = 10L, n_epochs_per_trial = 50L,
                                 input_dim = 3L, latent_dim = 3L, device = NULL) {
  best_loss <- Inf
  best_params <- list()
  for (trial in seq_len(as.integer(n_trials))) {
    lr <- 10^stats::runif(1, -5, -3)
    beta <- stats::runif(1, 0.5, 2.0)
    gamma <- stats::runif(1, 0.001, 0.01)
    hidden_dim <- sample(c(32L, 64L), 1)
    model <- CausalVAE(
      input_dim = input_dim, latent_dim = latent_dim, hidden_dim = hidden_dim,
      beta = beta, gamma = gamma, device = device
    )
    fit <- train_causalvae(
      model = model, train_loader = train_loader, val_loader = val_loader,
      epochs = as.integer(n_epochs_per_trial), warmup_epochs = min(20L, as.integer(n_epochs_per_trial)),
      learning_rate = lr, verbose = FALSE, device = device
    )
    trial_loss <- min(fit$val_losses[is.finite(fit$val_losses)], na.rm = TRUE)
    if (is.finite(trial_loss) && trial_loss < best_loss) {
      best_loss <- trial_loss
      best_params <- list(lr = lr, beta = beta, gamma = gamma, hidden_dim = hidden_dim)
    }
  }
  best_params
}

#' Print optimization summary for CausalVAE helpers
#'
#' @return
#' Object returned by \code{print_optimization_summary}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # print_optimization_summary(...)
#' }
#' @export
print_optimization_summary <- function() {
  cat("\n========================================\n")
  cat("CAUSALVAE PACKAGE HELPERS\n")
  cat("========================================\n")
  cat("  - Stable Gaussian reconstruction + KL + DAG/sparsity loss\n")
  cat("  - Gradient clipping and ReduceLROnPlateau scheduler\n")
  cat("  - Warmup scaling for structural penalty\n")
  cat("========================================\n")
}

#' Estimate ATE from CausalVAE_ATE model
#'
#' @param model A fitted \code{CausalVAE_ATE} model.
#' @param n_samples Number of Monte Carlo samples.
#' @param treatment_dim Latent dimension index to intervene on.
#' @param shift Intervention shift size.
#' @param device Device string.
#' @return Numeric scalar ATE estimate.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # estimate_ate_causalvae_ate(...)
#' }
#' @export
estimate_ate_causalvae_ate <- function(model, n_samples = 500L, treatment_dim = 1L, shift = 1.0, device = NULL) {
  dev <- .deepnet_select_device(device)
  model$eval()
  torch::with_no_grad({
    eps <- torch::torch_randn(as.integer(n_samples), model$latent_dim, device = dev)
    z <- model$causal_layer(eps)
    t0 <- torch::torch_zeros(as.integer(n_samples), 1L, device = dev)
    t1 <- torch::torch_ones(as.integer(n_samples), 1L, device = dev)
    y0 <- model$outcome_head(torch::torch_cat(list(z, t0), dim = 2L))
    delta <- torch::torch_zeros_like(z)
    delta[, as.integer(treatment_dim)] <- shift
    y1 <- model$outcome_head(torch::torch_cat(list(z + delta, t1), dim = 2L))
    as.numeric((y1$mean() - y0$mean())$item())
  })
}

#' CausalVAE
#'
#' Fits a causal variational autoencoder with DAG and sparsity penalties.
#' This integrates the package-safe model from \code{causalVAE.R}.
#'
#' @param X Covariate matrix or data.frame.
#' @param latent_dim Latent dimension (default 3).
#' @param hidden_dim Hidden dimension (default 64).
#' @param beta KL weight (default 1).
#' @param gamma Causal penalty weight (default 0.01).
#' @param lambda_sparsity Sparsity weight (default 0.001).
#' @param alpha DAG penalty scale (default 0.001).
#' @param num_epochs Number of epochs (default 200).
#' @param batch_size Batch size (default 256).
#' @param learning_rate Learning rate (default 1e-4).
#' @param weight_decay Weight decay (default 1e-5).
#' @param warmup_epochs Warmup epochs for causal term scaling (default 40).
#' @param verbose If TRUE, print progress.
#' @param device Device string: \code{"cuda"}, \code{"cpu"}, or \code{NULL}.
#' @param ... Ignored.
#' @return Object of class \code{causal_vae}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # causal_vae(...)
#' }
#' @export
causal_vae <- function(X,
                       latent_dim = 3L,
                       hidden_dim = 64L,
                       beta = 1.0,
                       gamma = 0.01,
                       lambda_sparsity = 0.001,
                       alpha = 0.001,
                       num_epochs = 200L,
                       batch_size = 256L,
                       learning_rate = 1e-4,
                       weight_decay = 1e-5,
                       warmup_epochs = 40L,
                       verbose = TRUE,
                       device = NULL,
                       ...) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("causal_vae() requires package 'torch'.")
  .deepnet_fit_causalvae(
    X = X,
    latent_dim = latent_dim,
    hidden_dim = hidden_dim,
    beta = beta,
    gamma = gamma,
    lambda_sparsity = lambda_sparsity,
    alpha = alpha,
    num_epochs = num_epochs,
    batch_size = batch_size,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    warmup_epochs = warmup_epochs,
    verbose = verbose,
    device = device
  )
}

#' Optimized CausalVAE
#'
#' Optimized CausalVAE interface integrated from \code{causalVAE_Opt.R}. Uses the same
#' package-safe implementation as \code{\link{causal_vae}} with optimized defaults.
#'
#' @inheritParams causal_vae
#' @return Object of class \code{causal_vae}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # causal_vae_opt(...)
#' }
#' @export
causal_vae_opt <- function(X,
                           latent_dim = 3L,
                           hidden_dim = 64L,
                           beta = 2.0,
                           gamma = 2.0,
                           lambda_sparsity = 0.2,
                           alpha = 0.05,
                           num_epochs = 300L,
                           batch_size = 256L,
                           learning_rate = 5e-4,
                           weight_decay = 1e-5,
                           warmup_epochs = 60L,
                           verbose = TRUE,
                           device = NULL,
                           ...) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("causal_vae_opt() requires package 'torch'.")
  .deepnet_fit_causalvae(
    X = X,
    latent_dim = latent_dim,
    hidden_dim = hidden_dim,
    beta = beta,
    gamma = gamma,
    lambda_sparsity = lambda_sparsity,
    alpha = alpha,
    num_epochs = num_epochs,
    batch_size = batch_size,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    warmup_epochs = warmup_epochs,
    verbose = verbose,
    device = device
  )
}

#' Predict latent codes from CausalVAE
#'
#' Returns latent representation \eqn{z} from a fitted \code{causal_vae} model.
#'
#' @param object Fitted \code{causal_vae} object.
#' @param newdata Covariate matrix or data.frame.
#' @param ... Ignored.
#' @return Matrix of latent codes.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.causal_vae(...)
#' }
#' @export
predict.causal_vae <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  object$model$eval()
  torch::with_no_grad({
    x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = object$device)
    out <- object$model(x_t)
    as.matrix(out$z$to(device = "cpu"))
  })
}

.deepnet_dscm_treatment_module <- function(input_dim, hidden_dim, dropout) {
  torch::nn_module(
    "DSCMTreatmentNet",
    initialize = function() {
      self$net <- torch::nn_sequential(
        torch::nn_linear(input_dim, hidden_dim),
        torch::nn_layer_norm(hidden_dim),
        torch::nn_gelu(),
        torch::nn_dropout(p = dropout),
        torch::nn_linear(hidden_dim, hidden_dim),
        torch::nn_layer_norm(hidden_dim),
        torch::nn_gelu(),
        torch::nn_dropout(p = dropout),
        torch::nn_linear(hidden_dim, 1L)
      )
    },
    forward = function(x) self$net(x)$squeeze(2L)
  )
}

.deepnet_dscm_outcome_module <- function(input_dim, hidden_dim, latent_dim, dropout) {
  torch::nn_module(
    "OutcomeDSCM",
    initialize = function() {
      self$latent_dim <- as.integer(latent_dim)
      self$phi <- torch::nn_sequential(
        torch::nn_linear(input_dim, hidden_dim),
        torch::nn_layer_norm(hidden_dim),
        torch::nn_gelu(),
        torch::nn_dropout(p = dropout),
        torch::nn_linear(hidden_dim, hidden_dim),
        torch::nn_layer_norm(hidden_dim),
        torch::nn_gelu()
      )
      self$encoder <- torch::nn_sequential(
        torch::nn_linear(hidden_dim + 2L, hidden_dim),
        torch::nn_layer_norm(hidden_dim),
        torch::nn_gelu(),
        torch::nn_dropout(p = dropout),
        torch::nn_linear(hidden_dim, hidden_dim),
        torch::nn_gelu()
      )
      self$fc_mu <- torch::nn_linear(hidden_dim, latent_dim)
      self$fc_logvar <- torch::nn_linear(hidden_dim, latent_dim)
      self$head_y0 <- torch::nn_sequential(
        torch::nn_linear(hidden_dim + latent_dim, hidden_dim),
        torch::nn_layer_norm(hidden_dim),
        torch::nn_gelu(),
        torch::nn_dropout(p = dropout),
        torch::nn_linear(hidden_dim, 1L)
      )
      self$head_y1 <- torch::nn_sequential(
        torch::nn_linear(hidden_dim + latent_dim, hidden_dim),
        torch::nn_layer_norm(hidden_dim),
        torch::nn_gelu(),
        torch::nn_dropout(p = dropout),
        torch::nn_linear(hidden_dim, 1L)
      )
    },
    encode = function(x, t, y) {
      rep <- self$phi(x)
      h <- self$encoder(torch::torch_cat(list(rep, t$unsqueeze(2L), y$unsqueeze(2L)), dim = 2L))
      mu <- self$fc_mu(h)
      logvar <- self$fc_logvar(h)$clamp(min = -8.0, max = 8.0)
      list(mu = mu, logvar = logvar)
    },
    reparameterize = function(mu, logvar) {
      std <- torch::torch_exp(0.5 * logvar)
      eps <- torch::torch_randn_like(std)
      mu + eps * std
    },
    potential_outcomes = function(x, z = NULL) {
      rep <- self$phi(x)
      if (is.null(z)) {
        z <- torch::torch_zeros(rep$size(1), self$latent_dim, device = rep$device, dtype = rep$dtype)
      }
      h <- torch::torch_cat(list(rep, z), dim = 2L)
      y0 <- self$head_y0(h)$squeeze(2L)
      y1 <- self$head_y1(h)$squeeze(2L)
      list(y0 = y0, y1 = y1)
    },
    decode = function(x, t, z = NULL) {
      po <- self$potential_outcomes(x, z = z)
      torch::torch_where(t > 0.5, po$y1, po$y0)
    },
    forward = function(x, t, y, y0_true = NULL, y1_true = NULL, beta = 1.0, lambda_oracle = 0.0) {
      enc <- self$encode(x, t, y)
      z <- self$reparameterize(enc$mu, enc$logvar)
      po <- self$potential_outcomes(x, z = z)
      y_pred <- torch::torch_where(t > 0.5, po$y1, po$y0)
      factual_loss <- torch::nnf_mse_loss(y_pred, y, reduction = "mean")
      kl_loss <- -0.5 * torch::torch_mean(1 + enc$logvar - enc$mu^2 - torch::torch_exp(enc$logvar))
      oracle_loss <- torch::torch_tensor(0.0, device = x$device)
      if (!is.null(y0_true) && !is.null(y1_true) && lambda_oracle > 0) {
        oracle_loss <- 0.5 * (
          torch::nnf_mse_loss(po$y0, y0_true, reduction = "mean") +
            torch::nnf_mse_loss(po$y1, y1_true, reduction = "mean")
        )
      }
      total <- factual_loss + beta * kl_loss + lambda_oracle * oracle_loss
      list(total = total, factual = factual_loss, kl = kl_loss, oracle = oracle_loss)
    }
  )
}

#' Deep Structural Causal Model (DSCM)
#'
#' Fits a Deep Structural Causal Model following the tutorial in
#' \code{deep_structural _causalML_DSCMs.ipynb}. The model contains:
#' (1) a treatment mechanism \eqn{p(T \mid X)} and
#' (2) an outcome mechanism with latent abduction \eqn{q(Z \mid X, T, Y)}
#' and potential-outcome heads \eqn{Y(0), Y(1)}.
#'
#' Counterfactual inference follows abduction-action-prediction (AAP):
#' infer latent noise from factual data, intervene on treatment, then
#' decode counterfactual outcomes.
#'
#' @param X Covariate matrix or data.frame.
#' @param treatment Binary treatment vector (0/1).
#' @param y Outcome vector.
#' @param y0_true Optional true \eqn{Y(0)} for oracle supervision (e.g., IHDP simulations).
#' @param y1_true Optional true \eqn{Y(1)} for oracle supervision (e.g., IHDP simulations).
#' @param hidden_dim Hidden size for outcome network (default 256).
#' @param treatment_hidden_dim Hidden size for treatment network (default 128).
#' @param latent_dim Latent noise dimension for abduction (default 16).
#' @param dropout Dropout rate (default 0.1).
#' @param num_epochs Number of epochs (default 300).
#' @param batch_size Batch size (default 256).
#' @param learning_rate AdamW learning rate (default 2e-4).
#' @param weight_decay AdamW weight decay (default 1e-5).
#' @param val_frac Validation fraction (default 0.15).
#' @param kl_warmup_epochs Number of KL warmup epochs (default 80).
#' @param max_beta Maximum KL weight after warmup (default 0.25).
#' @param lambda_oracle Oracle supervision weight (default 0).
#' @param patience Early stopping patience on validation objective (default 60).
#' @param max_grad_norm Gradient clipping norm (default 5).
#' @param verbose If TRUE, print progress every 25 epochs.
#' @param device Device string: \code{"cuda"}, \code{"cpu"}, or \code{NULL}.
#' @param ... Ignored.
#' @return Object of class \code{dscm}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # dscm(...)
#' }
#' @export
dscm <- function(X, treatment, y,
                 y0_true = NULL,
                 y1_true = NULL,
                 hidden_dim = 256L,
                 treatment_hidden_dim = 128L,
                 latent_dim = 16L,
                 dropout = 0.1,
                 num_epochs = 300L,
                 batch_size = 256L,
                 learning_rate = 2e-4,
                 weight_decay = 1e-5,
                 val_frac = 0.15,
                 kl_warmup_epochs = 80L,
                 max_beta = 0.25,
                 lambda_oracle = 0.0,
                 patience = 60L,
                 max_grad_norm = 5.0,
                 verbose = TRUE,
                 device = NULL,
                 ...) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("dscm() requires package 'torch'.")
  X <- as.matrix(X)
  treatment <- as.numeric(treatment)
  y <- as.numeric(y)
  if (nrow(X) != length(treatment) || nrow(X) != length(y))
    stop("X, treatment, and y must have matching number of rows/elements.")
  if (any(!is.finite(X)) || any(!is.finite(treatment)) || any(!is.finite(y)))
    stop("X, treatment, and y must be finite.")
  if (any(!(treatment %in% c(0, 1))))
    stop("treatment must be binary (0/1).")

  has_oracle <- !is.null(y0_true) && !is.null(y1_true)
  if (has_oracle) {
    y0_true <- as.numeric(y0_true)
    y1_true <- as.numeric(y1_true)
    if (length(y0_true) != nrow(X) || length(y1_true) != nrow(X))
      stop("y0_true and y1_true must have length nrow(X).")
    if (any(!is.finite(y0_true)) || any(!is.finite(y1_true)))
      stop("y0_true and y1_true must be finite.")
  } else {
    lambda_oracle <- 0.0
  }

  n <- nrow(X)
  p <- ncol(X)
  dev <- .deepnet_select_device(device)
  y_mean <- mean(y)
  y_std <- stats::sd(y)
  if (!is.finite(y_std) || y_std <= 0) y_std <- 1.0
  y_std_vec <- (y - y_mean) / y_std
  if (has_oracle) {
    y0_std_vec <- (y0_true - y_mean) / y_std
    y1_std_vec <- (y1_true - y_mean) / y_std
  }

  Treatment <- .deepnet_dscm_treatment_module(as.integer(p), as.integer(treatment_hidden_dim), dropout)
  Outcome <- .deepnet_dscm_outcome_module(as.integer(p), as.integer(hidden_dim), as.integer(latent_dim), dropout)
  treatment_net <- Treatment()
  outcome_model <- Outcome()
  treatment_net$to(device = dev)
  outcome_model$to(device = dev)

  x_t <- torch::torch_tensor(X, dtype = torch::torch_float32(), device = dev)
  t_t <- torch::torch_tensor(treatment, dtype = torch::torch_float32(), device = dev)
  y_t <- torch::torch_tensor(y_std_vec, dtype = torch::torch_float32(), device = dev)
  if (has_oracle) {
    y0_t <- torch::torch_tensor(y0_std_vec, dtype = torch::torch_float32(), device = dev)
    y1_t <- torch::torch_tensor(y1_std_vec, dtype = torch::torch_float32(), device = dev)
  }

  val_size <- floor(max(1, min(n - 1, round(val_frac * n))))
  perm <- sample.int(n)
  val_idx <- perm[seq_len(val_size)]
  tr_idx <- perm[(val_size + 1L):n]
  if (length(tr_idx) == 0L) {
    tr_idx <- val_idx
    val_idx <- integer(0)
  }

  num_pos <- max(sum(treatment[tr_idx] > 0.5), 1.0)
  num_neg <- max(length(tr_idx) - sum(treatment[tr_idx] > 0.5), 1.0)
  pos_weight <- torch::torch_tensor(num_neg / num_pos, dtype = torch::torch_float32(), device = dev)
  bce_logits <- torch::nn_bce_with_logits_loss(pos_weight = pos_weight)

  opt_t <- torch::optim_adamw(treatment_net$parameters, lr = learning_rate, weight_decay = weight_decay)
  opt_y <- torch::optim_adamw(outcome_model$parameters, lr = learning_rate, weight_decay = weight_decay)
  sched_t <- torch::lr_reduce_on_plateau(opt_t, mode = "min", factor = 0.5, patience = 15L)
  sched_y <- torch::lr_reduce_on_plateau(opt_y, mode = "min", factor = 0.5, patience = 15L)

  best_val <- Inf
  best_t_state <- NULL
  best_y_state <- NULL
  wait <- 0L
  batch_size <- min(as.integer(batch_size), length(tr_idx))
  if (batch_size < 1L) batch_size <- 1L
  n_batches <- max(1L, ceiling(length(tr_idx) / batch_size))
  history <- vector("list", as.integer(num_epochs))

  for (epoch in seq_len(as.integer(num_epochs))) {
    treatment_net$train(TRUE)
    outcome_model$train(TRUE)
    beta <- min(max_beta, max_beta * epoch / max(1L, as.integer(kl_warmup_epochs)))
    sums <- c(train_t = 0, train_y = 0, factual = 0, kl = 0, oracle = 0)

    train_perm <- sample(tr_idx)
    for (start in seq(1L, length(train_perm), by = batch_size)) {
      idx <- train_perm[start:min(start + batch_size - 1L, length(train_perm))]
      xb <- x_t[idx, , drop = FALSE]
      tb <- t_t[idx]
      yb <- y_t[idx]

      opt_t$zero_grad()
      t_logits <- treatment_net(xb)
      loss_t <- bce_logits(t_logits, tb)
      loss_t$backward()
      torch::nn_utils_clip_grad_norm_(treatment_net$parameters, max_norm = max_grad_norm)
      opt_t$step()
      sums["train_t"] <- sums["train_t"] + as.numeric(loss_t$item())

      opt_y$zero_grad()
      losses <- outcome_model$forward(
        xb, tb, yb,
        y0_true = if (has_oracle) y0_t[idx] else NULL,
        y1_true = if (has_oracle) y1_t[idx] else NULL,
        beta = beta,
        lambda_oracle = lambda_oracle
      )
      losses$total$backward()
      torch::nn_utils_clip_grad_norm_(outcome_model$parameters, max_norm = max_grad_norm)
      opt_y$step()
      sums["train_y"] <- sums["train_y"] + as.numeric(losses$total$item())
      sums["factual"] <- sums["factual"] + as.numeric(losses$factual$item())
      sums["kl"] <- sums["kl"] + as.numeric(losses$kl$item())
      sums["oracle"] <- sums["oracle"] + as.numeric(losses$oracle$item())
    }

    if (length(val_idx) > 0L) {
      treatment_net$eval()
      outcome_model$eval()
      val_t_loss <- 0
      val_y_loss <- 0
      val_pehe_proxy <- 0
      n_val_batches <- 0L
      torch::with_no_grad({
        for (start in seq(1L, length(val_idx), by = batch_size)) {
          idx <- val_idx[start:min(start + batch_size - 1L, length(val_idx))]
          xb <- x_t[idx, , drop = FALSE]
          tb <- t_t[idx]
          yb <- y_t[idx]
          n_val_batches <<- n_val_batches + 1L
          val_t_loss <<- val_t_loss + as.numeric(bce_logits(treatment_net(xb), tb)$item())
          val_losses <- outcome_model$forward(
            xb, tb, yb,
            y0_true = if (has_oracle) y0_t[idx] else NULL,
            y1_true = if (has_oracle) y1_t[idx] else NULL,
            beta = max_beta,
            lambda_oracle = lambda_oracle
          )
          val_y_loss <<- val_y_loss + as.numeric(val_losses$total$item())
          if (has_oracle) {
            po <- outcome_model$potential_outcomes(xb, z = NULL)
            pehe_b <- torch::torch_sqrt(torch::torch_mean(((po$y1 - po$y0) - (y1_t[idx] - y0_t[idx]))^2))
            val_pehe_proxy <<- val_pehe_proxy + as.numeric(pehe_b$item())
          }
        }
      })
      if (n_val_batches > 0L) {
        val_t_loss <- val_t_loss / n_val_batches
        val_y_loss <- val_y_loss / n_val_batches
        val_pehe_proxy <- if (has_oracle) val_pehe_proxy / n_val_batches else 0
        val_total <- val_t_loss + val_y_loss + if (has_oracle) 0.2 * val_pehe_proxy else 0
      } else {
        val_t_loss <- NA_real_
        val_y_loss <- NA_real_
        val_pehe_proxy <- NA_real_
        val_total <- NA_real_
      }
      if (is.finite(val_t_loss)) sched_t$step(val_t_loss)
      if (is.finite(val_y_loss)) sched_y$step(val_y_loss)

      if (is.finite(val_total) && val_total < best_val) {
        best_val <- val_total
        wait <- 0L
        best_t_state <- treatment_net$state_dict()
        best_y_state <- outcome_model$state_dict()
      } else {
        wait <- wait + 1L
      }
    } else {
      val_t_loss <- NA_real_
      val_y_loss <- NA_real_
      val_pehe_proxy <- NA_real_
      val_total <- NA_real_
    }

    metrics <- list(
      beta = beta,
      train_t = sums["train_t"] / n_batches,
      train_y = sums["train_y"] / n_batches,
      factual = sums["factual"] / n_batches,
      kl = sums["kl"] / n_batches,
      oracle = sums["oracle"] / n_batches,
      val_t = val_t_loss,
      val_y = val_y_loss,
      val_pehe = val_pehe_proxy,
      val_total = val_total
    )
    history[[epoch]] <- metrics

    if (isTRUE(verbose) && (epoch == 1L || epoch %% 25L == 0L)) {
      msg <- paste0(
        "DSCM epoch ", epoch,
        " beta=", round(metrics$beta, 3),
        " train_t=", round(metrics$train_t, 4),
        " train_y=", round(metrics$train_y, 4),
        " (factual=", round(metrics$factual, 4),
        ", kl=", round(metrics$kl, 4),
        ", oracle=", round(metrics$oracle, 4), ")"
      )
      if (length(val_idx) > 0L) {
        msg <- paste0(
          msg,
          " val_t=", round(metrics$val_t, 4),
          " val_y=", round(metrics$val_y, 4)
        )
        if (has_oracle) msg <- paste0(msg, " val_pehe~=", round(metrics$val_pehe, 4))
      }
      message(msg)
    }

    if (length(val_idx) > 0L && wait >= as.integer(patience)) {
      if (isTRUE(verbose))
        message("DSCM early stopping at epoch ", epoch, " (best val=", round(best_val, 4), ").")
      break
    }
  }

  if (!is.null(best_t_state) && !is.null(best_y_state)) {
    treatment_net$load_state_dict(best_t_state)
    outcome_model$load_state_dict(best_y_state)
  }

  structure(
    list(
      treatment_net = treatment_net,
      outcome_model = outcome_model,
      X_names = colnames(X),
      device = dev,
      y_mean = y_mean,
      y_std = y_std,
      latent_dim = as.integer(latent_dim),
      history = history,
      type = "dscm_torch"
    ),
    class = "dscm"
  )
}

#' Predict from Deep Structural Causal Model (DSCM)
#'
#' Supports deterministic potential outcomes and AAP counterfactuals.
#'
#' @param object Fitted \code{dscm} object.
#' @param newdata Covariate matrix or data.frame.
#' @param type Prediction type: \code{"ite"}, \code{"potential_outcomes"}, or \code{"counterfactual"}.
#' @param t_cf Counterfactual treatment assignment for \code{type = "counterfactual"} (scalar or length \code{nrow(newdata)}).
#' @param use_abduction If TRUE, use latent abduction with \code{t_obs} and \code{y_obs}.
#' @param t_obs Observed factual treatment for abduction (required if \code{use_abduction = TRUE}).
#' @param y_obs Observed factual outcome for abduction (required if \code{use_abduction = TRUE}).
#' @param n_samples Number of Monte Carlo samples for abduction-based counterfactuals.
#' @param ... Ignored.
#' @return For \code{type = "ite"}, numeric ITE vector. For
#' \code{type = "potential_outcomes"}, list with \code{y0} and \code{y1}.
#' For \code{type = "counterfactual"}, numeric vector.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.dscm(...)
#' }
#' @export
predict.dscm <- function(object, newdata, type = c("ite", "potential_outcomes", "counterfactual"),
                         t_cf = 1,
                         use_abduction = FALSE,
                         t_obs = NULL,
                         y_obs = NULL,
                         n_samples = 64L,
                         ...) {
  type <- match.arg(type)
  X <- as.matrix(newdata)
  n <- nrow(X)
  x_t <- torch::torch_tensor(X, dtype = torch::torch_float32(), device = object$device)
  out_model <- object$outcome_model
  out_model$eval()

  to_obs_tensor <- function(v, nm) {
    if (length(v) == 1L) v <- rep(v, n)
    if (length(v) != n) stop(nm, " must be scalar or length nrow(newdata).")
    torch::torch_tensor(as.numeric(v), dtype = torch::torch_float32(), device = object$device)
  }

  denorm <- function(v) as.numeric(v$to(device = "cpu")) * object$y_std + object$y_mean

  torch::with_no_grad({
    po <- out_model$potential_outcomes(x_t, z = NULL)
    y0_det <- po$y0
    y1_det <- po$y1

    if (identical(type, "potential_outcomes")) {
      return(list(y0 = denorm(y0_det), y1 = denorm(y1_det)))
    }
    if (identical(type, "ite")) {
      return(denorm(y1_det) - denorm(y0_det))
    }

    t_cf_t <- to_obs_tensor(t_cf, "t_cf")
    if (isTRUE(use_abduction)) {
      if (is.null(t_obs) || is.null(y_obs)) {
        stop("t_obs and y_obs are required when use_abduction = TRUE.")
      }
      t_obs_t <- to_obs_tensor(t_obs, "t_obs")
      y_obs_t <- to_obs_tensor((as.numeric(y_obs) - object$y_mean) / object$y_std, "y_obs")
      enc <- out_model$encode(x_t, t_obs_t, y_obs_t)
      z <- out_model$reparameterize(enc$mu, enc$logvar)
      n_samples <- max(1L, as.integer(n_samples))
      x_rep <- x_t$repeat_interleave(repeats = n_samples, dim = 1L)
      z_rep <- z$repeat_interleave(repeats = n_samples, dim = 1L)
      t_rep <- t_cf_t$repeat_interleave(repeats = n_samples, dim = 1L)
      y_cf <- out_model$decode(x_rep, t_rep, z = z_rep)$view(c(n, n_samples))$mean(dim = 2L)
      return(denorm(y_cf))
    }

    y_cf <- torch::torch_where(t_cf_t > 0.5, y1_det, y0_det)
    denorm(y_cf)
  })
}

.deepnet_cdvae_rbf_kernel <- function(x, y, bandwidth) {
  x_sq <- (x^2)$sum(dim = 2L, keepdim = TRUE)
  y_sq <- (y^2)$sum(dim = 2L, keepdim = TRUE)
  cross <- torch::torch_matmul(x, y$t())
  dist_sq <- x_sq + y_sq$t() - 2.0 * cross
  torch::torch_exp(-dist_sq / (2.0 * bandwidth^2))
}

.deepnet_cdvae_mmd_loss <- function(z_treated, z_control, n_kernels = 5L) {
  if (z_treated$size(1) < 2L || z_control$size(1) < 2L) {
    return(torch::torch_tensor(0.0, device = z_treated$device))
  }
  all_bandwidths <- c(0.1, 0.5, 1.0, 2.0, 5.0)
  bandwidths <- all_bandwidths[seq_len(min(as.integer(n_kernels), length(all_bandwidths)))]
  total_mmd <- torch::torch_tensor(0.0, device = z_treated$device)
  for (bw in bandwidths) {
    k_tt <- .deepnet_cdvae_rbf_kernel(z_treated, z_treated, bw)$mean()
    k_cc <- .deepnet_cdvae_rbf_kernel(z_control, z_control, bw)$mean()
    k_tc <- .deepnet_cdvae_rbf_kernel(z_treated, z_control, bw)$mean()
    total_mmd <- total_mmd + k_tt + k_cc - 2.0 * k_tc
  }
  total_mmd / length(bandwidths)
}

.deepnet_cdvae_module <- function(input_dim, hidden_dim, latent_dim) {
  torch::nn_module(
    "CausalDiscrepancyVAE",
    initialize = function() {
      h2 <- as.integer(max(2L, hidden_dim %/% 2L))
      h4 <- as.integer(max(1L, hidden_dim %/% 4L))

      self$enc_fc1 <- torch::nn_linear(input_dim, hidden_dim)
      self$enc_ln1 <- torch::nn_layer_norm(hidden_dim)
      self$enc_fc2 <- torch::nn_linear(hidden_dim, hidden_dim)
      self$enc_ln2 <- torch::nn_layer_norm(hidden_dim)
      self$enc_fc3 <- torch::nn_linear(hidden_dim, h2)
      self$enc_ln3 <- torch::nn_layer_norm(h2)
      self$enc_mu <- torch::nn_linear(h2, latent_dim)
      self$enc_logvar <- torch::nn_linear(h2, latent_dim)

      self$dec_fc1 <- torch::nn_linear(latent_dim, h2)
      self$dec_ln1 <- torch::nn_layer_norm(h2)
      self$dec_fc2 <- torch::nn_linear(h2, hidden_dim)
      self$dec_ln2 <- torch::nn_layer_norm(hidden_dim)
      self$dec_fc3 <- torch::nn_linear(hidden_dim, hidden_dim)
      self$dec_ln3 <- torch::nn_layer_norm(hidden_dim)
      self$dec_out <- torch::nn_linear(hidden_dim, input_dim)

      self$y0_fc1 <- torch::nn_linear(latent_dim, h2)
      self$y0_fc2 <- torch::nn_linear(h2, h4)
      self$y0_out <- torch::nn_linear(h4, 1L)

      self$y1_fc1 <- torch::nn_linear(latent_dim, h2)
      self$y1_fc2 <- torch::nn_linear(h2, h4)
      self$y1_out <- torch::nn_linear(h4, 1L)
    },
    encode = function(x) {
      h <- torch::nnf_leaky_relu(self$enc_ln1(self$enc_fc1(x)), negative_slope = 0.1)
      h <- torch::nnf_leaky_relu(self$enc_ln2(self$enc_fc2(h)), negative_slope = 0.1)
      h <- torch::nnf_leaky_relu(self$enc_ln3(self$enc_fc3(h)), negative_slope = 0.1)
      list(mu = self$enc_mu(h), logvar = self$enc_logvar(h)$clamp(min = -4.0, max = 4.0))
    },
    reparameterize = function(mu, logvar) {
      std <- torch::torch_exp(0.5 * logvar)
      eps <- torch::torch_randn_like(std)
      mu + std * eps
    },
    decode = function(z) {
      h <- torch::nnf_relu(self$dec_ln1(self$dec_fc1(z)))
      h <- torch::nnf_relu(self$dec_ln2(self$dec_fc2(h)))
      h <- torch::nnf_relu(self$dec_ln3(self$dec_fc3(h)))
      self$dec_out(h)
    },
    outcome_head0 = function(z) {
      h <- torch::nnf_relu(self$y0_fc1(z))
      h <- torch::nnf_relu(self$y0_fc2(h))
      self$y0_out(h)$squeeze(2L)
    },
    outcome_head1 = function(z) {
      h <- torch::nnf_relu(self$y1_fc1(z))
      h <- torch::nnf_relu(self$y1_fc2(h))
      self$y1_out(h)$squeeze(2L)
    },
    forward = function(x, t) {
      enc <- self$encode(x)
      z <- self$reparameterize(enc$mu, enc$logvar)
      x_hat <- self$decode(z)
      y0_hat <- self$outcome_head0(z)
      y1_hat <- self$outcome_head1(z)
      y_factual_hat <- torch::torch_where(t > 0.5, y1_hat, y0_hat)
      list(
        x_hat = x_hat, mu = enc$mu, logvar = enc$logvar, z = z,
        y0_hat = y0_hat, y1_hat = y1_hat, y_factual_hat = y_factual_hat
      )
    },
    potential_outcomes = function(x) {
      enc <- self$encode(x)
      z <- self$reparameterize(enc$mu, enc$logvar)
      list(y0_hat = self$outcome_head0(z), y1_hat = self$outcome_head1(z))
    }
  )
}

.deepnet_cdvae_loss <- function(x_in, y_true, t, out, beta = 1.0, lambda_mmd = 1.0, lambda_y = 10.0, n_kernels = 5L) {
  recon_loss <- torch::nnf_mse_loss(out$x_hat, x_in, reduction = "mean")
  kl_loss <- -0.5 * (1 + out$logvar - out$mu^2 - torch::torch_exp(out$logvar))$mean()
  y_loss <- torch::nnf_mse_loss(out$y_factual_hat, y_true, reduction = "mean")

  treated_idx <- torch::torch_where(t > 0.5)[[1]]
  control_idx <- torch::torch_where(t <= 0.5)[[1]]
  if (treated_idx$numel() >= 2L && control_idx$numel() >= 2L) {
    z_t <- out$z$index_select(1L, treated_idx)
    z_c <- out$z$index_select(1L, control_idx)
    mmd <- .deepnet_cdvae_mmd_loss(z_t, z_c, n_kernels = n_kernels)
  } else {
    mmd <- torch::torch_tensor(0.0, device = out$z$device)
  }

  total <- recon_loss + beta * kl_loss + lambda_mmd * mmd + lambda_y * y_loss
  list(total = total, recon = recon_loss, kl = kl_loss, mmd = mmd, y = y_loss)
}

#' CausalDiscrepancyVAE
#'
#' Fits a discrepancy-regularized causal VAE with latent MMD balancing between
#' treatment groups and treatment-specific outcome heads.
#' Ported from \code{causalDiscrepanct_VAE.ipynb} into package-safe R/\pkg{torch}.
#'
#' @param X Covariate matrix or data.frame.
#' @param treatment Binary treatment vector (0/1).
#' @param y Outcome vector.
#' @param latent_dim Latent dimension (default 16).
#' @param hidden_dim Hidden dimension (default 128).
#' @param beta KL weight (default 1).
#' @param lambda_mmd MMD penalty weight (default 1).
#' @param lambda_y Outcome loss weight (default 10).
#' @param n_kernels Number of RBF kernels for MMD (default 5).
#' @param num_epochs Number of training epochs (default 100).
#' @param batch_size Batch size (default 128).
#' @param learning_rate Learning rate (default 1e-3).
#' @param weight_decay Weight decay for AdamW (default 1e-4).
#' @param verbose If TRUE, print progress every 10 epochs.
#' @param device Device string: \code{"cuda"}, \code{"cpu"}, or \code{NULL}.
#' @param ... Ignored.
#' @return Object of class \code{causal_discrepancy_vae}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # causal_discrepancy_vae(...)
#' }
#' @export
causal_discrepancy_vae <- function(X, treatment, y,
                                   latent_dim = 16L,
                                   hidden_dim = 128L,
                                   beta = 1.0,
                                   lambda_mmd = 1.0,
                                   lambda_y = 10.0,
                                   n_kernels = 5L,
                                   num_epochs = 100L,
                                   batch_size = 128L,
                                   learning_rate = 1e-3,
                                   weight_decay = 1e-4,
                                   verbose = TRUE,
                                   device = NULL,
                                   ...) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("causal_discrepancy_vae() requires package 'torch'.")
  X <- as.matrix(X)
  treatment <- as.numeric(treatment)
  y <- as.numeric(y)
  if (nrow(X) != length(treatment) || nrow(X) != length(y))
    stop("X, treatment, and y must have matching number of rows/elements.")
  if (any(!is.finite(X)) || any(!is.finite(treatment)) || any(!is.finite(y)))
    stop("X, treatment, and y must be finite.")
  if (any(!(treatment %in% c(0, 1))))
    stop("treatment must be binary (0/1).")

  n <- nrow(X)
  p <- ncol(X)
  dev <- .deepnet_select_device(device)

  Module <- .deepnet_cdvae_module(as.integer(p), as.integer(hidden_dim), as.integer(latent_dim))
  model <- Module()
  model$to(device = dev)

  x_t <- torch::torch_tensor(X, dtype = torch::torch_float32(), device = dev)
  t_t <- torch::torch_tensor(treatment, dtype = torch::torch_float32(), device = dev)
  y_t <- torch::torch_tensor(y, dtype = torch::torch_float32(), device = dev)

  opt <- torch::optim_adamw(model$parameters, lr = learning_rate, weight_decay = weight_decay)
  batch_size <- min(as.integer(batch_size), n)
  n_batches <- max(1L, ceiling(n / batch_size))

  history <- list(train = vector("list", as.integer(num_epochs)))
  for (epoch in seq_len(as.integer(num_epochs))) {
    model$train(TRUE)
    perm <- sample.int(n)
    sums <- c(total = 0, recon = 0, kl = 0, mmd = 0, y = 0)

    for (start in seq(1L, n, by = batch_size)) {
      idx <- perm[start:min(start + batch_size - 1L, n)]
      xb <- x_t[idx, , drop = FALSE]
      tb <- t_t[idx]
      yb <- y_t[idx]
      opt$zero_grad()
      out <- model(xb, tb)
      losses <- .deepnet_cdvae_loss(
        x_in = xb, y_true = yb, t = tb, out = out,
        beta = beta, lambda_mmd = lambda_mmd, lambda_y = lambda_y, n_kernels = n_kernels
      )
      losses$total$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 5.0)
      opt$step()
      sums["total"] <- sums["total"] + as.numeric(losses$total$item())
      sums["recon"] <- sums["recon"] + as.numeric(losses$recon$item())
      sums["kl"] <- sums["kl"] + as.numeric(losses$kl$item())
      sums["mmd"] <- sums["mmd"] + as.numeric(losses$mmd$item())
      sums["y"] <- sums["y"] + as.numeric(losses$y$item())
    }

    metrics <- as.list(sums / n_batches)
    history$train[[epoch]] <- metrics
    if (isTRUE(verbose) && (epoch == 1L || epoch %% 10L == 0L)) {
      message(
        "CausalDiscrepancyVAE epoch ", epoch,
        " total=", round(metrics$total, 4),
        " recon=", round(metrics$recon, 4),
        " kl=", round(metrics$kl, 4),
        " mmd=", round(metrics$mmd, 4),
        " y=", round(metrics$y, 4)
      )
    }
  }

  structure(
    list(
      model = model,
      X_names = colnames(X),
      type = "causal_discrepancy_vae_torch",
      device = dev,
      latent_dim = as.integer(latent_dim),
      history = history
    ),
    class = "causal_discrepancy_vae"
  )
}

#' Alias for \code{causal_discrepancy_vae}
#'
#' @inheritParams causal_discrepancy_vae
#' @return Object of class \code{causal_discrepancy_vae}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # CausalDiscrepancyVAE(...)
#' }
#' @export
CausalDiscrepancyVAE <- function(X, treatment, y, ...) {
  causal_discrepancy_vae(X = X, treatment = treatment, y = y, ...)
}

#' Predict from CausalDiscrepancyVAE
#'
#' @param object Fitted \code{causal_discrepancy_vae} object.
#' @param newdata Covariate matrix or data.frame.
#' @param type One of \code{"ite"}, \code{"mu0"}, \code{"mu1"}, or \code{"latent"}.
#' @param ... Ignored.
#' @return Numeric vector (for \code{"ite"}, \code{"mu0"}, \code{"mu1"}) or matrix (for \code{"latent"}).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.causal_discrepancy_vae(...)
#' }
#' @export
predict.causal_discrepancy_vae <- function(object, newdata, type = c("ite", "mu0", "mu1", "latent"), ...) {
  type <- match.arg(type)
  x <- as.matrix(newdata)
  model <- object$model
  model$eval()
  torch::with_no_grad({
    x_t <- torch::torch_tensor(x, dtype = torch::torch_float32(), device = object$device)
    if (type == "latent") {
      enc <- model$encode(x_t)
      return(as.matrix(enc$mu$to(device = "cpu")))
    }
    po <- model$potential_outcomes(x_t)
    y0 <- as.numeric(po$y0_hat$to(device = "cpu"))
    y1 <- as.numeric(po$y1_hat$to(device = "cpu"))
    if (type == "mu0") return(y0)
    if (type == "mu1") return(y1)
    y1 - y0
  })
}


.deepnet_causalegm_module <- function(input_dim, dim_c = 8L, dim_t = 4L, dim_y = 4L, hidden_dim = 128L) {
  torch::nn_module(
    "CausalEGM",
    initialize = function() {
      self$enc_shared <- torch::nn_sequential(
        torch::nn_linear(input_dim, hidden_dim),
        torch::nn_elu(),
        torch::nn_linear(hidden_dim, hidden_dim),
        torch::nn_elu()
      )
      self$enc_c <- torch::nn_linear(hidden_dim, dim_c)
      self$enc_t <- torch::nn_linear(hidden_dim, dim_t)
      self$enc_y <- torch::nn_linear(hidden_dim, dim_y)

      latent_total <- as.integer(dim_c + dim_t + dim_y)
      self$decoder <- torch::nn_sequential(
        torch::nn_linear(latent_total, hidden_dim),
        torch::nn_elu(),
        torch::nn_linear(hidden_dim, hidden_dim),
        torch::nn_elu(),
        torch::nn_linear(hidden_dim, input_dim)
      )

      h_head <- as.integer(max(4L, hidden_dim %/% 2L))
      self$treat_head <- torch::nn_sequential(
        torch::nn_linear(as.integer(dim_c + dim_t), h_head),
        torch::nn_elu(),
        torch::nn_linear(h_head, 1L)
      )
      self$outcome_head <- torch::nn_sequential(
        torch::nn_linear(as.integer(dim_c + dim_y + 1L), h_head),
        torch::nn_elu(),
        torch::nn_linear(h_head, h_head),
        torch::nn_elu(),
        torch::nn_linear(h_head, 1L)
      )
    },
    encode = function(x) {
      h <- self$enc_shared(x)
      list(
        zc = self$enc_c(h),
        zt = self$enc_t(h),
        zy = self$enc_y(h)
      )
    },
    decode = function(zc, zt, zy) {
      self$decoder(torch::torch_cat(list(zc, zt, zy), dim = 2L))
    },
    treatment_logit = function(zc, zt) {
      self$treat_head(torch::torch_cat(list(zc, zt), dim = 2L))$squeeze(2L)
    },
    outcome_pred = function(zc, zy, t) {
      t_col <- t$view(c(t$size(1), 1L))
      self$outcome_head(torch::torch_cat(list(zc, zy, t_col), dim = 2L))$squeeze(2L)
    },
    forward = function(x, t) {
      z <- self$encode(x)
      x_hat <- self$decode(z$zc, z$zt, z$zy)
      t_logit <- self$treatment_logit(z$zc, z$zt)
      y_hat <- self$outcome_pred(z$zc, z$zy, t)
      list(
        x_hat = x_hat,
        t_logit = t_logit,
        y_hat = y_hat,
        zc = z$zc,
        zt = z$zt,
        zy = z$zy
      )
    }
  )
}

.deepnet_causalegm_disc_module <- function(dim_t = 4L, dim_y = 4L, hidden_dim = 64L) {
  torch::nn_module(
    "CausalEGMDiscriminator",
    initialize = function() {
      self$net <- torch::nn_sequential(
        torch::nn_linear(as.integer(dim_t + dim_y), hidden_dim),
        torch::nn_elu(),
        torch::nn_linear(hidden_dim, 1L)
      )
    },
    forward = function(zt, zy) {
      self$net(torch::torch_cat(list(zt, zy), dim = 2L))$squeeze(2L)
    }
  )
}

.deepnet_causalegm_loss <- function(x_in, t, y, out, disc = NULL,
                                    lambda_recon = 1.0, lambda_treat = 2.0,
                                    lambda_outcome = 2.0, lambda_disent = 0.5) {
  l_recon <- torch::nnf_mse_loss(out$x_hat, x_in, reduction = "mean")
  l_treat <- torch::nnf_binary_cross_entropy_with_logits(out$t_logit, t, reduction = "mean")
  l_outcome <- torch::nnf_mse_loss(out$y_hat, y, reduction = "mean")

  l_disent <- torch::torch_tensor(0.0, device = x_in$device)
  if (!is.null(disc)) {
    perm <- torch::torch_randperm(out$zy$size(1), device = out$zy$device, dtype = torch::torch_long())$add(1L)
    zy_shuf <- out$zy$index_select(1L, perm)
    real_score <- disc(out$zt, out$zy)
    fake_score <- disc(out$zt, zy_shuf)
    l_disent <- (
      torch::nnf_binary_cross_entropy_with_logits(real_score, torch::torch_ones_like(real_score), reduction = "mean") +
        torch::nnf_binary_cross_entropy_with_logits(fake_score, torch::torch_zeros_like(fake_score), reduction = "mean")
    ) / 2.0
  }

  total <- lambda_recon * l_recon +
    lambda_treat * l_treat +
    lambda_outcome * l_outcome +
    lambda_disent * l_disent

  list(
    total = total,
    recon = l_recon,
    treat = l_treat,
    outcome = l_outcome,
    disent = l_disent
  )
}

#' CausalEGM
#'
#' Fits a Causal Encoding Generative Model with disentangled latent factors for
#' confounding (\code{Zc}), treatment-specific (\code{Zt}), and outcome-specific
#' (\code{Zy}) signals. The model jointly optimizes reconstruction, treatment,
#' and outcome losses with a discriminator-based disentanglement term.
#'
#' Ported from \code{causalEGM.ipynb} into package-safe R/\pkg{torch}.
#'
#' @param X Covariate matrix or data.frame.
#' @param treatment Binary treatment vector (0/1).
#' @param y Outcome vector.
#' @param dim_c Latent size for confounding signal (default 8).
#' @param dim_t Latent size for treatment-specific signal (default 4).
#' @param dim_y Latent size for outcome-specific signal (default 4).
#' @param hidden_dim Hidden layer width (default 128).
#' @param num_epochs Number of training epochs (default 150).
#' @param batch_size Mini-batch size (default 256).
#' @param learning_rate Model learning rate (default 1e-3).
#' @param learning_rate_disc Discriminator learning rate (default 5e-4).
#' @param weight_decay Weight decay for model optimizer (default 1e-4).
#' @param lambda_recon Reconstruction loss weight (default 1).
#' @param lambda_treat Treatment loss weight (default 2).
#' @param lambda_outcome Outcome loss weight (default 2).
#' @param lambda_disent Disentanglement loss weight (default 0.5).
#' @param max_grad_norm Gradient clipping threshold (default 1).
#' @param verbose If TRUE, print progress every 10 epochs.
#' @param device Device string: \code{"cuda"}, \code{"cpu"}, or \code{NULL}.
#' @param ... Ignored.
#' @return Object of class \code{causal_egm}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # causal_egm(...)
#' }
#' @export
causal_egm <- function(X, treatment, y,
                       dim_c = 8L,
                       dim_t = 4L,
                       dim_y = 4L,
                       hidden_dim = 128L,
                       num_epochs = 150L,
                       batch_size = 256L,
                       learning_rate = 1e-3,
                       learning_rate_disc = 5e-4,
                       weight_decay = 1e-4,
                       lambda_recon = 1.0,
                       lambda_treat = 2.0,
                       lambda_outcome = 2.0,
                       lambda_disent = 0.5,
                       max_grad_norm = 1.0,
                       verbose = TRUE,
                       device = NULL,
                       ...) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("causal_egm() requires package 'torch'.")

  X <- as.matrix(X)
  treatment <- as.numeric(treatment)
  y <- as.numeric(y)
  if (nrow(X) != length(treatment) || nrow(X) != length(y))
    stop("X, treatment, and y must have matching number of rows/elements.")
  if (any(!is.finite(X)) || any(!is.finite(treatment)) || any(!is.finite(y)))
    stop("X, treatment, and y must be finite.")
  if (any(!(treatment %in% c(0, 1))))
    stop("treatment must be binary (0/1).")

  n <- nrow(X)
  p <- ncol(X)
  batch_size <- max(1L, min(as.integer(batch_size), n))
  n_batches <- max(1L, ceiling(n / batch_size))
  dev <- .deepnet_select_device(device)

  Model <- .deepnet_causalegm_module(
    input_dim = as.integer(p),
    dim_c = as.integer(dim_c),
    dim_t = as.integer(dim_t),
    dim_y = as.integer(dim_y),
    hidden_dim = as.integer(hidden_dim)
  )
  Disc <- .deepnet_causalegm_disc_module(
    dim_t = as.integer(dim_t),
    dim_y = as.integer(dim_y),
    hidden_dim = as.integer(max(8L, hidden_dim %/% 2L))
  )

  model <- Model()
  disc <- Disc()
  model$to(device = dev)
  disc$to(device = dev)

  x_t <- torch::torch_tensor(X, dtype = torch::torch_float32(), device = dev)
  t_t <- torch::torch_tensor(treatment, dtype = torch::torch_float32(), device = dev)
  y_t <- torch::torch_tensor(y, dtype = torch::torch_float32(), device = dev)

  opt_model <- torch::optim_adam(model$parameters, lr = learning_rate, weight_decay = weight_decay)
  opt_disc <- torch::optim_adam(disc$parameters, lr = learning_rate_disc)

  history <- list(train = vector("list", as.integer(num_epochs)))
  for (epoch in seq_len(as.integer(num_epochs))) {
    model$train(TRUE)
    disc$train(TRUE)
    perm <- sample.int(n)
    sums <- c(loss = 0, recon = 0, treat = 0, outcome = 0, disent = 0)

    for (start in seq(1L, n, by = batch_size)) {
      idx <- perm[start:min(start + batch_size - 1L, n)]
      xb <- x_t[idx, , drop = FALSE]
      tb <- t_t[idx]
      yb <- y_t[idx]
      bsz <- length(idx)

      opt_disc$zero_grad()
      z <- model$encode(xb)
      zt_det <- z$zt$detach()
      zy_det <- z$zy$detach()
      d_perm <- torch::torch_randperm(bsz, device = dev, dtype = torch::torch_long())$add(1L)
      zy_shuf <- zy_det$index_select(1L, d_perm)
      real_logit <- disc(zt_det, zy_det)
      fake_logit <- disc(zt_det, zy_shuf)
      d_loss <- (
        torch::nnf_binary_cross_entropy_with_logits(real_logit, torch::torch_ones_like(real_logit), reduction = "mean") +
          torch::nnf_binary_cross_entropy_with_logits(fake_logit, torch::torch_zeros_like(fake_logit), reduction = "mean")
      ) / 2.0
      d_loss$backward()
      opt_disc$step()

      opt_model$zero_grad()
      out <- model(xb, tb)
      losses <- .deepnet_causalegm_loss(
        x_in = xb, t = tb, y = yb, out = out, disc = disc,
        lambda_recon = lambda_recon,
        lambda_treat = lambda_treat,
        lambda_outcome = lambda_outcome,
        lambda_disent = lambda_disent
      )
      losses$total$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = max_grad_norm)
      opt_model$step()

      sums["loss"] <- sums["loss"] + as.numeric(losses$total$item())
      sums["recon"] <- sums["recon"] + as.numeric(losses$recon$item())
      sums["treat"] <- sums["treat"] + as.numeric(losses$treat$item())
      sums["outcome"] <- sums["outcome"] + as.numeric(losses$outcome$item())
      sums["disent"] <- sums["disent"] + as.numeric(losses$disent$item())
    }

    metrics <- as.list(sums / n_batches)
    history$train[[epoch]] <- metrics
    if (isTRUE(verbose) && (epoch == 1L || epoch %% 10L == 0L)) {
      message(
        "CausalEGM epoch ", epoch,
        " loss=", round(metrics$loss, 4),
        " recon=", round(metrics$recon, 4),
        " treat=", round(metrics$treat, 4),
        " outcome=", round(metrics$outcome, 4),
        " disent=", round(metrics$disent, 4)
      )
    }
  }

  structure(
    list(
      model = model,
      disc = disc,
      X_names = colnames(X),
      type = "causal_egm_torch",
      device = dev,
      latent_dims = c(dim_c = as.integer(dim_c), dim_t = as.integer(dim_t), dim_y = as.integer(dim_y)),
      history = history
    ),
    class = "causal_egm"
  )
}

#' Alias for \code{causal_egm}
#'
#' @inheritParams causal_egm
#' @return Object of class \code{causal_egm}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # CausalEGM(...)
#' }
#' @export
CausalEGM <- function(X, treatment, y, ...) {
  causal_egm(X = X, treatment = treatment, y = y, ...)
}

#' Predict from CausalEGM
#'
#' @param object Fitted \code{causal_egm} object.
#' @param newdata Covariate matrix or data.frame.
#' @param type One of \code{"ite"}, \code{"mu0"}, \code{"mu1"}, \code{"propensity"}, or \code{"latent"}.
#' @param ... Ignored.
#' @return Numeric vector (for \code{"ite"}, \code{"mu0"}, \code{"mu1"}, \code{"propensity"})
#'   or matrix (for \code{"latent"}).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.causal_egm(...)
#' }
#' @export
predict.causal_egm <- function(object, newdata,
                               type = c("ite", "mu0", "mu1", "propensity", "latent"),
                               ...) {
  type <- match.arg(type)
  x <- as.matrix(newdata)
  model <- object$model
  model$eval()
  torch::with_no_grad({
    x_t <- torch::torch_tensor(x, dtype = torch::torch_float32(), device = object$device)
    z <- model$encode(x_t)
    if (type == "latent")
      return(as.matrix(torch::torch_cat(list(z$zc, z$zt, z$zy), dim = 2L)$to(device = "cpu")))

    n <- x_t$size(1)
    zeros <- torch::torch_zeros(c(n), dtype = torch::torch_float32(), device = object$device)
    ones <- torch::torch_ones(c(n), dtype = torch::torch_float32(), device = object$device)
    y0 <- model$outcome_pred(z$zc, z$zy, zeros)
    y1 <- model$outcome_pred(z$zc, z$zy, ones)

    if (type == "mu0") return(as.numeric(y0$to(device = "cpu")))
    if (type == "mu1") return(as.numeric(y1$to(device = "cpu")))
    if (type == "propensity") {
      ps <- torch::torch_sigmoid(model$treatment_logit(z$zc, z$zt))
      return(as.numeric(ps$to(device = "cpu")))
    }
    as.numeric((y1 - y0)$to(device = "cpu"))
  })
}

# --- Time Series deepCausalML / Neural Granger Causality Models ---

.deepnet_ngc_build_lag_dataset <- function(data, lag = 5L) {
  x <- as.matrix(data)
  storage.mode(x) <- "double"
  lag <- as.integer(lag)
  if (nrow(x) <= lag) stop("Need more rows than lag for lagged dataset.", call. = FALSE)
  n <- nrow(x) - lag
  d <- ncol(x)

  x_flat <- matrix(0, nrow = n, ncol = d * lag)
  x_seq <- array(0, dim = c(n, lag, d))
  y <- matrix(0, nrow = n, ncol = d)

  for (i in seq_len(n)) {
    win <- x[i:(i + lag - 1L), , drop = FALSE]
    x_flat[i, ] <- as.numeric(t(win))
    x_seq[i, , ] <- win
    y[i, ] <- x[i + lag, ]
  }

  list(
    x_flat = x_flat,
    x_seq = x_seq,
    y = y
  )
}

.deepnet_ngc_group_lasso_penalty <- function(weight_matrix, d, lag) {
  penalty <- torch::torch_tensor(0, dtype = torch::torch_float32(), device = weight_matrix$device)
  for (j in seq_len(d)) {
    lo <- (j - 1L) * lag + 1L
    hi <- j * lag
    group <- weight_matrix[, lo:hi, drop = FALSE]
    penalty <- penalty + torch::torch_sqrt((group * group)$sum())
  }
  penalty
}

.deepnet_ngc_component_mlp <- function(d, lag, hidden = 32L) {
  torch::nn_module(
    initialize = function() {
      self$net <- torch::nn_sequential(
        torch::nn_linear(as.integer(d * lag), as.integer(hidden)),
        torch::nn_relu(),
        torch::nn_linear(as.integer(hidden), as.integer(hidden)),
        torch::nn_relu(),
        torch::nn_linear(as.integer(hidden), 1L)
      )
    },
    forward = function(x) {
      self$net(x)$squeeze(-1)
    },
    first_layer_weights = function() self$net[[1]]$weight
  )
}

.deepnet_ngc_cmlp <- function(d, lag, hidden = 32L) {
  ComponentMLP <- .deepnet_ngc_component_mlp(d = d, lag = lag, hidden = hidden)
  torch::nn_module(
    initialize = function() {
      self$d <- as.integer(d)
      self$lag <- as.integer(lag)
      self$models <- torch::nn_module_list(lapply(seq_len(self$d), function(i) ComponentMLP()))
    },
    forward = function(x) {
      torch::torch_stack(lapply(self$models, function(m) m(x)), dim = 2L)
    },
    group_lasso_loss = function(lam = 0.01) {
      penalty <- torch::torch_tensor(0, dtype = torch::torch_float32(), device = self$models[[1]]$first_layer_weights()$device)
      for (ii in seq_along(self$models)) {
        penalty <- penalty + .deepnet_ngc_group_lasso_penalty(self$models[[ii]]$first_layer_weights(), self$d, self$lag)
      }
      penalty * lam
    },
    causal_matrix = function() {
      cmat <- matrix(0, nrow = self$d, ncol = self$d)
      for (i in seq_len(self$d)) {
        w <- as.matrix(self$models[[i]]$first_layer_weights()$detach()$to(device = "cpu"))
        for (j in seq_len(self$d)) {
          lo <- (j - 1L) * self$lag + 1L
          hi <- j * self$lag
          cmat[i, j] <- sqrt(sum(w[, lo:hi, drop = FALSE]^2))
        }
      }
      cmat
    }
  )
}

.deepnet_ngc_component_lstm <- function(d, hidden = 32L) {
  torch::nn_module(
    initialize = function() {
      self$lstm <- torch::nn_lstm(input_size = as.integer(d), hidden_size = as.integer(hidden), num_layers = 1L, batch_first = TRUE)
      self$fc <- torch::nn_linear(as.integer(hidden), 1L)
    },
    forward = function(x_seq) {
      out <- self$lstm(x_seq)[[1]]
      last <- out[, out$size(2), , drop = FALSE]$squeeze(2)
      self$fc(last)$squeeze(-1)
    },
    input_weights = function() self$lstm$parameters[[1]]
  )
}

.deepnet_ngc_clstm <- function(d, hidden = 32L) {
  ComponentLSTM <- .deepnet_ngc_component_lstm(d = d, hidden = hidden)
  torch::nn_module(
    initialize = function() {
      self$d <- as.integer(d)
      self$models <- torch::nn_module_list(lapply(seq_len(self$d), function(i) ComponentLSTM()))
    },
    forward = function(x_seq) {
      torch::torch_stack(lapply(self$models, function(m) m(x_seq)), dim = 2L)
    },
    group_lasso_loss = function(lam = 0.01) {
      penalty <- torch::torch_tensor(0, dtype = torch::torch_float32(), device = self$models[[1]]$input_weights()$device)
      for (ii in seq_along(self$models)) {
        w <- self$models[[ii]]$input_weights()
        col_norms <- torch::torch_sqrt((w * w)$sum(dim = 1L) + 1e-8)
        penalty <- penalty + col_norms$sum()
      }
      penalty * lam
    },
    causal_matrix = function() {
      cmat <- matrix(0, nrow = self$d, ncol = self$d)
      for (i in seq_len(self$d)) {
        w <- as.matrix(self$models[[i]]$input_weights()$detach()$to(device = "cpu"))
        for (j in seq_len(self$d)) {
          cmat[i, j] <- sqrt(sum(w[, j]^2))
        }
      }
      cmat
    }
  )
}

.deepnet_ngc_economy_sru <- function(d, hidden = 32L) {
  torch::nn_module(
    initialize = function() {
      self$d <- as.integer(d)
      self$hidden <- as.integer(hidden)
      self$causal_logits <- torch::nn_parameter(torch::torch_zeros(c(self$d, self$d)))
      self$wx <- torch::nn_module_list(lapply(seq_len(self$d), function(i) torch::nn_linear(self$d, self$hidden)))
      self$wh <- torch::nn_module_list(lapply(seq_len(self$d), function(i) torch::nn_linear(self$hidden, self$hidden)))
      self$fc <- torch::nn_module_list(lapply(seq_len(self$d), function(i) torch::nn_linear(self$hidden, 1L)))
    },
    causal_mask = function() {
      torch::torch_sigmoid(self$causal_logits)
    },
    forward = function(x_seq) {
      batch <- x_seq$size(1)
      seq_len <- x_seq$size(2)
      mask <- self$causal_mask()
      h <- lapply(seq_len(self$d), function(i) torch::torch_zeros(c(batch, self$hidden), device = x_seq$device))
      for (tt in seq_len(seq_len)) {
        x_t <- x_seq[, tt, , drop = FALSE]$squeeze(2)
        new_h <- vector("list", self$d)
        for (i in seq_len(self$d)) {
          masked_x <- x_t * mask[i, ]$unsqueeze(1)
          new_h[[i]] <- torch::torch_tanh(self$wx[[i]](masked_x) + self$wh[[i]](h[[i]]))
        }
        h <- new_h
      }
      torch::torch_stack(lapply(seq_len(self$d), function(i) self$fc[[i]](h[[i]])$squeeze(-1)), dim = 2L)
    },
    sparsity_loss = function(lam = 0.01) {
      lam * self$causal_mask()$sum()
    },
    causal_matrix = function() {
      as.matrix(self$causal_mask()$detach()$to(device = "cpu"))
    }
  )
}

.deepnet_ngc_mlp_encoder <- function(d, lag, hidden = 32L, n_edge_types = 2L) {
  torch::nn_module(
    initialize = function() {
      self$d <- as.integer(d)
      self$n_edge_types <- as.integer(n_edge_types)
      self$embed <- torch::nn_sequential(
        torch::nn_linear(as.integer(lag), as.integer(hidden)),
        torch::nn_relu(),
        torch::nn_linear(as.integer(hidden), as.integer(hidden))
      )
      self$edge_mlp <- torch::nn_sequential(
        torch::nn_linear(as.integer(hidden * 2L), as.integer(hidden)),
        torch::nn_relu(),
        torch::nn_linear(as.integer(hidden), self$n_edge_types)
      )
    },
    forward = function(x_seq) {
      x <- x_seq$permute(c(1, 3, 2))
      h <- self$embed(x)
      h_i <- h$unsqueeze(3)$expand(c(-1, -1, self$d, -1))
      h_j <- h$unsqueeze(2)$expand(c(-1, self$d, -1, -1))
      edge_feat <- torch::torch_cat(list(h_i, h_j), dim = 4L)
      self$edge_mlp(edge_feat)
    }
  )
}

.deepnet_ngc_mlp_decoder <- function(d, lag, hidden = 32L, n_edge_types = 2L) {
  torch::nn_module(
    initialize = function() {
      self$d <- as.integer(d)
      self$hidden <- as.integer(hidden)
      self$msg_mlps <- torch::nn_module_list(lapply(seq_len(as.integer(n_edge_types)), function(i) {
        torch::nn_sequential(
          torch::nn_linear(as.integer(lag * 2L), self$hidden),
          torch::nn_relu(),
          torch::nn_linear(self$hidden, self$hidden)
        )
      }))
      self$out_mlp <- torch::nn_sequential(
        torch::nn_linear(as.integer(self$hidden + lag), self$hidden),
        torch::nn_relu(),
        torch::nn_linear(self$hidden, 1L)
      )
    },
    forward = function(x_seq, edge_probs) {
      batch <- x_seq$size(1)
      x <- x_seq$permute(c(1, 3, 2))
      agg_msg <- torch::torch_zeros(c(batch, self$d, self$hidden), device = x_seq$device)
      x_i <- x$unsqueeze(3)$expand(c(-1, -1, self$d, -1))
      x_j <- x$unsqueeze(2)$expand(c(-1, self$d, -1, -1))
      msg_input <- torch::torch_cat(list(x_i, x_j), dim = 4L)
      for (k in seq_along(self$msg_mlps)) {
        prob_k <- edge_probs[, , , k, drop = FALSE]$squeeze(4)
        msg <- self$msg_mlps[[k]](msg_input)
        agg_msg <- agg_msg + (prob_k$unsqueeze(4) * msg)$sum(dim = 3L)
      }
      node_feat <- torch::torch_cat(list(agg_msg, x), dim = 3L)
      self$out_mlp(node_feat)$squeeze(-1)
    }
  )
}

.deepnet_ngc_nri <- function(d, lag, hidden = 32L, n_edge_types = 2L, temp = 0.5) {
  Encoder <- .deepnet_ngc_mlp_encoder(d = d, lag = lag, hidden = hidden, n_edge_types = n_edge_types)
  Decoder <- .deepnet_ngc_mlp_decoder(d = d, lag = lag, hidden = hidden, n_edge_types = n_edge_types)
  torch::nn_module(
    initialize = function() {
      self$encoder <- Encoder()
      self$decoder <- Decoder()
      self$temp <- temp
      self$n_edge_types <- as.integer(n_edge_types)
    },
    forward = function(x_seq) {
      logits <- self$encoder(x_seq)
      edge_probs <- torch::nnf_softmax(logits / self$temp, dim = 4L)
      preds <- self$decoder(x_seq, edge_probs)
      list(preds = preds, logits = logits, edge_probs = edge_probs)
    },
    causal_matrix = function(x_seq_sample) {
      logits <- self$encoder(x_seq_sample)
      probs <- torch::nnf_softmax(logits, dim = 4L)
      as.matrix(probs[, , , 2, drop = FALSE]$squeeze(4)$mean(dim = 1L)$detach()$to(device = "cpu"))
    }
  )
}

.deepnet_ngc_train_model <- function(
    model, model_type,
    x_flat_tr, x_seq_tr, y_tr, x_flat_val, x_seq_val, y_val,
    lam = 0.01, epochs = 100L, batch_size = 64L, lr = 1e-3, device = "cpu", verbose = FALSE) {
  n <- nrow(y_tr)
  optimizer <- torch::optim_adam(model$parameters, lr = lr)
  mse <- torch::nn_mse_loss()
  history <- list(train_loss = numeric(epochs), val_loss = numeric(epochs))

  for (epoch in seq_len(epochs)) {
    model$train()
    idx <- sample.int(n)
    train_loss_sum <- 0
    n_batches <- 0L

    for (start in seq.int(1L, n, by = batch_size)) {
      bidx <- idx[start:min(start + batch_size - 1L, n)]
      xb <- if (identical(model_type, "cmlp")) {
        torch::torch_tensor(x_flat_tr[bidx, , drop = FALSE], dtype = torch::torch_float32(), device = device)
      } else {
        torch::torch_tensor(x_seq_tr[bidx, , , drop = FALSE], dtype = torch::torch_float32(), device = device)
      }
      yb <- torch::torch_tensor(y_tr[bidx, , drop = FALSE], dtype = torch::torch_float32(), device = device)

      optimizer$zero_grad()

      if (identical(model_type, "nri")) {
        out <- model(xb)
        preds <- out$preds
        logits <- out$logits
        k <- as.numeric(model$n_edge_types)
        log_prior <- log(1 / k)
        probs <- torch::nnf_softmax(logits, dim = 4L)
        log_probs <- torch::nnf_log_softmax(logits, dim = 4L)
        kl <- (probs * (log_probs - log_prior))$sum(dim = 4L)$mean()
        loss <- mse(preds, yb) + 0.001 * kl
      } else {
        preds <- model(xb)
        loss <- mse(preds, yb)
        if (identical(model_type, "cmlp") || identical(model_type, "clstm")) {
          loss <- loss + model$group_lasso_loss(lam)
        } else if (identical(model_type, "economysru")) {
          loss <- loss + model$sparsity_loss(lam)
        }
      }

      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1)
      optimizer$step()

      train_loss_sum <- train_loss_sum + as.numeric(loss$item())
      n_batches <- n_batches + 1L
    }

    model$eval()
    torch::with_no_grad({
      xv <- if (identical(model_type, "cmlp")) {
        torch::torch_tensor(x_flat_val, dtype = torch::torch_float32(), device = device)
      } else {
        torch::torch_tensor(x_seq_val, dtype = torch::torch_float32(), device = device)
      }
      yv <- torch::torch_tensor(y_val, dtype = torch::torch_float32(), device = device)
      if (identical(model_type, "nri")) {
        preds_v <- model(xv)$preds
      } else {
        preds_v <- model(xv)
      }
      history$val_loss[epoch] <- as.numeric(mse(preds_v, yv)$item())
    })

    history$train_loss[epoch] <- train_loss_sum / max(n_batches, 1L)
    if (isTRUE(verbose) && (epoch %% 20L == 0L || epoch == 1L)) {
      message(sprintf(
        "[%s] Epoch %d | Train %.5f | Val %.5f",
        toupper(model_type), epoch, history$train_loss[epoch], history$val_loss[epoch]
      ))
    }
  }
  history
}

#' Neural Granger Causality Models for Multivariate Time Series
#'
#' Fits neural Granger-causality models translated from
#' \code{01_neural_granger_causality_models_tutorial.ipynb}:
#' \code{cMLP}, \code{cLSTM}, \code{EconomySRU}, and \code{NRI}.
#'
#' @param data Numeric matrix/data.frame with rows as time points and columns as variables.
#' @param lag Number of lag steps for supervised framing (default 5).
#' @param models Character vector subset of \code{c("cmlp","clstm","economysru","nri")}.
#' @param hidden Hidden width for all models (default 32).
#' @param lam Sparsity regularization strength for cMLP/cLSTM/EconomySRU (default 0.005).
#' @param epochs Number of training epochs (default 60).
#' @param batch_size Batch size (default 32).
#' @param lr Learning rate (default 5e-4).
#' @param val_split Validation fraction in (0, 1) (default 0.2).
#' @param n_edge_types Number of edge types for NRI (default 2).
#' @param temp Softmax temperature for NRI edge probabilities (default 0.5).
#' @param device Torch device string (\code{"cpu"} or \code{"cuda"}). Default auto-select.
#' @param verbose If TRUE, print training progress.
#' @return Object of class \code{neural_granger_ml} with trained models, histories,
#'   validation losses, and inferred causal matrices.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # neural_granger_ml(...)
#' }
#' @export
neural_granger_ml <- function(
    data,
    lag = 5L,
    models = c("cmlp", "clstm", "economysru", "nri"),
    hidden = 32L,
    lam = 0.005,
    epochs = 60L,
    batch_size = 32L,
    lr = 5e-4,
    val_split = 0.2,
    n_edge_types = 2L,
    temp = 0.5,
    device = NULL,
    verbose = FALSE) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("neural_granger_ml() requires package 'torch'.", call. = FALSE)
  x <- as.matrix(data)
  if (!is.numeric(x)) stop("`data` must be numeric.", call. = FALSE)

  dev <- .deepnet_select_device(device)
  prep <- .deepnet_ngc_build_lag_dataset(x, lag = lag)
  n <- nrow(prep$y)
  split <- max(1L, min(n - 1L, floor((1 - val_split) * n)))
  x_flat_tr <- prep$x_flat[seq_len(split), , drop = FALSE]
  x_flat_val <- prep$x_flat[(split + 1L):n, , drop = FALSE]
  x_seq_tr <- prep$x_seq[seq_len(split), , , drop = FALSE]
  x_seq_val <- prep$x_seq[(split + 1L):n, , , drop = FALSE]
  y_tr <- prep$y[seq_len(split), , drop = FALSE]
  y_val <- prep$y[(split + 1L):n, , drop = FALSE]

  model_keys <- c("cmlp", "clstm", "economysru", "nri")
  req_models <- unique(tolower(models))
  bad <- setdiff(req_models, model_keys)
  if (length(bad)) stop("Unknown models: ", paste(bad, collapse = ", "), call. = FALSE)

  d <- ncol(x)
  fit_models <- list()
  histories <- list()
  val_mse <- list()
  causal_matrices <- list()

  for (m in req_models) {
    model <- switch(
      m,
      cmlp = .deepnet_ngc_cmlp(d = d, lag = lag, hidden = hidden)(),
      clstm = .deepnet_ngc_clstm(d = d, hidden = hidden)(),
      economysru = .deepnet_ngc_economy_sru(d = d, hidden = hidden)(),
      nri = .deepnet_ngc_nri(d = d, lag = lag, hidden = hidden, n_edge_types = n_edge_types, temp = temp)()
    )
    model <- model$to(device = dev)
    hist <- .deepnet_ngc_train_model(
      model = model, model_type = m,
      x_flat_tr = x_flat_tr, x_seq_tr = x_seq_tr, y_tr = y_tr,
      x_flat_val = x_flat_val, x_seq_val = x_seq_val, y_val = y_val,
      lam = lam, epochs = as.integer(epochs), batch_size = as.integer(batch_size),
      lr = lr, device = dev, verbose = verbose
    )

    model$eval()
    torch::with_no_grad({
      xv <- if (identical(m, "cmlp")) {
        torch::torch_tensor(x_flat_val, dtype = torch::torch_float32(), device = dev)
      } else {
        torch::torch_tensor(x_seq_val, dtype = torch::torch_float32(), device = dev)
      }
      yv <- torch::torch_tensor(y_val, dtype = torch::torch_float32(), device = dev)
      preds <- if (identical(m, "nri")) model(xv)$preds else model(xv)
      val_mse[[m]] <- as.numeric(torch::nnf_mse_loss(preds, yv)$item())
    })

    causal_matrices[[m]] <- if (identical(m, "nri")) {
      n_sample <- min(128L, dim(x_seq_val)[1])
      x_sample <- torch::torch_tensor(x_seq_val[seq_len(n_sample), , , drop = FALSE], dtype = torch::torch_float32(), device = dev)
      model$causal_matrix(x_sample)
    } else {
      model$causal_matrix()
    }

    fit_models[[m]] <- model
    histories[[m]] <- hist
  }

  structure(
    list(
      models = fit_models,
      histories = histories,
      val_mse = val_mse,
      causal_matrices = causal_matrices,
      lag = as.integer(lag),
      var_names = colnames(x),
      device = dev
    ),
    class = "neural_granger_ml"
  )
}

#' Alias for \code{neural_granger_ml}
#'
#' @inheritParams neural_granger_ml
#' @return Object of class \code{neural_granger_ml}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # neuralGrangerML(...)
#' }
#' @export
neuralGrangerML <- function(data, ...) {
  neural_granger_ml(data = data, ...)
}

#' Predict from neural Granger models
#'
#' @param object Fitted \code{neural_granger_ml} object.
#' @param model One of \code{"cmlp"}, \code{"clstm"}, \code{"economysru"}, \code{"nri"}.
#' @param x_lagged New lagged inputs as matrix (\code{n x (d*lag)}) for cMLP or
#'   3D array (\code{n x lag x d}) for sequence models.
#' @param ... Ignored.
#' @return Numeric matrix of predictions with one column per target variable.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.neural_granger_ml(...)
#' }
#' @export
predict.neural_granger_ml <- function(object, model = c("cmlp", "clstm", "economysru", "nri"), x_lagged, ...) {
  model <- match.arg(model)
  if (is.null(object$models[[model]])) stop("Model not fitted: ", model, call. = FALSE)
  fit <- object$models[[model]]
  fit$eval()
  dev <- if (is.null(object$device)) "cpu" else object$device
  torch::with_no_grad({
    x_t <- if (identical(model, "cmlp")) {
      torch::torch_tensor(as.matrix(x_lagged), dtype = torch::torch_float32(), device = dev)
    } else {
      torch::torch_tensor(x_lagged, dtype = torch::torch_float32(), device = dev)
    }
    preds <- if (identical(model, "nri")) fit(x_t)$preds else fit(x_t)
    as.matrix(preds$detach()$to(device = "cpu"))
  })
}


# =============================================================================
# Structural Causal Models (SCMs) with Deep Components
# =============================================================================
# Models:
#   • DeepSCM    — Fixed-graph SCM; nonlinear structural equations with
#                  variational noise encoders (ELBO-style training).
#   • DECI       — Deep End-to-end Causal Inference; jointly learns the causal
#                  graph and structural equations, NOTEARS acyclicity penalty.
#   • DynoTEARS  — Lagged causal discovery for multivariate time series with
#                  augmented-Lagrangian DAG constraint.
# =============================================================================

# --- SCM utility functions ---------------------------------------------------

.scm_acyclicity_penalty <- function(A) {
  d_local <- A$shape[1]
  torch::torch_trace(torch::torch_matrix_exp(A * A)) - d_local
}

.scm_threshold_adjacency <- function(A_soft, threshold = 0.35) {
  A_bin <- (abs(A_soft) > threshold) * 1.0
  diag(A_bin) <- 0.0
  A_bin
}

# --- StructuralEquationNet ---------------------------------------------------

.deepnet_scm_se_net <- function(n_parents, lag, latent_dim = 4L, hidden = 64L) {
  n_parents  <- as.integer(n_parents)
  lag        <- as.integer(lag)
  latent_dim <- as.integer(latent_dim)
  hidden     <- as.integer(hidden)
  torch::nn_module(
    "StructuralEquationNet",
    initialize = function() {
      self$net <- torch::nn_sequential(
        torch::nn_linear(n_parents * lag + latent_dim, hidden),
        torch::nn_silu(),
        torch::nn_layer_norm(hidden),
        torch::nn_linear(hidden, hidden),
        torch::nn_silu(),
        torch::nn_linear(hidden, 1L)
      )
    },
    forward = function(pa_ctx, u) {
      self$net(torch::torch_cat(list(pa_ctx, u), dim = -1L))
    }
  )
}

# --- NoiseEncoder ------------------------------------------------------------

.deepnet_scm_noise_encoder <- function(d, lag, latent_dim = 4L, hidden = 64L) {
  d          <- as.integer(d)
  lag        <- as.integer(lag)
  latent_dim <- as.integer(latent_dim)
  hidden     <- as.integer(hidden)
  torch::nn_module(
    "NoiseEncoder",
    initialize = function() {
      self$net <- torch::nn_sequential(
        torch::nn_linear(d * lag, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, hidden),
        torch::nn_relu()
      )
      self$mu_head     <- torch::nn_linear(hidden, latent_dim)
      self$logvar_head <- torch::nn_linear(hidden, latent_dim)
    },
    forward = function(x_flat) {
      h <- self$net(x_flat)
      list(
        mu     = self$mu_head(h),
        logvar = self$logvar_head(h)$clamp(-4, 4)
      )
    },
    reparameterize = function(mu, logvar) {
      std <- torch::torch_exp(0.5 * logvar)
      mu + torch::torch_randn_like(std) * std
    }
  )
}

# --- DeepSCM (fixed-graph) ---------------------------------------------------

.deepnet_deep_scm_module <- function(d, lag, adjacency, latent_dim = 4L, hidden = 64L) {
  d          <- as.integer(d)
  lag        <- as.integer(lag)
  latent_dim <- as.integer(latent_dim)
  hidden     <- as.integer(hidden)
  torch::nn_module(
    "DeepSCM",
    initialize = function() {
      self$d    <- d
      self$A_buf <- torch::torch_tensor(adjacency, dtype = torch::torch_float32())
      n_parents_vec <- pmax(1L, as.integer(colSums(adjacency)))
      self$se_nets  <- torch::nn_module_list(
        lapply(seq_len(d), function(i)
          .deepnet_scm_se_net(n_parents_vec[i], lag, latent_dim, hidden)()
        )
      )
      self$encoders <- torch::nn_module_list(
        lapply(seq_len(d), function(i)
          .deepnet_scm_noise_encoder(d, lag, latent_dim, hidden)()
        )
      )
    },
    get_parent_context = function(x_seq, i) {
      A_cpu    <- self$A_buf$to(device = "cpu")
      col_i    <- as.numeric(A_cpu[, i])
      parent_r <- which(col_i > 0.5)
      if (length(parent_r) == 0L) {
        return(x_seq[, , i, drop = FALSE]$reshape(c(x_seq$size(1L), -1L)))
      }
      x_seq[, , parent_r, drop = FALSE]$reshape(c(x_seq$size(1L), -1L))
    },
    forward = function(x_seq) {
      x_flat    <- x_seq$reshape(c(x_seq$size(1L), -1L))
      kl_total  <- torch::torch_tensor(0.0, device = x_seq$device)
      preds_lst <- vector("list", self$d)
      for (i in seq_len(self$d)) {
        enc_out  <- self$encoders[[i]](x_flat)
        mu       <- enc_out$mu
        logvar   <- enc_out$logvar
        u_i      <- self$encoders[[i]]$reparameterize(mu, logvar)
        kl_i     <- -0.5 * (1 + logvar - mu$pow(2) - logvar$exp())$sum(-1L)$mean()
        kl_total <- kl_total + kl_i
        pa_ctx   <- self$get_parent_context(x_seq, i)
        preds_lst[[i]] <- self$se_nets[[i]](pa_ctx, u_i)
      }
      list(
        preds    = torch::torch_cat(preds_lst, dim = -1L),
        kl_total = kl_total
      )
    },
    intervene = function(x_seq, target_var, do_value) {
      torch::with_no_grad({
        x_do <- x_seq$clone()
        x_do[, , target_var] <- do_value
        out <- self$forward(x_do)
        out$preds
      })
    }
  )
}

# --- DECIAdjacency -----------------------------------------------------------

.deepnet_deci_adjacency_module <- function(d) {
  d <- as.integer(d)
  torch::nn_module(
    "DECIAdjacency",
    initialize = function() {
      self$d      <- d
      self$logits <- torch::nn_parameter(
        torch::torch_randn(c(d, d)) * 0.1
      )
    },
    forward = function() {
      A <- torch::torch_sigmoid(self$logits)
      A * (1.0 - torch::torch_eye(self$d, device = self$logits$device))
    }
  )
}

# --- DECIEncoder -------------------------------------------------------------

.deepnet_deci_encoder_module <- function(d, lag, latent_dim = 8L, hidden = 64L) {
  d          <- as.integer(d)
  lag        <- as.integer(lag)
  latent_dim <- as.integer(latent_dim)
  hidden     <- as.integer(hidden)
  torch::nn_module(
    "DECIEncoder",
    initialize = function() {
      self$d          <- d
      self$latent_dim <- latent_dim
      self$net <- torch::nn_sequential(
        torch::nn_linear(d * lag, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, hidden),
        torch::nn_relu()
      )
      self$mu_head     <- torch::nn_linear(hidden, d * latent_dim)
      self$logvar_head <- torch::nn_linear(hidden, d * latent_dim)
    },
    forward = function(x_seq) {
      b <- x_seq$size(1L)
      h <- self$net(x_seq$reshape(c(b, -1L)))
      mu     <- self$mu_head(h)$view(c(b, self$d, self$latent_dim))
      logvar <- self$logvar_head(h)$view(c(b, self$d, self$latent_dim))$clamp(-4, 4)
      list(mu = mu, logvar = logvar)
    },
    reparameterize = function(mu, logvar) {
      std <- torch::torch_exp(0.5 * logvar)
      mu + torch::torch_randn_like(std) * std
    }
  )
}

# --- DECIDecoder -------------------------------------------------------------

.deepnet_deci_decoder_module <- function(d, lag, latent_dim = 8L, hidden = 64L) {
  d          <- as.integer(d)
  lag        <- as.integer(lag)
  latent_dim <- as.integer(latent_dim)
  hidden     <- as.integer(hidden)
  torch::nn_module(
    "DECIDecoder",
    initialize = function() {
      self$d        <- d
      self$edge_net <- torch::nn_sequential(
        torch::nn_linear(lag + 2L, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, hidden %/% 2L),
        torch::nn_relu(),
        torch::nn_linear(hidden %/% 2L, 1L)
      )
      self$var_embed  <- torch::nn_embedding(d, 1L)
      self$noise_net  <- torch::nn_sequential(
        torch::nn_linear(latent_dim, hidden %/% 2L),
        torch::nn_silu(),
        torch::nn_linear(hidden %/% 2L, 1L)
      )
    },
    forward = function(x_seq, A, U) {
      b        <- x_seq$size(1L)
      dev      <- x_seq$device
      preds_lst <- vector("list", self$d)
      for (i in seq_len(self$d)) {
        agg <- torch::torch_zeros(c(b, 1L), device = dev)
        for (j in seq_len(self$d)) {
          if (i == j) next
          x_j  <- x_seq[, , j, drop = FALSE]$squeeze(3L)
          src  <- self$var_embed(
            torch::torch_tensor(j, dtype = torch::torch_long(), device = dev)
          )$expand(c(b, 1L))
          tgt  <- self$var_embed(
            torch::torch_tensor(i, dtype = torch::torch_long(), device = dev)
          )$expand(c(b, 1L))
          msg  <- self$edge_net(torch::torch_cat(list(x_j, src, tgt), dim = -1L))
          agg  <- agg + A[i, j] * msg
        }
        preds_lst[[i]] <- agg + self$noise_net(U[, i, ])
      }
      torch::torch_cat(preds_lst, dim = -1L)
    }
  )
}

# --- DECI (joint graph + structural equations) --------------------------------

.deepnet_deci_module <- function(d, lag, latent_dim = 8L, hidden = 64L) {
  d          <- as.integer(d)
  lag        <- as.integer(lag)
  latent_dim <- as.integer(latent_dim)
  hidden     <- as.integer(hidden)
  torch::nn_module(
    "DECI",
    initialize = function() {
      self$adjacency <- .deepnet_deci_adjacency_module(d)()
      self$encoder   <- .deepnet_deci_encoder_module(d, lag, latent_dim, hidden)()
      self$decoder   <- .deepnet_deci_decoder_module(d, lag, latent_dim, hidden)()
    },
    forward = function(x_seq) {
      A             <- self$adjacency()
      enc_out       <- self$encoder(x_seq)
      mu            <- enc_out$mu
      logvar        <- enc_out$logvar
      U             <- self$encoder$reparameterize(mu, logvar)
      preds         <- self$decoder(x_seq, A, U)
      kl            <- -0.5 * (1 + logvar - mu$pow(2) - logvar$exp())$sum(-1L)$mean()
      list(preds = preds, kl = kl, A = A)
    },
    elbo_loss = function(x_seq, y, lam_kl = 0.05, lam_dag = 1.0, lam_sparse = 0.01) {
      out       <- self$forward(x_seq)
      preds     <- out$preds
      kl        <- out$kl
      A         <- out$A
      recon     <- torch::nnf_mse_loss(preds, y)
      dag_pen   <- .scm_acyclicity_penalty(A)
      sparse_pen <- A$sum()
      loss      <- recon + lam_kl * kl + lam_dag * dag_pen + lam_sparse * sparse_pen
      list(
        loss    = loss,
        recon   = recon$item(),
        kl      = kl$item(),
        dag     = dag_pen$item(),
        sparse  = sparse_pen$item(),
        A       = A
      )
    },
    causal_matrix = function() {
      torch::with_no_grad({
        as.matrix(self$adjacency()$detach()$to(device = "cpu"))
      })
    },
    compute_ate = function(x_seq, source, target, do_values = c(-1.0, 1.0), n_samples = 200L) {
      self$eval()
      torch::with_no_grad({
        n_use  <- min(n_samples, x_seq$size(1L))
        x_seed <- x_seq[seq_len(n_use), , ]$clone()
        A      <- self$adjacency()
        vals   <- setNames(numeric(length(do_values)), as.character(do_values))
        for (v in do_values) {
          x_do <- x_seed$clone()
          x_do[, , source] <- v
          enc_out <- self$encoder(x_do)
          mu_do   <- enc_out$mu
          lv_do   <- enc_out$logvar
          tmeans  <- numeric(10L)
          for (s in seq_len(10L)) {
            U     <- self$encoder$reparameterize(mu_do, lv_do)
            preds <- self$decoder(x_do, A, U)
            tmeans[s] <- preds[, target]$mean()$item()
          }
          vals[as.character(v)] <- mean(tmeans)
        }
        vals[as.character(do_values[2L])] - vals[as.character(do_values[1L])]
      })
    }
  )
}

# --- DynoTEARS (lagged DAG-constrained discovery) ----------------------------

.deepnet_dynotears_module <- function(n_vars, lag) {
  n_vars <- as.integer(n_vars)
  lag    <- as.integer(lag)
  torch::nn_module(
    "DynoTEARS",
    initialize = function() {
      self$n_vars <- n_vars
      self$lag    <- lag
      for (k in seq_len(lag)) {
        self[[paste0("W_lag_", k)]] <- torch::nn_parameter(
          torch::torch_zeros(c(n_vars, n_vars))
        )
      }
    },
    forward = function(x) {
      pred <- torch::torch_zeros(c(x$size(1L), self$n_vars), device = x$device)
      for (k in seq_len(self$lag)) {
        W    <- self[[paste0("W_lag_", k)]]
        pred <- pred + x[, self$lag - k + 1L, ] $matmul(W$t())
      }
      pred
    },
    dag_penalty = function() {
      W_total <- torch::torch_zeros(c(self$n_vars, self$n_vars),
                                    device = self[[paste0("W_lag_", 1L)]]$device)
      for (k in seq_len(self$lag)) {
        W_total <- W_total + self[[paste0("W_lag_", k)]]$abs()
      }
      torch::torch_trace(torch::torch_matrix_exp(W_total * W_total)) - self$n_vars
    },
    l1_penalty = function() {
      total <- torch::torch_tensor(0.0,
                                   device = self[[paste0("W_lag_", 1L)]]$device)
      for (k in seq_len(self$lag)) {
        total <- total + self[[paste0("W_lag_", k)]]$abs()$sum()
      }
      total
    },
    get_causal_matrix = function(threshold = 0.1) {
      torch::with_no_grad({
        W_agg <- matrix(0.0, self$n_vars, self$n_vars)
        for (k in seq_len(self$lag)) {
          W_agg <- W_agg + abs(as.matrix(
            self[[paste0("W_lag_", k)]]$detach()$to(device = "cpu")
          ))
        }
        A_bin <- (W_agg > threshold) * 1L
        diag(A_bin) <- 0L
        list(A = A_bin, W_agg = W_agg)
      })
    }
  )
}

# --- Training helpers --------------------------------------------------------

.deepnet_fit_deep_scm <- function(x_seq, adjacency, lag, latent_dim, hidden,
                                  n_epochs, batch_size, lr, lam_kl, verbose, device) {
  dev  <- device
  d    <- dim(x_seq)[3L]
  T_n  <- dim(x_seq)[1L]

  x_t  <- torch::torch_tensor(x_seq, dtype = torch::torch_float32(), device = dev)
  y_t  <- x_t[, lag, ]

  ds   <- torch::tensor_dataset(x_t, y_t)
  dl   <- torch::dataloader(ds, batch_size = as.integer(batch_size), shuffle = TRUE)

  model <- .deepnet_deep_scm_module(d, lag, adjacency, latent_dim, hidden)()
  model <- model$to(device = dev)
  opt   <- torch::optim_adam(model$parameters, lr = lr)

  train_losses <- numeric(n_epochs)
  for (epoch in seq_len(n_epochs)) {
    model$train()
    ep_loss <- 0.0
    n_batches <- 0L
    coro::loop(for (batch in dl) {
      xb <- batch[[1]]$to(device = dev)
      yb <- batch[[2]]$to(device = dev)
      out  <- model(xb)
      loss <- torch::nnf_mse_loss(out$preds, yb) + lam_kl * out$kl_total
      opt$zero_grad()
      loss$backward()
      opt$step()
      ep_loss   <- ep_loss + loss$item()
      n_batches <- n_batches + 1L
    })
    train_losses[epoch] <- ep_loss / max(1L, n_batches)
    if (verbose && (epoch %% 5L == 0L || epoch == 1L))
      message("[DeepSCM] epoch=", sprintf("%02d", epoch),
              " loss=", round(train_losses[epoch], 4L))
  }
  list(
    model        = model,
    adjacency    = adjacency,
    lag          = lag,
    var_names    = if (!is.null(colnames(x_seq[1L, , , drop = FALSE]))) colnames(x_seq[1L, , , drop = FALSE]) else paste0("X", seq_len(d)),
    train_losses = train_losses,
    device       = dev,
    class_type   = "deep_scm"
  )
}

.deepnet_fit_deci <- function(x_seq, lag, latent_dim, hidden, n_epochs, batch_size,
                               lr, lam_kl, lam_dag, lam_sparse, threshold,
                               verbose, device) {
  dev  <- device
  d    <- dim(x_seq)[3L]

  x_t  <- torch::torch_tensor(x_seq, dtype = torch::torch_float32(), device = dev)
  y_t  <- x_t[, lag, ]

  ds   <- torch::tensor_dataset(x_t, y_t)
  dl   <- torch::dataloader(ds, batch_size = as.integer(batch_size), shuffle = TRUE)

  model <- .deepnet_deci_module(d, lag, latent_dim, hidden)()
  model <- model$to(device = dev)
  opt   <- torch::optim_adam(model$parameters, lr = lr)

  train_losses <- numeric(n_epochs)
  last_out     <- NULL
  for (epoch in seq_len(n_epochs)) {
    model$train()
    ep_loss <- 0.0
    n_batches <- 0L
    coro::loop(for (batch in dl) {
      xb <- batch[[1]]$to(device = dev)
      yb <- batch[[2]]$to(device = dev)
      out  <- model$elbo_loss(xb, yb, lam_kl = lam_kl, lam_dag = lam_dag,
                              lam_sparse = lam_sparse)
      opt$zero_grad()
      out$loss$backward()
      opt$step()
      ep_loss   <- ep_loss + out$loss$item()
      n_batches <- n_batches + 1L
      last_out  <- out
    })
    train_losses[epoch] <- ep_loss / max(1L, n_batches)
    if (verbose && (epoch %% 10L == 0L || epoch == 1L))
      message("[DECI] epoch=", sprintf("%02d", epoch),
              " loss=", round(train_losses[epoch], 4L),
              " dag=",  if (!is.null(last_out)) round(last_out$dag, 4L) else "NA",
              " sparse=", if (!is.null(last_out)) round(last_out$sparse, 2L) else "NA")
  }

  A_soft <- model$causal_matrix()
  A_bin  <- .scm_threshold_adjacency(A_soft, threshold = threshold)
  dnames <- if (!is.null(colnames(x_seq[1L, , , drop = FALSE]))) colnames(x_seq[1L, , , drop = FALSE]) else paste0("X", seq_len(d))
  rownames(A_soft) <- colnames(A_soft) <- dnames
  rownames(A_bin)  <- colnames(A_bin)  <- dnames

  list(
    model        = model,
    A_soft       = A_soft,
    A_binary     = A_bin,
    lag          = lag,
    threshold    = threshold,
    var_names    = dnames,
    train_losses = train_losses,
    device       = dev,
    class_type   = "deci_model"
  )
}

.deepnet_fit_dynotears <- function(x_seq, y_mat, lag, n_epochs, batch_size, lr,
                                    lambda_l1, rho_init, threshold, verbose, device) {
  dev    <- device
  n_vars <- ncol(y_mat)

  x_t <- torch::torch_tensor(x_seq,  dtype = torch::torch_float32(), device = dev)
  y_t <- torch::torch_tensor(y_mat,  dtype = torch::torch_float32(), device = dev)

  ds  <- torch::tensor_dataset(x_t, y_t)
  dl  <- torch::dataloader(ds, batch_size = as.integer(batch_size), shuffle = TRUE)

  model <- .deepnet_dynotears_module(n_vars, lag)()
  model <- model$to(device = dev)
  opt   <- torch::optim_adam(model$parameters, lr = lr)

  rho   <- rho_init
  alpha <- 0.0
  train_losses <- numeric(n_epochs)
  dag_vals     <- numeric(n_epochs)

  for (epoch in seq_len(n_epochs)) {
    model$train()
    ep_mse    <- 0.0
    n_batches <- 0L
    coro::loop(for (batch in dl) {
      xb <- batch[[1]]$to(device = dev)
      yb <- batch[[2]]$to(device = dev)
      opt$zero_grad()
      h_val <- model$dag_penalty()
      loss  <- (torch::nnf_mse_loss(model(xb), yb)
                + lambda_l1 * model$l1_penalty()
                + (rho / 2) * h_val$pow(2)
                + alpha * h_val)
      loss$backward()
      opt$step()
      ep_mse    <- ep_mse + torch::nnf_mse_loss(model(xb), yb)$item()
      n_batches <- n_batches + 1L
    })
    h_np             <- model$dag_penalty()$item()
    train_losses[epoch] <- ep_mse / max(1L, n_batches)
    dag_vals[epoch]     <- abs(h_np)
    alpha               <- alpha + rho * h_np
    if (epoch %% 50L == 0L && epoch > 0L) rho <- min(rho * 2, 1e4)
    if (verbose && epoch %% 50L == 0L)
      message("[DynoTEARS] epoch=", sprintf("%3d", epoch),
              " MSE=",  round(train_losses[epoch], 4L),
              " h(W)=", round(h_np, 4L))
  }

  cm       <- model$get_causal_matrix(threshold = threshold)
  dnames   <- if (!is.null(colnames(y_mat))) colnames(y_mat) else paste0("X", seq_len(n_vars))
  rownames(cm$A)     <- colnames(cm$A)     <- dnames
  rownames(cm$W_agg) <- colnames(cm$W_agg) <- dnames

  list(
    model        = model,
    A_binary     = cm$A,
    W_agg        = cm$W_agg,
    lag          = lag,
    threshold    = threshold,
    var_names    = dnames,
    train_losses = train_losses,
    dag_vals     = dag_vals,
    device       = dev,
    class_type   = "dynotears"
  )
}

# --- Graph evaluation helper -------------------------------------------------

#' Evaluate causal graph recovery
#'
#' Computes precision, recall, F1, and Structural Hamming Distance (SHD)
#' between a predicted binary adjacency matrix and a ground-truth DAG.
#'
#' @param A_true Integer matrix (n x n): ground-truth adjacency (0/1).
#' @param A_pred Integer matrix (n x n): predicted adjacency (0/1).
#' @param name   Character label for the method (default \code{"Model"}).
#' @return Named list with \code{Method}, \code{TP}, \code{FP}, \code{FN},
#'   \code{Precision}, \code{Recall}, \code{F1}, and \code{SHD}.
#' @seealso \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' A_true <- matrix(c(0,1,0, 0,0,1, 0,0,0), 3, 3, byrow = TRUE)
#' A_pred <- matrix(c(0,1,0, 0,0,0, 0,0,0), 3, 3, byrow = TRUE)
#' evaluate_graph_recovery(A_true, A_pred, name = "MyModel")
#' }
#' @export
evaluate_graph_recovery <- function(A_true, A_pred, name = "Model") {
  A_true <- (as.matrix(A_true) > 0) * 1L
  A_pred <- (as.matrix(A_pred) > 0) * 1L
  mask   <- !diag(nrow(A_true))
  yt     <- as.integer(A_true[mask])
  yp     <- as.integer(A_pred[mask])
  TP <- sum(yt == 1L & yp == 1L)
  FP <- sum(yt == 0L & yp == 1L)
  FN <- sum(yt == 1L & yp == 0L)
  precision <- TP / (TP + FP + 1e-9)
  recall    <- TP / (TP + FN + 1e-9)
  f1        <- 2 * precision * recall / (precision + recall + 1e-9)
  list(
    Method    = name,
    TP        = TP,
    FP        = FP,
    FN        = FN,
    Precision = round(precision, 3L),
    Recall    = round(recall, 3L),
    F1        = round(f1, 3L),
    SHD       = FP + FN
  )
}

#' Plot a causal DAG as a heatmap
#'
#' Visualises an adjacency (or weight) matrix as a colour-coded heatmap using
#' \pkg{ggplot2} (or base graphics if \pkg{ggplot2} is unavailable).
#' Diagonal entries are masked.
#'
#' @param A         Numeric matrix (n x n) of edge weights or binary indicators.
#' @param var_names Character vector of variable names (length n).
#' @param title     Plot title.
#' @param ...       Ignored.
#' @return Invisibly returns \code{NULL}.
#' @seealso \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' A <- matrix(runif(25), 5, 5); diag(A) <- 0
#' plot_scm_dag(A, paste0("X", 1:5), "Example DAG")
#' }
#' @export
plot_scm_dag <- function(A, var_names = NULL, title = "Causal DAG", ...) {
  A <- as.matrix(A)
  n <- nrow(A)
  diag(A) <- NA_real_
  if (is.null(var_names)) var_names <- paste0("X", seq_len(n))
  if (requireNamespace("ggplot2", quietly = TRUE) &&
      requireNamespace("reshape2", quietly = TRUE)) {
    rownames(A) <- var_names
    colnames(A) <- var_names
    df <- reshape2::melt(A, na.rm = TRUE)
    colnames(df) <- c("Target", "Source", "Weight")
    df$Target <- factor(df$Target, levels = rev(var_names))
    df$Source <- factor(df$Source, levels = var_names)
    p <- ggplot2::ggplot(df, ggplot2::aes(x = Source, y = Target, fill = Weight)) +
      ggplot2::geom_tile(colour = "white") +
      ggplot2::geom_text(ggplot2::aes(label = round(Weight, 2L)),
                         size = 3.2, colour = "black") +
      ggplot2::scale_fill_gradient(low = "white", high = "#D94F3D",
                                   na.value = "grey90", name = "Weight") +
      ggplot2::labs(title = title,
                    x = "Source (cause)", y = "Target (effect)") +
      ggplot2::theme_minimal(base_size = 11L) +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 30L, hjust = 1L))
    print(p)
  } else {
    diag(A) <- 0
    image(t(A[n:1L, ]), axes = FALSE,
          col  = grDevices::hcl.colors(20L, "YlOrRd", rev = TRUE),
          main = title)
    axis(1L, at = seq(0, 1, length.out = n), labels = var_names, las = 2L)
    axis(2L, at = seq(0, 1, length.out = n), labels = rev(var_names), las = 1L)
  }
  invisible(NULL)
}

# --- Public API: deep_scm ----------------------------------------------------

#' Deep Structural Causal Model (DeepSCM) for time-series
#'
#' Trains a variational Deep SCM on multivariate time-series lagged windows.
#' Assumes a user-supplied (or correlation-derived) binary adjacency matrix.
#' Each variable's structural equation is a small MLP conditioned on its
#' graph-parents and a per-variable latent noise vector inferred by a
#' variational encoder.
#'
#' @param x_seq     3-D numeric array \code{(T x lag x d)}: lagged input
#'   windows.  Row \code{t} is the window \code{[t-lag, t-1]}.
#' @param adjacency Binary matrix \code{(d x d)}: \code{adjacency[j, i] = 1}
#'   means variable \code{j} is a parent of variable \code{i}.
#'   If \code{NULL} a correlation heuristic (threshold 0.25) is used.
#' @param lag        Integer lag order (number of time steps per window).
#'   Must equal \code{dim(x_seq)[2]}.
#' @param latent_dim Integer dimension of the per-variable noise latent space.
#' @param hidden     Integer hidden-layer width.
#' @param n_epochs   Number of training epochs.
#' @param batch_size Mini-batch size.
#' @param lr         Learning rate for Adam.
#' @param lam_kl     KL-divergence weight in the ELBO loss.
#' @param verbose    Print progress every 5 epochs.
#' @param device     Torch device string (\code{"cpu"} or \code{"cuda"}).
#'   \code{NULL} auto-selects.
#' @param ...        Ignored.
#' @return Object of class \code{deep_scm} containing \code{model},
#'   \code{adjacency}, \code{lag}, \code{var_names}, \code{train_losses},
#'   and \code{device}.
#' @seealso \code{\link{deci_model}}, \code{\link{dynotears}},
#'   \code{\link{plot_scm_dag}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # deep_scm(x_seq, adjacency, lag = 5L)
#' }
#' @export
deep_scm <- function(x_seq,
                     adjacency  = NULL,
                     lag        = dim(x_seq)[2L],
                     latent_dim = 4L,
                     hidden     = 64L,
                     n_epochs   = 25L,
                     batch_size = 64L,
                     lr         = 1e-3,
                     lam_kl     = 0.05,
                     verbose    = TRUE,
                     device     = NULL,
                     ...) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("deep_scm() requires package 'torch'.")
  if (!requireNamespace("coro", quietly = TRUE))
    stop("deep_scm() requires package 'coro'.")

  if (length(dim(x_seq)) != 3L)
    stop("x_seq must be a 3-D array (T x lag x d).")

  d   <- dim(x_seq)[3L]
  dev <- if (is.null(device)) (if (torch::cuda_is_available()) "cuda" else "cpu") else device

  if (is.null(adjacency)) {
    dat_mat   <- x_seq[, lag, ]
    corr_mat  <- abs(cor(dat_mat))
    diag(corr_mat) <- 0.0
    adjacency <- (corr_mat > 0.25) * 1.0
  }
  adjacency <- as.matrix(adjacency)

  res <- .deepnet_fit_deep_scm(
    x_seq      = x_seq,
    adjacency  = adjacency,
    lag        = as.integer(lag),
    latent_dim = as.integer(latent_dim),
    hidden     = as.integer(hidden),
    n_epochs   = as.integer(n_epochs),
    batch_size = as.integer(batch_size),
    lr         = lr,
    lam_kl     = lam_kl,
    verbose    = verbose,
    device     = dev
  )
  structure(res, class = "deep_scm")
}

#' Predict (reconstruct) from a DeepSCM
#'
#' Runs the forward pass of a fitted \code{deep_scm} model on new lagged
#' windows and returns predicted next-step values.
#'
#' @param object Fitted \code{deep_scm} object.
#' @param newdata 3-D array \code{(N x lag x d)}: new lagged windows.
#' @param ...    Ignored.
#' @return Numeric matrix \code{(N x d)} of predicted values.
#' @seealso \code{\link{deep_scm}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # predict.deep_scm(fit, newdata)
#' }
#' @export
predict.deep_scm <- function(object, newdata, ...) {
  object$model$eval()
  dev <- object$device
  torch::with_no_grad({
    x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = dev)
    out <- object$model(x_t)
    as.matrix(out$preds$detach()$to(device = "cpu"))
  })
}

#' Intervention (do-calculus) via DeepSCM
#'
#' Performs a \eqn{do(X_{\text{target}} = v)} intervention and returns
#' predicted outcomes under both low and high values.
#'
#' @param object     Fitted \code{deep_scm} object.
#' @param newdata    3-D array \code{(N x lag x d)}: input windows.
#' @param target_var Integer (1-indexed) column index of the intervention
#'   variable.
#' @param do_values  Numeric vector of length 2: \code{c(low, high)}.
#' @param ...        Ignored.
#' @return List with \code{pred_low}, \code{pred_high} (N x d matrices) and
#'   \code{delta} (mean difference on all variables under high vs low).
#' @seealso \code{\link{deep_scm}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # intervene.deep_scm(fit, newdata, target_var = 2L, do_values = c(-1, 1))
#' }
#' @export
intervene_deep_scm <- function(object, newdata, target_var, do_values = c(-1.0, 1.0), ...) {
  object$model$eval()
  dev <- object$device
  x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = dev)
  pred_lo <- as.matrix(
    object$model$intervene(x_t, target_var = as.integer(target_var),
                           do_value = do_values[1L])$detach()$to(device = "cpu")
  )
  pred_hi <- as.matrix(
    object$model$intervene(x_t, target_var = as.integer(target_var),
                           do_value = do_values[2L])$detach()$to(device = "cpu")
  )
  list(
    pred_low  = pred_lo,
    pred_high = pred_hi,
    delta     = colMeans(pred_hi - pred_lo)
  )
}

# --- Public API: deci_model --------------------------------------------------

#' DECI: Deep End-to-End Causal Inference
#'
#' Jointly learns a soft causal adjacency matrix and nonlinear structural
#' equations from multivariate time-series lagged windows.  Uses a NOTEARS
#' acyclicity penalty \eqn{h(A)=\mathrm{tr}(e^{A\circ A})-d} together with
#' an ELBO-style variational objective.
#'
#' @param x_seq     3-D numeric array \code{(T x lag x d)}.
#' @param lag        Integer lag order.  Must equal \code{dim(x_seq)[2]}.
#' @param latent_dim Integer latent noise dimension per variable.
#' @param hidden     Integer hidden-layer width.
#' @param n_epochs   Number of training epochs.
#' @param batch_size Mini-batch size.
#' @param lr         Learning rate for Adam.
#' @param lam_kl     KL weight.
#' @param lam_dag    DAG acyclicity penalty weight.
#' @param lam_sparse Sparsity (L1) penalty weight on the adjacency matrix.
#' @param threshold  Threshold used to binarise the soft adjacency matrix.
#' @param verbose    Print progress every 10 epochs.
#' @param device     Torch device string.  \code{NULL} auto-selects.
#' @param ...        Ignored.
#' @return Object of class \code{deci_model} containing \code{model},
#'   \code{A_soft}, \code{A_binary}, \code{lag}, \code{threshold},
#'   \code{var_names}, \code{train_losses}, and \code{device}.
#' @seealso \code{\link{deep_scm}}, \code{\link{dynotears}},
#'   \code{\link{evaluate_graph_recovery}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # deci_model(x_seq, lag = 5L)
#' }
#' @export
deci_model <- function(x_seq,
                       lag        = dim(x_seq)[2L],
                       latent_dim = 8L,
                       hidden     = 64L,
                       n_epochs   = 40L,
                       batch_size = 64L,
                       lr         = 1e-3,
                       lam_kl     = 0.05,
                       lam_dag    = 1.0,
                       lam_sparse = 0.01,
                       threshold  = 0.35,
                       verbose    = TRUE,
                       device     = NULL,
                       ...) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("deci_model() requires package 'torch'.")
  if (!requireNamespace("coro", quietly = TRUE))
    stop("deci_model() requires package 'coro'.")

  if (length(dim(x_seq)) != 3L)
    stop("x_seq must be a 3-D array (T x lag x d).")

  dev <- if (is.null(device)) (if (torch::cuda_is_available()) "cuda" else "cpu") else device

  res <- .deepnet_fit_deci(
    x_seq      = x_seq,
    lag        = as.integer(lag),
    latent_dim = as.integer(latent_dim),
    hidden     = as.integer(hidden),
    n_epochs   = as.integer(n_epochs),
    batch_size = as.integer(batch_size),
    lr         = lr,
    lam_kl     = lam_kl,
    lam_dag    = lam_dag,
    lam_sparse = lam_sparse,
    threshold  = threshold,
    verbose    = verbose,
    device     = dev
  )
  structure(res, class = "deci_model")
}

#' Predict from a DECI model
#'
#' @param object Fitted \code{deci_model} object.
#' @param newdata 3-D array \code{(N x lag x d)}.
#' @param ...    Ignored.
#' @return Numeric matrix \code{(N x d)}.
#' @seealso \code{\link{deci_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # predict.deci_model(fit, newdata)
#' }
#' @export
predict.deci_model <- function(object, newdata, ...) {
  object$model$eval()
  dev <- object$device
  torch::with_no_grad({
    x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = dev)
    out <- object$model$forward(x_t)
    as.matrix(out$preds$detach()$to(device = "cpu"))
  })
}

#' Estimate Average Treatment Effect (ATE) from DECI
#'
#' Uses the fitted DECI model's \code{compute_ate} method to estimate the
#' causal effect of setting variable \code{source} to low vs. high values.
#'
#' @param object     Fitted \code{deci_model} object.
#' @param newdata    3-D array \code{(N x lag x d)}.
#' @param source     Integer (1-indexed): intervention variable.
#' @param target     Integer (1-indexed): outcome variable.
#' @param do_values  Numeric vector of length 2: \code{c(low, high)}.
#' @param n_samples  Number of samples used for ATE Monte-Carlo averaging.
#' @param ...        Ignored.
#' @return Named scalar ATE estimate.
#' @seealso \code{\link{deci_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # ate_deci(fit, newdata, source = 2L, target = 5L)
#' }
#' @export
ate_deci <- function(object, newdata, source, target,
                     do_values = c(-1.0, 1.0), n_samples = 200L, ...) {
  dev <- object$device
  x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = dev)
  ate <- object$model$compute_ate(
    x_seq    = x_t,
    source   = as.integer(source),
    target   = as.integer(target),
    do_values = do_values,
    n_samples = as.integer(n_samples)
  )
  setNames(ate, paste0("ATE_do(X", source, ")_on_X", target))
}

# --- Public API: dynotears ---------------------------------------------------

#' DYNOTEARS: Lagged causal discovery for multivariate time series
#'
#' Learns a set of lag-specific weight matrices \eqn{W^{(1)},\ldots,W^{(p)}}
#' with an augmented-Lagrangian DAG constraint
#' \eqn{h(W)=\mathrm{tr}(e^{(\sum_k|W^{(k)}|)\circ(\sum_k|W^{(k)}|)})-d}.
#'
#' @param x_seq     3-D numeric array \code{(T x lag x d)}: lagged input
#'   windows.
#' @param y_mat     Numeric matrix \code{(T x d)}: next-step targets.  If
#'   \code{NULL}, \code{x_seq[, lag, ]} is used.
#' @param lag        Integer lag order.
#' @param n_epochs   Number of training epochs.
#' @param batch_size Mini-batch size.
#' @param lr         Adam learning rate.
#' @param lambda_l1  L1 sparsity regularization weight.
#' @param rho_init   Initial augmented-Lagrangian penalty coefficient.
#' @param threshold  Threshold for binarising \code{W_agg} into \code{A_binary}.
#' @param verbose    Print progress every 50 epochs.
#' @param device     Torch device string.  \code{NULL} auto-selects.
#' @param ...        Ignored.
#' @return Object of class \code{dynotears} containing \code{model},
#'   \code{A_binary}, \code{W_agg}, \code{lag}, \code{threshold},
#'   \code{var_names}, \code{train_losses}, \code{dag_vals}, and
#'   \code{device}.
#' @seealso \code{\link{deci_model}}, \code{\link{evaluate_graph_recovery}},
#'   \code{\link{plot_scm_dag}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # dynotears(x_seq, lag = 5L)
#' }
#' @export
dynotears <- function(x_seq,
                      y_mat      = NULL,
                      lag        = dim(x_seq)[2L],
                      n_epochs   = 300L,
                      batch_size = 256L,
                      lr         = 3e-3,
                      lambda_l1  = 0.02,
                      rho_init   = 1.0,
                      threshold  = 0.08,
                      verbose    = TRUE,
                      device     = NULL,
                      ...) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("dynotears() requires package 'torch'.")
  if (!requireNamespace("coro", quietly = TRUE))
    stop("dynotears() requires package 'coro'.")

  if (length(dim(x_seq)) != 3L)
    stop("x_seq must be a 3-D array (T x lag x d).")

  d   <- dim(x_seq)[3L]
  dev <- if (is.null(device)) (if (torch::cuda_is_available()) "cuda" else "cpu") else device

  if (is.null(y_mat)) y_mat <- x_seq[, as.integer(lag), , drop = FALSE]
  y_mat <- matrix(y_mat, ncol = d)

  res <- .deepnet_fit_dynotears(
    x_seq     = x_seq,
    y_mat     = y_mat,
    lag       = as.integer(lag),
    n_epochs  = as.integer(n_epochs),
    batch_size = as.integer(batch_size),
    lr        = lr,
    lambda_l1 = lambda_l1,
    rho_init  = rho_init,
    threshold = threshold,
    verbose   = verbose,
    device    = dev
  )
  structure(res, class = "dynotears")
}

#' Predict from a DYNOTEARS model
#'
#' @param object  Fitted \code{dynotears} object.
#' @param newdata 3-D array \code{(N x lag x d)}.
#' @param ...     Ignored.
#' @return Numeric matrix \code{(N x d)}.
#' @seealso \code{\link{dynotears}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # predict.dynotears(fit, newdata)
#' }
#' @export
predict.dynotears <- function(object, newdata, ...) {
  object$model$eval()
  dev <- object$device
  torch::with_no_grad({
    x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = dev)
    as.matrix(object$model(x_t)$detach()$to(device = "cpu"))
  })
}

#' CamelCase alias: deepSCM
#' @rdname deep_scm
#' @export
deepSCM <- function(x_seq, ...) deep_scm(x_seq, ...)

#' CamelCase alias: deciModel
#' @rdname deci_model
#' @export
deciModel <- function(x_seq, ...) deci_model(x_seq, ...)

#' CamelCase alias: dynoTEARS
#' @rdname dynotears
#' @export
dynoTEARS <- function(x_seq, ...) dynotears(x_seq, ...)


# =============================================================================
# Attention-Based / Transformer Causal Models
# =============================================================================
# Models (from 03_attention_transforner_causalML.ipynb):
#   • TCDFNet           — Temporal Causal Discovery Framework: stacked causal dilated
#                         convolutions + attention head for variable importance.
#   • CausalTransformer — Transformer encoder with causal masking + cross-variable
#                         multi-head attention for graph discovery.
#   • TFTNet            — Temporal Fusion Transformer: variable selection networks,
#                         LSTM encoder, interpretable multi-head temporal attention.
# Reference: Nauta et al. (2019) TCDF; Lim et al. (2021) TFT.
# =============================================================================

# --- Shared helpers ----------------------------------------------------------

.deepnet_attn_select_device <- function(device) {
  if (!is.null(device)) return(device)
  if (torch::cuda_is_available()) "cuda" else "cpu"
}

.deepnet_attn_build_dataset <- function(data_mat, lag, ahead = 1L) {
  T_len <- nrow(data_mat)
  d     <- ncol(data_mat)
  n_windows <- T_len - lag - ahead + 1L
  if (n_windows < 1L) stop("Insufficient rows for lag=", lag, " ahead=", ahead, ".")
  X <- array(0, dim = c(n_windows, lag, d))
  Y <- matrix(0, nrow = n_windows, ncol = d)
  for (t in seq_len(n_windows)) {
    X[t, , ] <- data_mat[(t):(t + lag - 1L), , drop = FALSE]
    Y[t,  ]  <- data_mat[t + lag + ahead - 2L, ]
  }
  list(X = X, Y = Y)
}

# --- Learnable Positional Encoding -------------------------------------------
# Uses nn_embedding for positional encoding (simpler and robust in R torch).

.deepnet_attn_pos_enc <- function(d_model, max_len = 512L, dropout = 0.1) {
  dm <- as.integer(d_model)
  ml <- as.integer(max_len)
  torch::nn_module(
    initialize = function() {
      self$emb  <- torch::nn_embedding(ml, dm)
      self$drop <- torch::nn_dropout(p = dropout)
    },
    forward = function(x) {
      # x: (batch, t_len, d_model)
      t_len <- x$size(2)
      dev   <- x$device
      # nn_embedding in R torch is 1-indexed
      pos   <- torch::torch_arange(start = 1L, end = as.integer(t_len),
                                    dtype = torch::torch_long(), device = dev)
      pe    <- self$emb(pos)            # (t_len, d_model)
      self$drop(x + pe$unsqueeze(1L))  # broadcast (1, t_len, d_model) + (batch, t_len, d_model)
    }
  )
}

# --- Gated Residual Network --------------------------------------------------

.deepnet_attn_grn <- function(d_in, d_hidden, d_out, dropout = 0.1) {
  d_in_i <- as.integer(d_in); d_h_i <- as.integer(d_hidden); d_o_i <- as.integer(d_out)
  torch::nn_module(
    initialize = function() {
      self$fc1  <- torch::nn_linear(d_in_i, d_h_i)
      self$fc2  <- torch::nn_linear(d_h_i, d_o_i)
      self$gate <- torch::nn_linear(d_in_i, d_o_i)
      self$skip <- if (d_in_i != d_o_i) torch::nn_linear(d_in_i, d_o_i) else torch::nn_identity()
      self$norm <- torch::nn_layer_norm(d_o_i)
      self$drop <- torch::nn_dropout(p = dropout)
    },
    forward = function(x) {
      h <- self$fc2(self$drop(torch::nnf_elu(self$fc1(x))))
      g <- torch::torch_sigmoid(self$gate(x))
      self$norm(g * h + (1 - g) * self$skip(x))
    }
  )
}

# --- Causal Dilated Conv Block -----------------------------------------------

.deepnet_attn_causal_dilated_conv_block <- function(in_ch, out_ch, kernel = 3L, dilation = 1L) {
  in_i <- as.integer(in_ch); out_i <- as.integer(out_ch)
  k_i  <- as.integer(kernel); d_i  <- as.integer(dilation)
  pad_amount <- as.integer((k_i - 1L) * d_i)
  torch::nn_module(
    initialize = function() {
      self$conv <- torch::nn_conv1d(in_i, out_i, kernel_size = k_i, dilation = d_i)
      self$norm <- torch::nn_layer_norm(out_i)
    },
    forward = function(x) {
      # x: (batch, channels, timesteps)
      xp <- torch::nnf_pad(x, c(pad_amount, 0L))
      y  <- self$conv(xp)
      # LayerNorm over feature dim: transpose to (batch, time, ch), norm, transpose back
      y  <- self$norm(y$permute(c(1L, 3L, 2L)))$permute(c(1L, 3L, 2L))
      torch::nnf_gelu(y)
    }
  )
}

# --- TCDFNet -----------------------------------------------------------------

.deepnet_attn_tcdf_net <- function(d, lag, hidden = 48L, n_layers = 4L) {
  d_i  <- as.integer(d)
  h_i  <- as.integer(hidden)
  nl_i <- as.integer(n_layers)
  ConvBlock <- .deepnet_attn_causal_dilated_conv_block
  torch::nn_module(
    initialize = function() {
      self$d       <- d_i
      self$nl      <- nl_i
      blocks <- list()
      in_ch  <- d_i
      for (i in seq_len(nl_i)) {
        blocks[[i]] <- ConvBlock(in_ch, h_i, kernel = 3L, dilation = 2L^(i - 1L))()
        in_ch <- h_i
      }
      self$tower   <- torch::nn_module_list(blocks)
      self$attn_fc <- torch::nn_linear(h_i, d_i)
      self$out_net <- torch::nn_sequential(
        torch::nn_linear(h_i, as.integer(h_i %/% 2L)),
        torch::nn_relu(),
        torch::nn_linear(as.integer(h_i %/% 2L), d_i)
      )
    },
    forward = function(x_seq) {
      # x_seq: (batch, lag, d) -> permute to (batch, d, lag) for Conv1d
      x <- x_seq$permute(c(1L, 3L, 2L))
      for (i in seq_len(self$nl)) x <- self$tower[[i]](x)
      h_last <- x[, , x$size(3)]  # last time step: (batch, hidden)
      list(
        pred         = self$out_net(h_last),
        attn_weights = torch::nnf_softmax(self$attn_fc(h_last), dim = 2L)
      )
    },
    causal_matrix = function(x_t, n_batch = 30L) {
      self$eval()
      d_val <- self$d
      bs <- min(as.integer(n_batch), x_t$size(1))
      torch::with_no_grad({
        w <- self$forward(x_t[1:bs, , ])$attn_weights$mean(1L)$detach()$to(device = "cpu")
        w_v <- as.numeric(w)
        matrix(rep(w_v, each = d_val), nrow = d_val, ncol = d_val, byrow = TRUE)
      })
    }
  )
}

# --- CausalTransformer -------------------------------------------------------

.deepnet_attn_causal_transformer <- function(d, lag,
                                              d_model  = 64L,
                                              n_heads  = 4L,
                                              n_layers = 2L,
                                              dropout  = 0.1) {
  d_i  <- as.integer(d)
  dm_i <- as.integer(d_model)
  nh_i <- as.integer(n_heads)
  nl_i <- as.integer(n_layers)
  GRN   <- .deepnet_attn_grn
  PosEnc <- .deepnet_attn_pos_enc
  torch::nn_module(
    initialize = function() {
      self$d          <- d_i
      self$dm         <- dm_i
      self$input_proj <- torch::nn_linear(d_i, dm_i)
      self$pos_enc    <- PosEnc(d_model = dm_i, max_len = as.integer(lag) + 8L, dropout = dropout)()
      enc_layer <- torch::nn_transformer_encoder_layer(
        d_model         = dm_i,
        nhead           = nh_i,
        dim_feedforward = as.integer(dm_i * 4L),
        dropout         = dropout,
        batch_first     = TRUE,
        activation      = "gelu"
      )
      self$encoder   <- torch::nn_transformer_encoder(enc_layer, num_layers = nl_i)
      self$cross_var <- torch::nn_multihead_attention(dm_i, nh_i, batch_first = TRUE)
      self$out_grns  <- torch::nn_module_list(
        lapply(seq_len(d_i), function(i) GRN(dm_i, as.integer(dm_i * 2L), dm_i, dropout)())
      )
      self$out_heads <- torch::nn_module_list(
        lapply(seq_len(d_i), function(i) torch::nn_linear(dm_i, 1L))
      )
    },
    forward = function(x_seq) {
      b     <- x_seq$size(1)
      t_len <- x_seq$size(2)
      h     <- self$pos_enc(self$input_proj(x_seq))
      # Causal mask: upper-triangular, shape (t_len, t_len)
      mask  <- torch::torch_triu(
        torch::torch_ones(t_len, t_len, device = x_seq$device), diagonal = 1L
      )$bool()
      h <- self$encoder(h, mask = mask)
      # Take last time step and expand to variable dimension
      # h[, t_len, ]: (batch, d_model)  ->  unsqueeze(2): (batch, 1, d_model)  ->  expand: (batch, d, d_model)
      h_var <- h[, t_len, ]$unsqueeze(2L)$expand(c(-1L, self$d, -1L))
      cv_out <- self$cross_var(h_var, h_var, h_var, need_weights = TRUE)
      h_var2 <- cv_out[[1]]   # (batch, d, d_model)
      attn   <- cv_out[[2]]   # (batch, d, d) - averaged over heads by default
      # Per-variable predictions
      preds <- lapply(seq_len(self$d), function(i)
        self$out_heads[[i]](self$out_grns[[i]](h_var2[, i, ]))
      )
      list(
        pred         = torch::torch_cat(preds, dim = 2L),  # (batch, d)
        attn_weights = attn$mean(1L)                       # average over batch -> (d, d)
      )
    },
    causal_matrix = function(x_t, n_batch = 30L) {
      self$eval()
      bs <- min(as.integer(n_batch), x_t$size(1))
      torch::with_no_grad({
        w <- self$forward(x_t[1:bs, , ])$attn_weights$detach()$to(device = "cpu")
        as.matrix(w)
      })
    }
  )
}

# --- Variable Selection Network (for TFT) ------------------------------------

.deepnet_attn_vsn <- function(d, lag, d_model, dropout = 0.1) {
  d_i   <- as.integer(d)
  lag_i <- as.integer(lag)
  dm_i  <- as.integer(d_model)
  GRN   <- .deepnet_attn_grn
  torch::nn_module(
    initialize = function() {
      self$d        <- d_i
      self$var_grns <- torch::nn_module_list(
        lapply(seq_len(d_i), function(j) GRN(lag_i, dm_i, dm_i, dropout)())
      )
      self$ctx_grn  <- GRN(as.integer(d_i * lag_i), dm_i, d_i, dropout)()
    },
    forward = function(x_seq) {
      # x_seq: (batch, lag, d)
      b <- x_seq$size(1)
      feats <- torch::torch_stack(
        lapply(seq_len(self$d), function(j) self$var_grns[[j]](x_seq[, , j])),
        dim = 2L
      )  # (batch, d, d_model)
      weights <- torch::nnf_softmax(
        self$ctx_grn(x_seq$reshape(c(b, -1L))),
        dim = 2L
      )  # (batch, d)
      mixed <- (feats * weights$unsqueeze(3L))$sum(dim = 2L)  # (batch, d_model)
      list(mixed = mixed, weights = weights)
    }
  )
}

# --- TFTNet ------------------------------------------------------------------

.deepnet_attn_tft_net <- function(d, lag, d_model = 64L, n_heads = 4L, dropout = 0.1) {
  d_i  <- as.integer(d)
  dm_i <- as.integer(d_model)
  nh_i <- as.integer(n_heads)
  GRN  <- .deepnet_attn_grn
  VSN  <- .deepnet_attn_vsn
  torch::nn_module(
    initialize = function() {
      self$d         <- d_i
      self$vsn       <- VSN(d_i, as.integer(lag), dm_i, dropout)()
      self$temp_proj <- torch::nn_linear(d_i, dm_i)
      self$lstm      <- torch::nn_lstm(dm_i, dm_i, batch_first = TRUE)
      self$attn      <- torch::nn_multihead_attention(dm_i, nh_i, batch_first = TRUE)
      self$grn_out   <- GRN(dm_i, as.integer(dm_i * 2L), dm_i, dropout)()
      self$head      <- torch::nn_linear(dm_i, d_i)
    },
    forward = function(x_seq) {
      vsn_out <- self$vsn(x_seq)
      vsn_ctx <- vsn_out$mixed    # (batch, d_model)
      var_w   <- vsn_out$weights  # (batch, d)
      h <- self$temp_proj(x_seq)  # (batch, lag, d_model)
      h <- self$lstm(h)[[1]]      # (batch, lag, d_model)
      t_sz <- h$size(2)
      mask  <- torch::torch_triu(
        torch::torch_ones(t_sz, t_sz, device = h$device), diagonal = 1L
      )$bool()
      attn_out  <- self$attn(h, h, h, attn_mask = mask, need_weights = TRUE)
      h_attn    <- attn_out[[1]]  # (batch, lag, d_model)
      t_weights <- attn_out[[2]]  # (batch, lag, lag) - averaged over heads
      h_last    <- self$grn_out(h_attn[, t_sz, ] + vsn_ctx)  # (batch, d_model)
      pred      <- self$head(h_last)   # (batch, d)
      vt        <- var_w$mean(1L)      # (d,) average over batch
      list(
        pred             = pred,
        attn_weights     = torch::torch_outer(vt, vt),  # (d, d) outer product
        temporal_weights = t_weights$mean(1L)            # (lag, lag)
      )
    },
    causal_matrix = function(x_t, n_batch = 30L) {
      self$eval()
      bs <- min(as.integer(n_batch), x_t$size(1))
      torch::with_no_grad({
        w <- self$forward(x_t[1:bs, , ])$attn_weights$detach()$to(device = "cpu")
        as.matrix(w)
      })
    }
  )
}

# --- Shared training loop for attention models --------------------------------

.deepnet_attn_train <- function(model, X_tr, Y_tr, X_val, Y_val,
                                model_name  = "model",
                                epochs      = 20L,
                                lr          = 3e-4,
                                patience    = 6L,
                                lam_sparse  = 1e-4,
                                batch_size  = 64L,
                                device      = "cpu",
                                verbose     = FALSE) {
  n_tr      <- dim(X_tr)[1]
  idx_tr    <- seq_len(n_tr)
  best_val  <- Inf; best_state <- NULL; no_improve <- 0L
  n_ep      <- as.integer(epochs)
  hist      <- list(train = numeric(n_ep), val = numeric(n_ep))

  opt   <- torch::optim_adamw(model$parameters, lr = lr, weight_decay = 1e-4)
  sched <- torch::lr_cosine_annealing(opt, T_max = n_ep)

  x_val_t <- torch::torch_tensor(X_val, dtype = torch::torch_float32(), device = device)
  y_val_t <- torch::torch_tensor(Y_val, dtype = torch::torch_float32(), device = device)

  for (epoch in seq_len(n_ep)) {
    model$train()
    idx_shuffled <- sample(idx_tr)
    n_batches    <- ceiling(n_tr / batch_size)
    tr_loss_sum  <- 0

    for (b in seq_len(n_batches)) {
      lo <- (b - 1L) * batch_size + 1L
      hi <- min(b * batch_size, n_tr)
      bi <- idx_shuffled[lo:hi]
      xb <- torch::torch_tensor(X_tr[bi, , , drop = FALSE],
                                 dtype = torch::torch_float32(), device = device)
      yb <- torch::torch_tensor(Y_tr[bi, , drop = FALSE],
                                 dtype = torch::torch_float32(), device = device)
      opt$zero_grad()
      out  <- model$forward(xb)
      loss <- torch::nnf_mse_loss(out$pred, yb)
      if (lam_sparse > 0 && !is.null(out$attn_weights))
        loss <- loss + lam_sparse * out$attn_weights$abs()$mean()
      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      opt$step()
      tr_loss_sum <- tr_loss_sum + loss$item()
    }
    sched$step()

    model$eval()
    val_loss <- torch::with_no_grad({
      torch::nnf_mse_loss(model$forward(x_val_t)$pred, y_val_t)$item()
    })

    hist$train[epoch] <- tr_loss_sum / max(n_batches, 1L)
    hist$val[epoch]   <- val_loss

    if (val_loss < best_val) {
      best_val   <- val_loss
      best_state <- lapply(model$state_dict(), function(t) t$clone())
      no_improve <- 0L
    } else {
      no_improve <- no_improve + 1L
      if (no_improve >= as.integer(patience)) {
        if (verbose) message(sprintf("[%s] Early stopping at epoch %d",
                                     toupper(model_name), epoch))
        break
      }
    }
    if (verbose && (epoch %% 5L == 0L || epoch == 1L))
      message(sprintf("[%s] Epoch %3d | Train %.5f | Val %.5f",
                      toupper(model_name), epoch, hist$train[epoch], val_loss))
  }
  if (!is.null(best_state)) model$load_state_dict(best_state)
  hist
}

# =============================================================================
# Public API: attn_causal_model
# =============================================================================

#' Attention-Based / Transformer Causal Models for Multivariate Time Series
#'
#' Fits one or more attention-based causal discovery models to multivariate
#' time-series data, translating the architectures from
#' \code{03_attention_transforner_causalML.ipynb}:
#' \itemize{
#'   \item \code{"tcdf"} — Temporal Causal Discovery Framework (Nauta et al., 2019):
#'     stacked causal dilated convolutions + attention head that produces a
#'     variable-importance causal matrix.
#'   \item \code{"causal_transformer"} — Transformer with causal (autoregressive)
#'     masking and inter-variable cross-attention whose weights form the causal graph.
#'   \item \code{"tft"} — Temporal Fusion Transformer (Lim et al., 2021):
#'     variable-selection networks, LSTM encoder, and interpretable multi-head
#'     temporal attention.
#' }
#'
#' @param data     Numeric matrix or data frame (rows = time, cols = variables).
#' @param lag      Integer lag window fed to each model (default 20).
#' @param models   Character vector subset of
#'   \code{c("tcdf","causal_transformer","tft")} (default all three).
#' @param hidden   Hidden dimension for TCDF conv blocks (default 48).
#' @param d_model  Internal model dimension for CausalTransformer and TFT
#'   (default 64).
#' @param n_heads  Number of attention heads for CausalTransformer and TFT
#'   (default 4).
#' @param n_layers Number of stacked layers: TCDF conv blocks / Transformer
#'   encoder layers (default 4 / 2, respectively).
#' @param dropout  Dropout probability (default 0.1).
#' @param epochs   Maximum training epochs (default 20).
#' @param lr       Adam learning rate (default 3e-4).
#' @param patience Early-stopping patience in epochs (default 6).
#' @param lam_sparse Sparsity penalty on attention weights (default 1e-4).
#' @param batch_size Mini-batch size (default 64).
#' @param val_split Validation fraction in (0, 1) (default 0.2).
#' @param device   Torch device string (\code{"cpu"} or \code{"cuda"}).
#'   \code{NULL} auto-selects.
#' @param verbose  Print per-epoch progress (default FALSE).
#' @param ...      Ignored.
#' @return Object of class \code{attn_causal_model} containing:
#'   \describe{
#'     \item{\code{models}}{Named list of fitted torch modules.}
#'     \item{\code{histories}}{Named list of train/val loss vectors.}
#'     \item{\code{val_mse}}{Named numeric vector of final validation MSE.}
#'     \item{\code{causal_matrices}}{Named list of \eqn{d \times d} causal weight
#'       matrices.}
#'     \item{\code{lag}}{Integer lag used.}
#'     \item{\code{var_names}}{Character vector of variable names.}
#'     \item{\code{device}}{Torch device string.}
#'   }
#' @references
#'   Nauta, M., Bucur, D., & Seifert, C. (2019). Causal discovery with attention-
#'   based convolutional neural networks. *Machine Learning and Knowledge
#'   Extraction*, 1(1), 312-340. \doi{10.3390/make1010019}
#'
#'   Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal fusion
#'   transformers for interpretable multi-horizon time series forecasting.
#'   *International Journal of Forecasting*, 37(4), 1748-1764.
#'   \doi{10.1016/j.ijforecast.2021.03.012}
#' @seealso \code{\link{neural_granger_ml}}, \code{\link{dynotears}},
#'   \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' set.seed(42)
#' d <- 5L; T_len <- 300L
#' A <- matrix(0, d, d)
#' for (i in seq_len(d)) { A[i, i] <- 0.45; if (i > 1) A[i, i-1] <- 0.25 }
#' x <- matrix(0, T_len, d)
#' x[1, ] <- rnorm(d)
#' for (t in 2:T_len) x[t, ] <- x[t-1, ] %*% t(A) + 0.15 * rnorm(d)
#' fit <- attn_causal_model(x, lag = 10L, models = c("tcdf","causal_transformer","tft"),
#'                          epochs = 10L, verbose = TRUE)
#' print(fit$val_mse)
#' print(fit$causal_matrices[["tcdf"]])
#' }
#' @export
attn_causal_model <- function(
    data,
    lag         = 20L,
    models      = c("tcdf", "causal_transformer", "tft"),
    hidden      = 48L,
    d_model     = 64L,
    n_heads     = 4L,
    n_layers    = 4L,
    dropout     = 0.1,
    epochs      = 20L,
    lr          = 3e-4,
    patience    = 6L,
    lam_sparse  = 1e-4,
    batch_size  = 64L,
    val_split   = 0.2,
    device      = NULL,
    verbose     = FALSE,
    ...) {

  if (!requireNamespace("torch", quietly = TRUE))
    stop("attn_causal_model() requires package 'torch'.", call. = FALSE)

  x <- as.matrix(data)
  if (!is.numeric(x)) stop("`data` must be numeric.", call. = FALSE)

  valid_models <- c("tcdf", "causal_transformer", "tft")
  req_models   <- unique(tolower(models))
  bad <- setdiff(req_models, valid_models)
  if (length(bad)) stop("Unknown models: ", paste(bad, collapse = ", "), call. = FALSE)

  dev  <- .deepnet_attn_select_device(device)
  d    <- ncol(x)
  lag  <- as.integer(lag)

  ds   <- .deepnet_attn_build_dataset(x, lag = lag)
  n    <- dim(ds$X)[1]
  split_idx <- max(1L, min(n - 1L, floor((1 - val_split) * n)))
  X_tr  <- ds$X[seq_len(split_idx), , , drop = FALSE]
  Y_tr  <- ds$Y[seq_len(split_idx), , drop = FALSE]
  X_val <- ds$X[(split_idx + 1L):n, , , drop = FALSE]
  Y_val <- ds$Y[(split_idx + 1L):n, , drop = FALSE]

  fit_models      <- list()
  histories       <- list()
  val_mse_list    <- list()
  causal_matrices <- list()

  for (m in req_models) {
    model_obj <- switch(m,
      tcdf = .deepnet_attn_tcdf_net(
        d = d, lag = lag, hidden = as.integer(hidden), n_layers = as.integer(n_layers))(),
      causal_transformer = .deepnet_attn_causal_transformer(
        d = d, lag = lag, d_model = as.integer(d_model),
        n_heads = as.integer(n_heads),
        n_layers = max(1L, as.integer(n_layers) %/% 2L),
        dropout = dropout)(),
      tft = .deepnet_attn_tft_net(
        d = d, lag = lag, d_model = as.integer(d_model),
        n_heads = as.integer(n_heads), dropout = dropout)()
    )
    model_obj <- model_obj$to(device = dev)

    hist <- .deepnet_attn_train(
      model     = model_obj,
      X_tr      = X_tr, Y_tr = Y_tr,
      X_val     = X_val, Y_val = Y_val,
      model_name = m,
      epochs    = as.integer(epochs),
      lr        = lr,
      patience  = as.integer(patience),
      lam_sparse = lam_sparse,
      batch_size = as.integer(batch_size),
      device    = dev,
      verbose   = verbose
    )

    model_obj$eval()
    x_val_t <- torch::torch_tensor(X_val, dtype = torch::torch_float32(), device = dev)
    y_val_t <- torch::torch_tensor(Y_val, dtype = torch::torch_float32(), device = dev)
    vmse <- torch::with_no_grad({
      pred_v <- model_obj$forward(x_val_t)$pred
      torch::nnf_mse_loss(pred_v, y_val_t)$item()
    })

    cmat <- tryCatch({
      model_obj$causal_matrix(x_val_t, n_batch = 30L)
    }, error = function(e) {
      matrix(NA_real_, d, d)
    })
    rownames(cmat) <- colnames(cmat) <- colnames(x)

    fit_models[[m]]      <- model_obj
    histories[[m]]       <- hist
    val_mse_list[[m]]    <- vmse
    causal_matrices[[m]] <- cmat
  }

  structure(
    list(
      models          = fit_models,
      histories       = histories,
      val_mse         = unlist(val_mse_list),
      causal_matrices = causal_matrices,
      lag             = lag,
      var_names       = colnames(x),
      device          = dev
    ),
    class = "attn_causal_model"
  )
}

#' Predict from fitted attention-based causal models
#'
#' @param object  Fitted \code{attn_causal_model} object.
#' @param model   One of \code{"tcdf"}, \code{"causal_transformer"}, \code{"tft"}.
#' @param newdata 3-D numeric array \code{(N x lag x d)}: new windowed inputs.
#' @param ...     Ignored.
#' @return Numeric matrix \code{(N x d)} of one-step-ahead predictions.
#' @seealso \code{\link{attn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # predict.attn_causal_model(fit, model = "tcdf", newdata = X_new)
#' }
#' @export
predict.attn_causal_model <- function(object, model = c("tcdf", "causal_transformer", "tft"),
                                      newdata, ...) {
  model <- match.arg(model)
  if (is.null(object$models[[model]]))
    stop("Model not fitted: ", model, call. = FALSE)
  fit <- object$models[[model]]
  fit$eval()
  dev <- if (is.null(object$device)) "cpu" else object$device
  torch::with_no_grad({
    x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = dev)
    as.matrix(fit$forward(x_t)$pred$detach()$to(device = "cpu"))
  })
}

#' Extract causal (attention) matrix from a fitted attention-based model
#'
#' Returns the \eqn{d \times d} matrix whose \code{[i,j]} entry reflects the
#' inferred causal influence of variable \eqn{j} on variable \eqn{i} as learned
#' by attention weights.
#'
#' @param object Fitted \code{attn_causal_model} object.
#' @param model  One of \code{"tcdf"}, \code{"causal_transformer"}, \code{"tft"}.
#' @return Named numeric matrix \eqn{d \times d}.
#' @seealso \code{\link{attn_causal_model}}, \code{\link{plot_scm_dag}},
#'   \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # causal_matrix_attn(fit, model = "tcdf")
#' }
#' @export
causal_matrix_attn <- function(object, model = c("tcdf", "causal_transformer", "tft")) {
  model <- match.arg(model)
  cm <- object$causal_matrices[[model]]
  if (is.null(cm)) stop("Model not fitted: ", model, call. = FALSE)
  cm
}

#' Convenience wrapper: fit only the TCDF model
#'
#' Calls \code{\link{attn_causal_model}} with \code{models = "tcdf"}.
#'
#' @inheritParams attn_causal_model
#' @return Object of class \code{attn_causal_model}.
#' @seealso \code{\link{attn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # tcdf_model(x, lag = 10L)
#' }
#' @export
tcdf_model <- function(data, ...) attn_causal_model(data, models = "tcdf", ...)

#' Convenience wrapper: fit only the CausalTransformer model
#'
#' Calls \code{\link{attn_causal_model}} with \code{models = "causal_transformer"}.
#'
#' @inheritParams attn_causal_model
#' @return Object of class \code{attn_causal_model}.
#' @seealso \code{\link{attn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # causal_transformer_model(x, lag = 10L)
#' }
#' @export
causal_transformer_model <- function(data, ...) attn_causal_model(data, models = "causal_transformer", ...)

#' Convenience wrapper: fit only the TFT model
#'
#' Calls \code{\link{attn_causal_model}} with \code{models = "tft"}.
#'
#' @inheritParams attn_causal_model
#' @return Object of class \code{attn_causal_model}.
#' @seealso \code{\link{attn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # tft_model(x, lag = 10L)
#' }
#' @export
tft_model <- function(data, ...) attn_causal_model(data, models = "tft", ...)

#' CamelCase alias: attnCausalModel
#' @rdname attn_causal_model
#' @export
attnCausalModel <- function(data, ...) attn_causal_model(data, ...)

#' CamelCase alias: TCDFModel
#' @rdname tcdf_model
#' @export
TCDFModel <- function(data, ...) tcdf_model(data, ...)

#' CamelCase alias: CausalTransformerModel
#' @rdname causal_transformer_model
#' @export
CausalTransformerModel <- function(data, ...) causal_transformer_model(data, ...)

#' CamelCase alias: TFTModel
#' @rdname tft_model
#' @export
TFTModel <- function(data, ...) tft_model(data, ...)


# =============================================================================
# RNN / LSTM-Based Causal Models
# (Causal LSTM, RETAIN, Intervention-Aware RNN)
# Translated from 04_rnn_lstm_causalML.ipynb
# =============================================================================

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

.deepnet_rnn_select_device <- function(device) {
  if (!is.null(device)) return(device)
  if (torch::cuda_is_available()) "cuda" else "cpu"
}

.deepnet_rnn_build_dataset <- function(x, lag, intervention = NULL) {
  T_n <- nrow(x)
  d   <- ncol(x)
  n   <- T_n - lag
  if (n < 1L) stop("`lag` is too large for the supplied data.", call. = FALSE)
  X   <- array(0.0, c(n, lag, d))
  Y   <- matrix(0.0, n, d)
  I_out <- if (!is.null(intervention)) matrix(0.0, n, lag) else NULL
  for (t in seq_len(n)) {
    X[t, , ]  <- x[t:(t + lag - 1L), ]
    Y[t, ]    <- x[t + lag, ]
    if (!is.null(intervention))
      I_out[t, ] <- intervention[t:(t + lag - 1L)]
  }
  list(X = X, Y = Y, I = I_out)
}

# ---------------------------------------------------------------------------
# Module: LearnableCausalMask
# ---------------------------------------------------------------------------

.deepnet_rnn_learnable_mask_module <- function(d) {
  d <- as.integer(d)
  torch::nn_module(
    "LearnableCausalMask",
    initialize = function() {
      self$d      <- d
      self$logits <- torch::nn_parameter(
        torch::torch_zeros(c(d, d))
      )
    },
    forward = function(hard = FALSE) {
      G   <- torch::torch_sigmoid(self$logits)
      eye <- torch::torch_eye(d, device = G$device)
      G   <- G * (1.0 - eye) + eye
      if (hard) G <- (G > 0.5)$float()
      G
    },
    sparsity_loss = function(lam = 0.005) {
      G   <- torch::torch_sigmoid(self$logits)
      eye <- torch::torch_eye(d, device = G$device)
      lam * (G * (1.0 - eye))$sum()
    }
  )
}

# ---------------------------------------------------------------------------
# Module: CausalLSTM
# ---------------------------------------------------------------------------

.deepnet_rnn_causal_lstm_module <- function(d, lag,
                                             hidden     = 64L,
                                             n_layers   = 2L,
                                             dropout    = 0.2,
                                             lam_sparse = 0.005) {
  d          <- as.integer(d)
  lag        <- as.integer(lag)
  hidden     <- as.integer(hidden)
  n_layers   <- as.integer(n_layers)
  lam_sparse <- lam_sparse

  torch::nn_module(
    "CausalLSTM",
    initialize = function() {
      self$d          <- d
      self$lam_sparse <- lam_sparse
      self$causal_mask <- .deepnet_rnn_learnable_mask_module(d)()

      self$lstms <- torch::nn_module_list(
        lapply(seq_len(d), function(i)
          torch::nn_lstm(
            input_size  = d,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = TRUE,
            dropout     = if (n_layers > 1L) dropout else 0.0
          )
        )
      )

      self$heads <- torch::nn_module_list(
        lapply(seq_len(d), function(i)
          torch::nn_sequential(
            torch::nn_linear(hidden, hidden %/% 2L),
            torch::nn_relu(),
            torch::nn_dropout(dropout),
            torch::nn_linear(hidden %/% 2L, 1L)
          )
        )
      )
    },
    forward = function(x_seq) {
      G     <- self$causal_mask$forward()
      preds <- vector("list", self$d)
      for (i in seq_len(self$d)) {
        g_i      <- G[i, ]$view(c(1L, 1L, d))
        x_masked <- x_seq * g_i
        lstm_out  <- self$lstms[[i]](x_masked)
        h_last   <- lstm_out[[1L]][, -1L, ]
        preds[[i]] <- self$heads[[i]](h_last)
      }
      pred     <- torch::torch_cat(preds, dim = -1L)
      sp_loss  <- self$causal_mask$sparsity_loss(lam_sparse)
      list(pred = pred, mask = G$detach()$cpu(), sparsity_loss = sp_loss)
    },
    causal_matrix = function(hard = FALSE) {
      torch::with_no_grad({
        as.matrix(self$causal_mask$forward(hard = hard)$detach()$cpu())
      })
    }
  )
}

# ---------------------------------------------------------------------------
# Module: RETAIN
# ---------------------------------------------------------------------------

.deepnet_rnn_retain_module <- function(d, lag,
                                        hidden_alpha = 32L,
                                        hidden_beta  = 32L,
                                        dropout      = 0.2) {
  d            <- as.integer(d)
  lag          <- as.integer(lag)
  hidden_alpha <- as.integer(hidden_alpha)
  hidden_beta  <- as.integer(hidden_beta)

  torch::nn_module(
    "RETAIN",
    initialize = function() {
      self$d    <- d
      self$embed <- torch::nn_sequential(
        torch::nn_linear(d, hidden_alpha),
        torch::nn_tanh()
      )
      self$alpha_rnn <- torch::nn_gru(
        input_size  = hidden_alpha,
        hidden_size = hidden_alpha,
        batch_first = TRUE
      )
      self$alpha_fc  <- torch::nn_linear(hidden_alpha, 1L)
      self$beta_rnn  <- torch::nn_gru(
        input_size  = hidden_alpha,
        hidden_size = hidden_beta,
        batch_first = TRUE
      )
      self$beta_fc   <- torch::nn_linear(hidden_beta, d)
      self$drop      <- torch::nn_dropout(dropout)
      self$output_fc <- torch::nn_linear(d, d)
    },
    forward = function(x_seq) {
      v     <- self$embed(x_seq)
      v_rev <- torch::torch_flip(v, dims = 2L)

      e_alpha <- self$alpha_rnn(v_rev)[[1L]]
      e_alpha <- self$alpha_fc(e_alpha)$squeeze(-1L)
      alpha   <- torch::nnf_softmax(e_alpha, dim = 2L)

      e_beta  <- self$beta_rnn(v_rev)[[1L]]
      beta    <- torch::torch_tanh(self$beta_fc(e_beta))
      beta_fwd <- torch::torch_flip(beta, dims = 2L)

      alpha_exp <- alpha$unsqueeze(-1L)
      context   <- (alpha_exp * beta_fwd * x_seq)$sum(dim = 2L)
      context   <- self$drop(context)
      pred      <- self$output_fc(context)

      W   <- self$output_fc$weight$abs()$mean(dim = 1L)
      attr_mat <- alpha_exp * beta_fwd$abs() * W

      list(
        pred        = pred,
        alpha       = alpha$detach(),
        beta        = beta_fwd$detach(),
        attribution = attr_mat$detach()
      )
    },
    causal_matrix = function(x_batch, n_batch = 30L) {
      self$eval()
      torch::with_no_grad({
        W  <- as.matrix(self$output_fc$weight$detach()$cpu())
        C  <- matrix(0.0, d, d)
        nb <- 0L
        n_use <- min(nrow(x_batch), n_batch * 64L)
        step  <- 64L
        start <- 1L
        while (start <= n_use && nb < n_batch) {
          end <- min(start + step - 1L, n_use)
          xb  <- torch::torch_tensor(
            x_batch[start:end, , , drop = FALSE],
            dtype = torch::torch_float32()
          )
          out      <- self$forward(xb)
          mb       <- as.matrix(out$beta$abs()$mean(dim = 2L)$cpu())
          mb_mean  <- colMeans(mb)
          C  <- C + abs(W) * matrix(mb_mean, d, d, byrow = TRUE)
          nb <- nb + 1L
          start <- end + 1L
        }
        C / max(nb, 1L)
      })
    }
  )
}

# ---------------------------------------------------------------------------
# Module: RegimeDetector
# ---------------------------------------------------------------------------

.deepnet_rnn_regime_detector_module <- function(d, n_regimes = 3L, hidden = 32L) {
  d         <- as.integer(d)
  n_regimes <- as.integer(n_regimes)
  hidden    <- as.integer(hidden)

  torch::nn_module(
    "RegimeDetector",
    initialize = function() {
      self$rnn <- torch::nn_gru(d, hidden, batch_first = TRUE)
      self$fc  <- torch::nn_sequential(
        torch::nn_linear(hidden, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, n_regimes)
      )
    },
    forward = function(x_seq) {
      h      <- self$rnn(x_seq)[[1L]]
      logits <- self$fc(h[, -1L, ])
      torch::nnf_softmax(logits, dim = 2L)
    }
  )
}

# ---------------------------------------------------------------------------
# Module: InterventionAwareRNN
# ---------------------------------------------------------------------------

.deepnet_rnn_intervention_rnn_module <- function(d, lag,
                                                   hidden     = 64L,
                                                   n_layers   = 2L,
                                                   n_regimes  = 3L,
                                                   regime_dim = 8L,
                                                   dropout    = 0.2) {
  d          <- as.integer(d)
  lag        <- as.integer(lag)
  hidden     <- as.integer(hidden)
  n_layers   <- as.integer(n_layers)
  n_regimes  <- as.integer(n_regimes)
  regime_dim <- as.integer(regime_dim)

  torch::nn_module(
    "InterventionAwareRNN",
    initialize = function() {
      self$d         <- d
      self$n_regimes <- n_regimes
      self$regime_detector <- .deepnet_rnn_regime_detector_module(d, n_regimes, 32L)()
      self$regime_embed    <- torch::nn_linear(n_regimes, regime_dim)
      self$interv_proj     <- torch::nn_linear(1L, regime_dim)
      lstm_input <- d + regime_dim + regime_dim
      self$lstm  <- torch::nn_lstm(
        input_size  = lstm_input,
        hidden_size = hidden,
        num_layers  = n_layers,
        batch_first = TRUE,
        dropout     = if (n_layers > 1L) dropout else 0.0
      )
      self$head <- torch::nn_sequential(
        torch::nn_linear(hidden, hidden),
        torch::nn_gelu(),
        torch::nn_dropout(dropout),
        torch::nn_linear(hidden, d)
      )
      self$regime_causal_weights <- torch::nn_parameter(
        torch::torch_ones(c(n_regimes, d, d)) / d
      )
    },
    forward = function(x_seq, interv = NULL) {
      if (is.null(interv))
        interv <- torch::torch_zeros(
          c(x_seq$size(1L), x_seq$size(2L)), device = x_seq$device
        )
      regime_probs <- self$regime_detector(x_seq)
      regime_vec   <- self$regime_embed(regime_probs)
      regime_seq   <- regime_vec$unsqueeze(2L)$`repeat`(c(1L, x_seq$size(2L), 1L))
      interv_vec   <- self$interv_proj(interv$unsqueeze(-1L))
      x_aug <- torch::torch_cat(list(x_seq, regime_seq, interv_vec), dim = -1L)
      lstm_out <- self$lstm(x_aug)
      h_last   <- lstm_out[[1L]][, -1L, ]
      pred     <- self$head(h_last)
      list(
        pred         = pred,
        regime_probs = regime_probs$detach(),
        regime_ids   = regime_probs$argmax(dim = 2L)$detach()
      )
    },
    causal_matrix = function() {
      torch::with_no_grad({
        probs <- torch::nnf_softmax(self$regime_causal_weights, dim = -1L)
        as.matrix(probs$mean(dim = 1L)$detach()$cpu())
      })
    }
  )
}

# ---------------------------------------------------------------------------
# Training helper (handles both base and intervention-aware models)
# ---------------------------------------------------------------------------

.deepnet_rnn_train <- function(model, X_tr, Y_tr, X_val, Y_val,
                                I_tr = NULL, I_val = NULL,
                                model_name  = "model",
                                epochs      = 60L,
                                lr          = 3e-4,
                                patience    = 15L,
                                batch_size  = 64L,
                                device      = "cpu",
                                verbose     = FALSE) {
  use_interv <- !is.null(I_tr)

  x_tr_t  <- torch::torch_tensor(X_tr,  dtype = torch::torch_float32(), device = device)
  y_tr_t  <- torch::torch_tensor(Y_tr,  dtype = torch::torch_float32(), device = device)
  x_val_t <- torch::torch_tensor(X_val, dtype = torch::torch_float32(), device = device)
  y_val_t <- torch::torch_tensor(Y_val, dtype = torch::torch_float32(), device = device)
  if (use_interv) {
    i_tr_t  <- torch::torch_tensor(I_tr,  dtype = torch::torch_float32(), device = device)
    i_val_t <- torch::torch_tensor(I_val, dtype = torch::torch_float32(), device = device)
  }

  N        <- x_tr_t$size(1L)
  n_batch  <- ceiling(N / batch_size)
  opt      <- torch::optim_adamw(model$parameters, lr = lr, weight_decay = 1e-4)
  sched    <- torch::lr_cosine_annealing(opt, T_max = epochs)

  best_val   <- Inf
  best_state <- NULL
  no_improve <- 0L
  hist       <- list(train = numeric(epochs), val = numeric(epochs))

  for (epoch in seq_len(epochs)) {
    model$train()
    idx_perm <- torch::torch_randperm(N, device = device) + 1L
    tr_loss_sum <- 0.0
    for (b in seq_len(n_batch)) {
      start <- (b - 1L) * batch_size + 1L
      end   <- min(b * batch_size, N)
      idx_b <- idx_perm[start:end]
      xb    <- x_tr_t[idx_b, , ]
      yb    <- y_tr_t[idx_b, ]
      opt$zero_grad()
      if (use_interv) {
        ib  <- i_tr_t[idx_b, ]
        out <- model$forward(xb, interv = ib)
      } else {
        out <- model$forward(xb)
      }
      loss <- torch::nnf_mse_loss(out$pred, yb)
      if (!is.null(out$sparsity_loss)) loss <- loss + out$sparsity_loss
      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      opt$step()
      tr_loss_sum <- tr_loss_sum + loss$item()
    }
    sched$step()

    model$eval()
    val_loss <- torch::with_no_grad({
      if (use_interv) {
        torch::nnf_mse_loss(model$forward(x_val_t, interv = i_val_t)$pred, y_val_t)$item()
      } else {
        torch::nnf_mse_loss(model$forward(x_val_t)$pred, y_val_t)$item()
      }
    })

    hist$train[epoch] <- tr_loss_sum / max(n_batch, 1L)
    hist$val[epoch]   <- val_loss

    if (val_loss < best_val) {
      best_val   <- val_loss
      best_state <- lapply(model$state_dict(), function(t) t$clone())
      no_improve <- 0L
    } else {
      no_improve <- no_improve + 1L
      if (no_improve >= as.integer(patience)) {
        if (verbose) message(sprintf("[%s] Early stopping at epoch %d",
                                     toupper(model_name), epoch))
        break
      }
    }
    if (verbose && (epoch %% 10L == 0L || epoch == 1L))
      message(sprintf("[%s] Epoch %3d | Train %.5f | Val %.5f",
                      toupper(model_name), epoch, hist$train[epoch], val_loss))
  }
  if (!is.null(best_state)) model$load_state_dict(best_state)
  hist
}

# =============================================================================
# Public API: rnn_causal_model
# =============================================================================

#' RNN/LSTM-Based Causal Models for Multivariate Time Series
#'
#' Fits one or more recurrent-neural-network causal models to multivariate
#' time-series data, translated from \code{04_rnn_lstm_causalML.ipynb}:
#' \itemize{
#'   \item \code{"causal_lstm"} — CausalLSTM: per-variable LSTM with a
#'     learnable sparse causal-adjacency mask
#'     \eqn{G \in [0,1]^{d \times d}}; the mask is regularised with an
#'     L1 sparsity penalty so that only genuine predictors are retained.
#'   \item \code{"retain"} — RETAIN (Choi et al., 2016): reverse-time GRU
#'     with two-channel attention — temporal (\eqn{\alpha}) and variable
#'     (\eqn{\beta}) — producing an interpretable attribution matrix
#'     per time step and variable.
#'   \item \code{"intervention_rnn"} — Intervention-Aware RNN: LSTM
#'     augmented with a soft regime-detector (GRU) and an explicit
#'     intervention-indicator channel; supports regime-conditioned causal
#'     effect matrices.
#' }
#'
#' @param data        Numeric matrix or data frame (rows = time, cols =
#'   variables).
#' @param lag         Integer lag window (default 20).
#' @param models      Character vector subset of
#'   \code{c("causal_lstm","retain","intervention_rnn")} (default all
#'   three).
#' @param intervention Optional numeric vector of length \code{nrow(data)};
#'   binary or continuous intervention indicator used by
#'   \code{"intervention_rnn"} (and ignored by the other two models).
#'   When \code{NULL} a zero vector is used for \code{"intervention_rnn"}.
#' @param hidden      LSTM/GRU hidden dimension (default 64).
#' @param n_layers    Number of stacked LSTM layers (default 2).
#' @param n_regimes   Number of latent regimes for
#'   \code{"intervention_rnn"} (default 3).
#' @param regime_dim  Embedding dimension for the regime vector (default 8).
#' @param hidden_alpha Hidden size for RETAIN's temporal-attention GRU
#'   (default 32).
#' @param hidden_beta  Hidden size for RETAIN's variable-attention GRU
#'   (default 32).
#' @param dropout     Dropout probability (default 0.2).
#' @param lam_sparse  Sparsity penalty on the CausalLSTM adjacency mask
#'   (default 0.005).
#' @param epochs      Maximum training epochs (default 60).
#' @param lr          AdamW learning rate (default 3e-4).
#' @param patience    Early-stopping patience in epochs (default 15).
#' @param batch_size  Mini-batch size (default 64).
#' @param val_split   Validation fraction in (0, 1) (default 0.2).
#' @param device      Torch device string (\code{"cpu"} or \code{"cuda"}).
#'   \code{NULL} auto-selects.
#' @param verbose     Print per-epoch progress (default \code{FALSE}).
#' @param ...         Ignored.
#' @return Object of class \code{rnn_causal_model} containing:
#'   \describe{
#'     \item{\code{models}}{Named list of fitted torch modules.}
#'     \item{\code{histories}}{Named list of train/val loss vectors.}
#'     \item{\code{val_mse}}{Named numeric vector of final validation MSE.}
#'     \item{\code{causal_matrices}}{Named list of \eqn{d \times d} causal
#'       matrices.}
#'     \item{\code{X_val}}{Validation input array (for RETAIN attribution).}
#'     \item{\code{lag}}{Integer lag used.}
#'     \item{\code{var_names}}{Character vector of variable names.}
#'     \item{\code{device}}{Torch device string.}
#'   }
#' @references
#'   Choi, E., Bahadori, M. T., Sun, J., Kulas, J., Schuetz, A., &
#'   Stewart, W. (2016). RETAIN: An interpretable predictive model for
#'   healthcare using reverse time attention mechanism. \emph{Advances in
#'   Neural Information Processing Systems}, 29.
#'
#'   Tank, A., Covert, I., Foti, N., Shojaie, A., & Fox, E. (2021).
#'   Neural Granger causality. \emph{IEEE Transactions on Pattern Analysis
#'   and Machine Intelligence}, 44(8), 4267-4279.
#'   \doi{10.1109/TPAMI.2021.3065601}
#' @seealso \code{\link{attn_causal_model}}, \code{\link{neural_granger_ml}},
#'   \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' set.seed(42)
#' d <- 5L; T_len <- 300L
#' A <- matrix(0, d, d)
#' for (i in seq_len(d)) { A[i, i] <- 0.45; if (i > 1) A[i, i-1] <- 0.25 }
#' x <- matrix(0, T_len, d)
#' x[1, ] <- rnorm(d)
#' for (t in 2:T_len) x[t, ] <- x[t-1, ] %*% t(A) + 0.15 * rnorm(d)
#' interv <- as.numeric(runif(T_len) > 0.75)
#' fit <- rnn_causal_model(x, lag = 10L, epochs = 20L, verbose = TRUE,
#'                         intervention = interv)
#' print(fit$val_mse)
#' print(fit$causal_matrices[["causal_lstm"]])
#' }
#' @export
rnn_causal_model <- function(
    data,
    lag          = 20L,
    models       = c("causal_lstm", "retain", "intervention_rnn"),
    intervention = NULL,
    hidden       = 64L,
    n_layers     = 2L,
    n_regimes    = 3L,
    regime_dim   = 8L,
    hidden_alpha = 32L,
    hidden_beta  = 32L,
    dropout      = 0.2,
    lam_sparse   = 0.005,
    epochs       = 60L,
    lr           = 3e-4,
    patience     = 15L,
    batch_size   = 64L,
    val_split    = 0.2,
    device       = NULL,
    verbose      = FALSE,
    ...) {

  if (!requireNamespace("torch", quietly = TRUE))
    stop("rnn_causal_model() requires package 'torch'.", call. = FALSE)

  x <- as.matrix(data)
  if (!is.numeric(x)) stop("`data` must be numeric.", call. = FALSE)

  valid_models <- c("causal_lstm", "retain", "intervention_rnn")
  req_models   <- unique(tolower(models))
  bad <- setdiff(req_models, valid_models)
  if (length(bad))
    stop("Unknown models: ", paste(bad, collapse = ", "), call. = FALSE)

  dev  <- .deepnet_rnn_select_device(device)
  d    <- ncol(x)
  lag  <- as.integer(lag)

  interv_vec <- if (!is.null(intervention)) {
    stopifnot(length(intervention) == nrow(x))
    as.numeric(intervention)
  } else {
    rep(0.0, nrow(x))
  }

  ds    <- .deepnet_rnn_build_dataset(x, lag = lag, intervention = interv_vec)
  n     <- dim(ds$X)[1L]
  sp    <- max(1L, min(n - 1L, floor((1 - val_split) * n)))
  X_tr  <- ds$X[seq_len(sp), , , drop = FALSE]
  Y_tr  <- ds$Y[seq_len(sp), , drop = FALSE]
  X_val <- ds$X[(sp + 1L):n, , , drop = FALSE]
  Y_val <- ds$Y[(sp + 1L):n, , drop = FALSE]
  I_tr  <- ds$I[seq_len(sp), , drop = FALSE]
  I_val <- ds$I[(sp + 1L):n, , drop = FALSE]

  fit_models      <- list()
  histories       <- list()
  val_mse_list    <- list()
  causal_matrices <- list()

  for (m in req_models) {
    model_obj <- switch(m,
      causal_lstm = .deepnet_rnn_causal_lstm_module(
        d          = d,
        lag        = lag,
        hidden     = as.integer(hidden),
        n_layers   = as.integer(n_layers),
        dropout    = dropout,
        lam_sparse = lam_sparse
      )(),
      retain = .deepnet_rnn_retain_module(
        d            = d,
        lag          = lag,
        hidden_alpha = as.integer(hidden_alpha),
        hidden_beta  = as.integer(hidden_beta),
        dropout      = dropout
      )(),
      intervention_rnn = .deepnet_rnn_intervention_rnn_module(
        d          = d,
        lag        = lag,
        hidden     = as.integer(hidden),
        n_layers   = as.integer(n_layers),
        n_regimes  = as.integer(n_regimes),
        regime_dim = as.integer(regime_dim),
        dropout    = dropout
      )()
    )
    model_obj <- model_obj$to(device = dev)

    use_i <- (m == "intervention_rnn")
    hist  <- .deepnet_rnn_train(
      model      = model_obj,
      X_tr       = X_tr,   Y_tr  = Y_tr,
      X_val      = X_val,  Y_val = Y_val,
      I_tr       = if (use_i) I_tr  else NULL,
      I_val      = if (use_i) I_val else NULL,
      model_name = m,
      epochs     = as.integer(epochs),
      lr         = lr,
      patience   = as.integer(patience),
      batch_size = as.integer(batch_size),
      device     = dev,
      verbose    = verbose
    )

    model_obj$eval()
    x_val_t <- torch::torch_tensor(X_val, dtype = torch::torch_float32(), device = dev)
    y_val_t <- torch::torch_tensor(Y_val, dtype = torch::torch_float32(), device = dev)
    vmse <- torch::with_no_grad({
      if (use_i) {
        i_val_t <- torch::torch_tensor(I_val, dtype = torch::torch_float32(), device = dev)
        torch::nnf_mse_loss(
          model_obj$forward(x_val_t, interv = i_val_t)$pred, y_val_t
        )$item()
      } else {
        torch::nnf_mse_loss(
          model_obj$forward(x_val_t)$pred, y_val_t
        )$item()
      }
    })

    cmat <- tryCatch({
      switch(m,
        causal_lstm      = model_obj$causal_matrix(hard = FALSE),
        retain           = model_obj$causal_matrix(X_val, n_batch = 30L),
        intervention_rnn = model_obj$causal_matrix()
      )
    }, error = function(e) matrix(NA_real_, d, d))
    rownames(cmat) <- colnames(cmat) <- colnames(x)

    fit_models[[m]]      <- model_obj
    histories[[m]]       <- hist
    val_mse_list[[m]]    <- vmse
    causal_matrices[[m]] <- cmat
  }

  structure(
    list(
      models          = fit_models,
      histories       = histories,
      val_mse         = unlist(val_mse_list),
      causal_matrices = causal_matrices,
      X_val           = X_val,
      lag             = lag,
      var_names       = colnames(x),
      device          = dev
    ),
    class = "rnn_causal_model"
  )
}

#' Predict from fitted RNN/LSTM causal models
#'
#' @param object   Fitted \code{rnn_causal_model} object.
#' @param model    One of \code{"causal_lstm"}, \code{"retain"},
#'   \code{"intervention_rnn"}.
#' @param newdata  3-D numeric array \code{(N x lag x d)}: windowed inputs.
#' @param intervention Optional numeric matrix \code{(N x lag)} of
#'   intervention indicators; required / used only for
#'   \code{"intervention_rnn"}.
#' @param ...      Ignored.
#' @return Numeric matrix \code{(N x d)} of one-step-ahead predictions.
#' @seealso \code{\link{rnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # predict(fit, model = "causal_lstm", newdata = X_new)
#' }
#' @export
predict.rnn_causal_model <- function(
    object,
    model = c("causal_lstm", "retain", "intervention_rnn"),
    newdata,
    intervention = NULL,
    ...) {
  model <- match.arg(model)
  if (is.null(object$models[[model]]))
    stop("Model not fitted: ", model, call. = FALSE)
  fit <- object$models[[model]]
  fit$eval()
  dev <- if (is.null(object$device)) "cpu" else object$device
  torch::with_no_grad({
    x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = dev)
    if (model == "intervention_rnn" && !is.null(intervention)) {
      i_t <- torch::torch_tensor(intervention, dtype = torch::torch_float32(), device = dev)
      as.matrix(fit$forward(x_t, interv = i_t)$pred$detach()$cpu())
    } else {
      as.matrix(fit$forward(x_t)$pred$detach()$cpu())
    }
  })
}

#' Extract causal matrix from a fitted RNN/LSTM causal model
#'
#' Returns the \eqn{d \times d} matrix whose \code{[i,j]} entry reflects
#' the inferred causal influence of variable \eqn{j} on variable \eqn{i}.
#'
#' @param object Fitted \code{rnn_causal_model} object.
#' @param model  One of \code{"causal_lstm"}, \code{"retain"},
#'   \code{"intervention_rnn"}.
#' @return Named numeric matrix \eqn{d \times d}.
#' @seealso \code{\link{rnn_causal_model}}, \code{\link{plot_scm_dag}},
#'   \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # causal_matrix_rnn(fit, model = "causal_lstm")
#' }
#' @export
causal_matrix_rnn <- function(
    object,
    model = c("causal_lstm", "retain", "intervention_rnn")) {
  model <- match.arg(model)
  cm <- object$causal_matrices[[model]]
  if (is.null(cm)) stop("Model not fitted: ", model, call. = FALSE)
  cm
}

#' Convenience wrapper: fit only the CausalLSTM model
#'
#' Calls \code{\link{rnn_causal_model}} with \code{models = "causal_lstm"}.
#'
#' @inheritParams rnn_causal_model
#' @return Object of class \code{rnn_causal_model}.
#' @seealso \code{\link{rnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # causal_lstm_model(x, lag = 10L)
#' }
#' @export
causal_lstm_model <- function(data, ...) rnn_causal_model(data, models = "causal_lstm", ...)

#' Convenience wrapper: fit only the RETAIN model
#'
#' Calls \code{\link{rnn_causal_model}} with \code{models = "retain"}.
#'
#' @inheritParams rnn_causal_model
#' @return Object of class \code{rnn_causal_model}.
#' @seealso \code{\link{rnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # retain_model(x, lag = 10L)
#' }
#' @export
retain_model <- function(data, ...) rnn_causal_model(data, models = "retain", ...)

#' Convenience wrapper: fit only the Intervention-Aware RNN model
#'
#' Calls \code{\link{rnn_causal_model}} with
#' \code{models = "intervention_rnn"}.
#'
#' @inheritParams rnn_causal_model
#' @return Object of class \code{rnn_causal_model}.
#' @seealso \code{\link{rnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # intervention_rnn_model(x, lag = 10L, intervention = interv_vec)
#' }
#' @export
intervention_rnn_model <- function(data, ...) rnn_causal_model(data, models = "intervention_rnn", ...)

#' CamelCase alias: rnnCausalModel
#' @rdname rnn_causal_model
#' @export
rnnCausalModel <- function(data, ...) rnn_causal_model(data, ...)

#' CamelCase alias: CausalLSTMModel
#' @rdname causal_lstm_model
#' @export
CausalLSTMModel <- function(data, ...) causal_lstm_model(data, ...)

#' CamelCase alias: RETAINModel
#' @rdname retain_model
#' @export
RETAINModel <- function(data, ...) retain_model(data, ...)

#' CamelCase alias: InterventionRNNModel
#' @rdname intervention_rnn_model
#' @export
InterventionRNNModel <- function(data, ...) intervention_rnn_model(data, ...)

# =============================================================================
# GNN Causal Models: GVAR, CausalGNN, CUTS+like  (v0.3.0)
# =============================================================================
#
# Three graph-neural-network architectures for causal discovery in multivariate
# time series, translated from 05_graph_nn_causal_models_GNN.ipynb:
#
#  GVAR        — Graph Vector Autoregression: lag-specific learnable soft
#                adjacency matrices, per-lag GNN message passing, sparsity +
#                NOTEARS acyclicity penalty.
#  CausalGNN   — Joint graph learning (bilinear edge scoring) + GRU temporal
#                encoding + stacked EdgeConv message passing.
#  CUTS+like   — Variational Bernoulli graph + causal-aware imputation network
#                for datasets with synthetic or real missingness.
#
# Public API:  gnn_causal_model()
# Accessor:    causal_matrix_gnn()
# Prediction:  predict.gnn_causal_model()
# Wrappers:    gvar_model(), causal_gnn_model(), cuts_model()
# =============================================================================

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

.deepnet_gnn_select_device <- function(device) {
  if (!is.null(device)) return(device)
  if (torch::cuda_is_available()) "cuda" else "cpu"
}

.deepnet_gnn_build_dataset <- function(x, lag, miss_rate = 0.0, seed = 0L) {
  T_n <- nrow(x)
  d   <- ncol(x)
  n   <- T_n - lag
  if (n < 1L) stop("`lag` is too large for the supplied data.", call. = FALSE)
  X <- array(0.0, c(n, lag, d))
  Y <- matrix(0.0, n, d)
  M <- array(0.0, c(n, lag, d))
  for (t in seq_len(n)) {
    X[t, , ] <- x[t:(t + lag - 1L), ]
    Y[t, ]   <- x[t + lag, ]
  }
  if (miss_rate > 0.0) {
    set.seed(seed)
    mask    <- runif(prod(dim(X))) < miss_rate
    M[]     <- as.numeric(mask)
    X[mask] <- 0.0
  }
  list(X = X, Y = Y, M = M)
}

# NOTEARS acyclicity penalty: h(A) = tr(expm(A*A)) - d (8-term Taylor series)
.deepnet_gnn_acyclicity <- function(A) {
  d  <- A$shape[1L]
  M  <- A * A
  I  <- torch::torch_eye(d, device = A$device, dtype = A$dtype)
  ex <- I$clone()
  Mk <- I$clone()
  for (k in seq_len(8L)) {
    Mk <- torch::torch_matmul(Mk, M) / k
    ex <- ex + Mk
  }
  torch::torch_trace(ex) - d
}

# ---------------------------------------------------------------------------
# Module: GVARGraphLearner
# ---------------------------------------------------------------------------

.deepnet_gnn_gvar_graph_learner_module <- function(d, lag) {
  d   <- as.integer(d)
  lag <- as.integer(lag)
  torch::nn_module(
    "GVARGraphLearner",
    initialize = function() {
      self$d      <- d
      self$lag    <- lag
      self$logits <- torch::nn_parameter(
        torch::torch_randn(c(lag, d, d)) * 0.1
      )
    },
    forward = function() {
      A   <- torch::torch_sigmoid(self$logits)
      eye <- torch::torch_eye(self$d, device = A$device)$unsqueeze(1L)
      A * (1.0 - eye)
    },
    sparsity_loss = function(lam = 0.01) {
      lam * self$forward()$sum()
    },
    aggregate_causal = function() {
      torch::with_no_grad({
        as.matrix(self$forward()$sum(dim = 1L)$detach()$cpu())
      })
    }
  )
}

# ---------------------------------------------------------------------------
# Module: GVARMessagePass
# ---------------------------------------------------------------------------

.deepnet_gnn_gvar_message_pass_module <- function(d, lag, in_feat, out_feat) {
  d        <- as.integer(d)
  lag      <- as.integer(lag)
  in_feat  <- as.integer(in_feat)
  out_feat <- as.integer(out_feat)
  torch::nn_module(
    "GVARMessagePass",
    initialize = function() {
      self$lag <- lag
      self$lag_transforms <- torch::nn_module_list(
        lapply(seq_len(lag), function(k)
          torch::nn_sequential(
            torch::nn_linear(in_feat, out_feat),
            torch::nn_relu()
          )
        )
      )
      self$agg <- torch::nn_linear(out_feat * lag, out_feat)
    },
    forward = function(x_seq, A) {
      # x_seq: (batch, lag, d, in_feat);  A: (lag, d, d)
      msgs <- vector("list", self$lag)
      for (k in seq_len(self$lag)) {
        h_k      <- self$lag_transforms[[k]](x_seq[, k, , ])   # (batch, d, out_feat)
        A_k      <- A[k, , ]                                    # (d, d)
        msgs[[k]] <- torch::torch_matmul(A_k, h_k)             # (batch, d, out_feat)
      }
      out <- torch::torch_cat(msgs, dim = -1L)                  # (batch, d, out_feat*lag)
      self$agg(out)                                             # (batch, d, out_feat)
    }
  )
}

# ---------------------------------------------------------------------------
# Module: GVAR
# ---------------------------------------------------------------------------

.deepnet_gnn_gvar_module <- function(d, lag, hidden = 32L,
                                      lam_sparse = 0.005, lam_dag = 0.1) {
  d          <- as.integer(d)
  lag        <- as.integer(lag)
  hidden     <- as.integer(hidden)
  torch::nn_module(
    "GVAR",
    initialize = function() {
      self$d          <- d
      self$lag        <- lag
      self$lam_sparse <- lam_sparse
      self$lam_dag    <- lam_dag
      self$graph       <- .deepnet_gnn_gvar_graph_learner_module(d, lag)()
      self$input_proj  <- torch::nn_linear(1L, hidden)
      self$gnn1        <- .deepnet_gnn_gvar_message_pass_module(d, lag, hidden, hidden)()
      self$gnn2        <- .deepnet_gnn_gvar_message_pass_module(d, lag, hidden, hidden)()
      self$output_head <- torch::nn_sequential(
        torch::nn_linear(hidden, hidden),
        torch::nn_gelu(),
        torch::nn_linear(hidden, 1L)
      )
    },
    forward = function(x_seq) {
      # x_seq: (batch, lag, d)
      A      <- self$graph$forward()                            # (lag, d, d)
      x_feat <- self$input_proj(x_seq$unsqueeze(-1L))           # (batch, lag, d, hidden)
      h1     <- self$gnn1$forward(x_feat, A)                   # (batch, d, hidden)
      x_feat2 <- h1$unsqueeze(2L)$expand(c(-1L, self$lag, -1L, -1L))  # (batch, lag, d, hidden)
      h2     <- self$gnn2$forward(x_feat2, A)                  # (batch, d, hidden)
      pred   <- self$output_head(h2)$squeeze(-1L)              # (batch, d)
      sparse_loss <- self$graph$sparsity_loss(self$lam_sparse)
      dag_loss    <- self$lam_dag * .deepnet_gnn_acyclicity(A$mean(dim = 1L))
      list(pred = pred, A = A$detach(), sparse_loss = sparse_loss, dag_loss = dag_loss)
    },
    causal_matrix = function() {
      torch::with_no_grad({ self$graph$aggregate_causal() })
    }
  )
}

# ---------------------------------------------------------------------------
# Module: EdgeConvLayer  (shared by CausalGNN and CUTS+)
# ---------------------------------------------------------------------------

.deepnet_gnn_edge_conv_layer <- function(node_dim, out_dim) {
  node_dim <- as.integer(node_dim)
  out_dim  <- as.integer(out_dim)
  torch::nn_module(
    "EdgeConvLayer",
    initialize = function() {
      self$msg_net <- torch::nn_sequential(
        torch::nn_linear(node_dim * 2L + 1L, out_dim * 2L),
        torch::nn_relu(),
        torch::nn_linear(out_dim * 2L, out_dim)
      )
      self$update <- torch::nn_gru(
        input_size  = out_dim,
        hidden_size = node_dim,
        num_layers  = 1L,
        batch_first = TRUE
      )
      self$norm   <- torch::nn_layer_norm(node_dim)
    },
    forward = function(h, A) {
      # h: (batch, d, node_dim);  A: (d, d) or (batch, d, d)
      batch <- h$size(1L)
      d_n   <- h$size(2L)
      if (A$dim() == 2L)
        A <- A$unsqueeze(1L)$expand(c(batch, -1L, -1L))
      h_i    <- h$unsqueeze(3L)$expand(c(-1L, -1L, d_n, -1L))  # (batch, d, d, node_dim)
      h_j    <- h$unsqueeze(2L)$expand(c(-1L, d_n, -1L, -1L))  # (batch, d, d, node_dim)
      e_ij   <- A$unsqueeze(-1L)                                 # (batch, d, d, 1)
      msg_in <- torch::torch_cat(list(h_i, h_j, e_ij), dim = -1L)
      msgs   <- self$msg_net(msg_in)                             # (batch, d, d, out_dim)
      agg    <- (A$unsqueeze(-1L) * msgs)$sum(dim = 3L)         # (batch, d, out_dim)
      h_flat    <- h$reshape(c(batch * d_n, -1L))
      agg_flat  <- agg$reshape(c(batch * d_n, -1L))
      gru_in    <- agg_flat$unsqueeze(2L)             # (batch*d, 1, out_dim)
      h0        <- h_flat$unsqueeze(1L)               # (1, batch*d, node_dim)
      gru_out   <- self$update(gru_in, h0)
      h_new     <- gru_out[[1L]]$squeeze(2L)$reshape(c(batch, d_n, -1L))
      self$norm(h_new)
    }
  )
}

# ---------------------------------------------------------------------------
# Module: CausalGraphLearner
# ---------------------------------------------------------------------------

.deepnet_gnn_causal_graph_learner_module <- function(d, hidden = 32L) {
  d      <- as.integer(d)
  hidden <- as.integer(hidden)
  torch::nn_module(
    "CausalGraphLearner",
    initialize = function() {
      self$d          <- d
      self$node_embed <- torch::nn_sequential(
        torch::nn_linear(1L, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, hidden)
      )
      self$edge_score <- torch::nn_linear(hidden * 2L, 1L)
    },
    forward = function(x_summary) {
      # x_summary: (batch, d) → graph A: (d, d)
      x_mean <- x_summary$mean(dim = 1L, keepdim = TRUE)        # (1, d)
      h      <- self$node_embed(x_mean$t()$unsqueeze(-1L))$squeeze(2L)  # (d, hidden)
      hd     <- h$size(-1L)
      h_i    <- h$unsqueeze(2L)$expand(c(-1L, self$d, -1L))     # (d, d, hidden)
      h_j    <- h$unsqueeze(1L)$expand(c(self$d, -1L, -1L))     # (d, d, hidden)
      edge_in <- torch::torch_cat(
        list(h_i$reshape(c(self$d * self$d, hd)),
             h_j$reshape(c(self$d * self$d, hd))),
        dim = -1L
      )
      logits <- self$edge_score(edge_in)$reshape(c(self$d, self$d))
      A   <- torch::torch_sigmoid(logits)
      eye <- torch::torch_eye(self$d, device = A$device)
      A * (1.0 - eye)
    }
  )
}

# ---------------------------------------------------------------------------
# Module: CausalGNN
# ---------------------------------------------------------------------------

.deepnet_gnn_causal_gnn_module <- function(d, lag, hidden = 32L,
                                             n_gnn_layers = 3L,
                                             lam_dag = 0.5, lam_sparse = 0.01) {
  d            <- as.integer(d)
  lag          <- as.integer(lag)
  hidden       <- as.integer(hidden)
  n_gnn_layers <- as.integer(n_gnn_layers)
  torch::nn_module(
    "CausalGNN",
    initialize = function() {
      self$d          <- d
      self$lag        <- lag
      self$lam_dag    <- lam_dag
      self$lam_sparse <- lam_sparse
      self$graph_learner <- .deepnet_gnn_causal_graph_learner_module(d, hidden)()
      self$temporal_rnn  <- torch::nn_gru(
        input_size  = 1L,
        hidden_size = hidden,
        num_layers  = 1L,
        batch_first = TRUE
      )
      self$gnn_layers <- torch::nn_module_list(
        lapply(seq_len(n_gnn_layers), function(i)
          .deepnet_gnn_edge_conv_layer(hidden, hidden)()
        )
      )
      self$output_fc <- torch::nn_sequential(
        torch::nn_linear(hidden, hidden),
        torch::nn_gelu(),
        torch::nn_linear(hidden, 1L)
      )
    },
    forward = function(x_seq) {
      # x_seq: (batch, lag, d)
      batch <- x_seq$size(1L)
      lag_n <- x_seq$size(2L)
      d_n   <- x_seq$size(3L)
      x_var  <- x_seq$permute(c(1L, 3L, 2L))$reshape(c(batch * d_n, lag_n, 1L))
      gru_out <- self$temporal_rnn(x_var)
      h_last  <- gru_out[[2L]]$squeeze(1L)$reshape(c(batch, d_n, -1L))
      x_summary <- x_seq$mean(dim = 2L)
      A         <- self$graph_learner$forward(x_summary)
      h_gnn <- h_last
      for (i in seq_len(length(self$gnn_layers))) {
        h_gnn <- self$gnn_layers[[i]]$forward(h_gnn, A)
      }
      pred <- self$output_fc(h_gnn)$squeeze(-1L)
      dag_loss    <- self$lam_dag    * .deepnet_gnn_acyclicity(A)
      sparse_loss <- self$lam_sparse * A$sum()
      list(pred = pred, A = A$detach(), dag_loss = dag_loss, sparse_loss = sparse_loss)
    },
    causal_matrix = function(x_ref) {
      torch::with_no_grad({
        x_t <- torch::torch_tensor(x_ref, dtype = torch::torch_float32(),
                                   device = self$output_fc[[1L]]$weight$device)
        A   <- self$forward(x_t)$A
        as.matrix(A$cpu())
      })
    }
  )
}

# ---------------------------------------------------------------------------
# Module: CUTSPlusLike
# ---------------------------------------------------------------------------

.deepnet_gnn_cuts_module <- function(d, lag, hidden = 32L,
                                      lam_kl = 1e-3, lam_dag = 0.2) {
  d      <- as.integer(d)
  lag    <- as.integer(lag)
  hidden <- as.integer(hidden)
  torch::nn_module(
    "CUTSPlusLike",
    initialize = function() {
      self$d       <- d
      self$lag     <- lag
      self$lam_kl  <- lam_kl
      self$lam_dag <- lam_dag
      self$logits  <- torch::nn_parameter(torch::torch_zeros(c(d, d)))
      self$imputer <- torch::nn_sequential(
        torch::nn_linear(lag * d * 2L, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, lag * d)
      )
      self$encoder <- torch::nn_gru(
        input_size  = 1L,
        hidden_size = hidden,
        num_layers  = 1L,
        batch_first = TRUE
      )
      self$msg <- .deepnet_gnn_edge_conv_layer(hidden, hidden)()
      self$out  <- torch::nn_sequential(
        torch::nn_linear(hidden, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, 1L)
      )
    },
    sample_graph = function() {
      pi  <- torch::torch_sigmoid(self$logits)
      eye <- torch::torch_eye(self$d, device = pi$device)
      pi * (1.0 - eye)
    },
    forward = function(x_seq) {
      # x_seq: (batch, lag, d), zeros where values are missing
      batch <- x_seq$size(1L)
      miss  <- (x_seq$abs() < 1e-12)$float()
      x_flat <- x_seq$reshape(c(batch, -1L))
      m_flat <- miss$reshape(c(batch, -1L))
      x_imp  <- self$imputer(
        torch::torch_cat(list(x_flat, m_flat), dim = -1L)
      )$reshape(c(batch, self$lag, self$d))
      x_filled <- x_seq * (1.0 - miss) + x_imp * miss
      x_var    <- x_filled$permute(c(1L, 3L, 2L))$reshape(
        c(batch * self$d, self$lag, 1L)
      )
      enc_out  <- self$encoder(x_var)
      h_last   <- enc_out[[2L]]$squeeze(1L)$reshape(c(batch, self$d, -1L))
      A        <- self$sample_graph()
      h2       <- self$msg$forward(h_last, A)
      pred     <- self$out(h2)$squeeze(-1L)
      pi       <- torch::torch_clamp(A, 1e-6, 1 - 1e-6)
      p0       <- 0.1
      kl       <- (pi * torch::torch_log(pi / p0) +
                   (1 - pi) * torch::torch_log((1 - pi) / (1 - p0)))$mean()
      dag_loss <- self$lam_dag * .deepnet_gnn_acyclicity(A)
      kl_loss  <- self$lam_kl  * kl
      list(pred = pred, A = A$detach(), imputed = x_filled$detach(),
           dag_loss = dag_loss, kl_loss = kl_loss)
    },
    causal_matrix = function() {
      torch::with_no_grad({
        as.matrix(self$sample_graph()$detach()$cpu())
      })
    }
  )
}

# ---------------------------------------------------------------------------
# Training helper (handles GVAR / CausalGNN / CUTS+)
# ---------------------------------------------------------------------------

.deepnet_gnn_train <- function(model, X_tr, Y_tr, X_val, Y_val,
                                model_name = "model",
                                epochs     = 60L,
                                lr         = 3e-4,
                                patience   = 12L,
                                batch_size = 64L,
                                device     = "cpu",
                                verbose    = FALSE) {
  x_tr_t  <- torch::torch_tensor(X_tr,  dtype = torch::torch_float32(), device = device)
  y_tr_t  <- torch::torch_tensor(Y_tr,  dtype = torch::torch_float32(), device = device)
  x_val_t <- torch::torch_tensor(X_val, dtype = torch::torch_float32(), device = device)
  y_val_t <- torch::torch_tensor(Y_val, dtype = torch::torch_float32(), device = device)

  N        <- x_tr_t$size(1L)
  n_batch  <- ceiling(N / batch_size)
  opt      <- torch::optim_adamw(model$parameters, lr = lr, weight_decay = 1e-4)
  sched    <- torch::lr_cosine_annealing(opt, T_max = epochs)

  best_val   <- Inf
  best_state <- NULL
  no_improve <- 0L
  hist       <- list(train = numeric(epochs), val = numeric(epochs))

  for (epoch in seq_len(epochs)) {
    model$train()
    idx_perm    <- torch::torch_randperm(N, device = device) + 1L
    tr_loss_sum <- 0.0
    for (b in seq_len(n_batch)) {
      start <- (b - 1L) * batch_size + 1L
      end   <- min(b * batch_size, N)
      idx_b <- idx_perm[start:end]
      xb    <- x_tr_t[idx_b, , ]
      yb    <- y_tr_t[idx_b, ]
      opt$zero_grad()
      out  <- model$forward(xb)
      loss <- torch::nnf_mse_loss(out$pred, yb)
      if (!is.null(out$sparse_loss)) loss <- loss + out$sparse_loss
      if (!is.null(out$dag_loss))    loss <- loss + out$dag_loss
      if (!is.null(out$kl_loss))     loss <- loss + out$kl_loss
      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      opt$step()
      tr_loss_sum <- tr_loss_sum + loss$item()
    }
    sched$step()

    model$eval()
    val_loss <- torch::with_no_grad({
      torch::nnf_mse_loss(model$forward(x_val_t)$pred, y_val_t)$item()
    })
    hist$train[epoch] <- tr_loss_sum / max(n_batch, 1L)
    hist$val[epoch]   <- val_loss

    if (val_loss < best_val) {
      best_val   <- val_loss
      best_state <- lapply(model$state_dict(), function(t) t$clone())
      no_improve <- 0L
    } else {
      no_improve <- no_improve + 1L
      if (no_improve >= as.integer(patience)) {
        if (verbose)
          message(sprintf("[%s] Early stopping at epoch %d", toupper(model_name), epoch))
        break
      }
    }
    if (verbose && (epoch %% 10L == 0L || epoch == 1L))
      message(sprintf("[%s] Epoch %3d | Train %.5f | Val %.5f",
                      toupper(model_name), epoch, hist$train[epoch], val_loss))
  }
  if (!is.null(best_state)) model$load_state_dict(best_state)
  hist
}

# =============================================================================
# Public API: gnn_causal_model
# =============================================================================

#' GNN-Based Causal Models for Multivariate Time Series
#'
#' Fits one or more graph-neural-network causal models to multivariate
#' time-series data, translated from \code{05_graph_nn_causal_models_GNN.ipynb}:
#' \itemize{
#'   \item \code{"gvar"} — Graph Vector Autoregression: lag-specific soft
#'     adjacency matrices learned jointly with two stacked GNN message-passing
#'     layers; sparsity + NOTEARS acyclicity penalty.
#'   \item \code{"causalgnn"} — CausalGNN / CD-GNN: per-variable GRU temporal
#'     encoder feeds stacked EdgeConv message-passing layers over a bilinear
#'     soft graph; DAG + sparsity penalties.
#'   \item \code{"cuts"} — CUTS+-inspired model: variational Bernoulli graph
#'     with causal-aware imputation for datasets with missing values; KL +
#'     DAG penalties.
#' }
#'
#' @param data       Numeric matrix or data frame \eqn{(T \times d)}: multivariate
#'   time series.
#' @param lag        Integer; number of lagged time steps used as input window
#'   (default 10).
#' @param models     Character vector; subset of \code{c("gvar","causalgnn","cuts")}
#'   (default all three).
#' @param hidden     Integer; hidden-layer width for all submodules (default 48).
#' @param n_gnn_layers Integer; number of EdgeConv GNN layers for
#'   \code{causal_gnn} (default 3).
#' @param lam_sparse Numeric; L1 sparsity penalty on adjacency entries for
#'   \code{gvar} and \code{causal_gnn} (default 0.005).
#' @param lam_dag    Numeric; NOTEARS acyclicity penalty coefficient (default 0.1).
#' @param lam_kl     Numeric; KL-to-sparse-prior coefficient for \code{cuts}
#'   (default 0.001).
#' @param miss_rate  Numeric in \eqn{[0,1]}; fraction of values to set missing
#'   (synthetic missingness for \code{cuts} demo, default 0.0).
#' @param epochs     Integer; maximum training epochs (default 60).
#' @param lr         Numeric; AdamW learning rate (default 3e-4).
#' @param patience   Integer; early-stopping patience in epochs (default 12).
#' @param batch_size Integer; mini-batch size (default 64).
#' @param val_split  Numeric; fraction of samples reserved for validation
#'   (default 0.2).
#' @param device     Character or \code{NULL}; \code{"cpu"} or \code{"cuda"};
#'   auto-detected when \code{NULL}.
#' @param verbose    Logical; print per-epoch training progress (default FALSE).
#' @param ...        Ignored.
#'
#' @return Object of class \code{gnn_causal_model} (a list) with:
#' \describe{
#'   \item{\code{models}}{Named list of fitted \code{torch::nn_module} objects.}
#'   \item{\code{histories}}{Named list of training / validation MSE histories.}
#'   \item{\code{val_mse}}{Named numeric vector of final validation MSE per model.}
#'   \item{\code{causal_matrices}}{Named list of \eqn{d \times d} causal matrices.}
#'   \item{\code{X_val}}{Validation input array \eqn{(N_{val} \times lag \times d)}.}
#'   \item{\code{lag}, \code{var_names}, \code{device}}{Metadata.}
#' }
#' @seealso \code{\link{causal_matrix_gnn}}, \code{\link{predict.gnn_causal_model}},
#'   \code{\link{gvar_model}}, \code{\link{causal_gnn_model}},
#'   \code{\link{cuts_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # gnn_fit <- gnn_causal_model(data_matrix, lag = 10L, models = c("gvar","causalgnn"))
#' }
#' @export
gnn_causal_model <- function(
    data,
    lag          = 10L,
    models       = c("gvar", "causalgnn", "cuts"),
    hidden       = 48L,
    n_gnn_layers = 3L,
    lam_sparse   = 0.005,
    lam_dag      = 0.1,
    lam_kl       = 1e-3,
    miss_rate    = 0.0,
    epochs       = 60L,
    lr           = 3e-4,
    patience     = 12L,
    batch_size   = 64L,
    val_split    = 0.2,
    device       = NULL,
    verbose      = FALSE,
    ...) {

  if (!requireNamespace("torch", quietly = TRUE))
    stop("gnn_causal_model() requires package 'torch'.", call. = FALSE)

  x <- as.matrix(data)
  if (!is.numeric(x)) stop("`data` must be numeric.", call. = FALSE)

  valid_models <- c("gvar", "causalgnn", "cuts")
  req_models   <- unique(tolower(models))
  bad          <- setdiff(req_models, valid_models)
  if (length(bad))
    stop("Unknown models: ", paste(bad, collapse = ", "), call. = FALSE)

  dev <- .deepnet_gnn_select_device(device)
  d   <- ncol(x)
  lag <- as.integer(lag)

  ds    <- .deepnet_gnn_build_dataset(x, lag, miss_rate = miss_rate)
  n     <- dim(ds$X)[1L]
  sp    <- max(1L, min(n - 1L, floor((1 - val_split) * n)))
  X_tr  <- ds$X[seq_len(sp),       , , drop = FALSE]
  Y_tr  <- ds$Y[seq_len(sp),         , drop = FALSE]
  X_val <- ds$X[(sp + 1L):n,       , , drop = FALSE]
  Y_val <- ds$Y[(sp + 1L):n,         , drop = FALSE]

  fit_models      <- list()
  histories       <- list()
  val_mse_list    <- list()
  causal_matrices <- list()

  for (m in req_models) {
    if (identical(m, "gvar")) {
      model_obj <- .deepnet_gnn_gvar_module(
        d          = d,        lag        = lag,
        hidden     = as.integer(hidden),
        lam_sparse = lam_sparse,
        lam_dag    = lam_dag
      )()
    } else if (identical(m, "causalgnn")) {
      model_obj <- .deepnet_gnn_causal_gnn_module(
        d            = d,        lag          = lag,
        hidden       = as.integer(hidden),
        n_gnn_layers = as.integer(n_gnn_layers),
        lam_dag      = lam_dag,
        lam_sparse   = lam_sparse
      )()
    } else {
      model_obj <- .deepnet_gnn_cuts_module(
        d       = d,        lag     = lag,
        hidden  = as.integer(hidden),
        lam_kl  = lam_kl,
        lam_dag = lam_dag
      )()
    }
    model_obj <- model_obj$to(device = dev)

    hist <- .deepnet_gnn_train(
      model      = model_obj,
      X_tr       = X_tr,  Y_tr  = Y_tr,
      X_val      = X_val, Y_val = Y_val,
      model_name = m,
      epochs     = as.integer(epochs),
      lr         = lr,
      patience   = as.integer(patience),
      batch_size = as.integer(batch_size),
      device     = dev,
      verbose    = verbose
    )

    model_obj$eval()
    x_val_t <- torch::torch_tensor(X_val, dtype = torch::torch_float32(), device = dev)
    y_val_t <- torch::torch_tensor(Y_val, dtype = torch::torch_float32(), device = dev)
    vmse    <- torch::with_no_grad({
      torch::nnf_mse_loss(model_obj$forward(x_val_t)$pred, y_val_t)$item()
    })

    cmat <- tryCatch({
      if (identical(m, "gvar")) {
        model_obj$causal_matrix()
      } else if (identical(m, "causalgnn")) {
        x_ref <- X_tr[seq_len(min(128L, nrow(X_tr))), , , drop = FALSE]
        model_obj$causal_matrix(x_ref)
      } else {
        model_obj$causal_matrix()
      }
    }, error = function(e) matrix(NA_real_, d, d))
    rownames(cmat) <- colnames(cmat) <- colnames(x)

    fit_models[[m]]      <- model_obj
    histories[[m]]       <- hist
    val_mse_list[[m]]    <- vmse
    causal_matrices[[m]] <- cmat
  }

  structure(
    list(
      models          = fit_models,
      histories       = histories,
      val_mse         = unlist(val_mse_list),
      causal_matrices = causal_matrices,
      X_val           = X_val,
      lag             = lag,
      var_names       = colnames(x),
      device          = dev
    ),
    class = "gnn_causal_model"
  )
}

#' Predict from fitted GNN causal models
#'
#' @param object  Fitted \code{gnn_causal_model} object.
#' @param model   One of \code{"gvar"}, \code{"causalgnn"}, \code{"cuts"}.
#' @param newdata 3-D numeric array \code{(N x lag x d)}: windowed inputs.
#' @param ...     Ignored.
#' @return Numeric matrix \code{(N x d)} of one-step-ahead predictions.
#' @seealso \code{\link{gnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # predict(gnn_fit, model = "gvar", newdata = X_new)
#' }
#' @export
predict.gnn_causal_model <- function(
    object,
    model   = c("gvar", "causalgnn", "cuts"),
    newdata,
    ...) {
  model <- match.arg(model)
  if (is.null(object$models[[model]]))
    stop("Model not fitted: ", model, call. = FALSE)
  fit <- object$models[[model]]
  fit$eval()
  dev <- if (is.null(object$device)) "cpu" else object$device
  torch::with_no_grad({
    x_t <- torch::torch_tensor(newdata, dtype = torch::torch_float32(), device = dev)
    as.matrix(fit$forward(x_t)$pred$detach()$cpu())
  })
}

#' Extract causal matrix from a fitted GNN causal model
#'
#' Returns the \eqn{d \times d} adjacency matrix whose \code{[i,j]} entry
#' encodes the inferred causal influence of variable \eqn{j} on variable
#' \eqn{i}.
#'
#' @param object Fitted \code{gnn_causal_model} object.
#' @param model  One of \code{"gvar"}, \code{"causalgnn"}, \code{"cuts"}.
#' @return Named numeric matrix \eqn{d \times d}.
#' @seealso \code{\link{gnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # causal_matrix_gnn(gnn_fit, model = "gvar")
#' }
#' @export
causal_matrix_gnn <- function(
    object,
    model = c("gvar", "causalgnn", "cuts")) {
  model <- match.arg(model)
  cm    <- object$causal_matrices[[model]]
  if (is.null(cm)) stop("Model not fitted: ", model, call. = FALSE)
  cm
}

#' Convenience wrapper: fit only the GVAR model
#'
#' Calls \code{\link{gnn_causal_model}} with \code{models = "gvar"}.
#'
#' @inheritParams gnn_causal_model
#' @return Object of class \code{gnn_causal_model}.
#' @seealso \code{\link{gnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # gvar_model(x, lag = 10L)
#' }
#' @export
gvar_model <- function(data, ...) gnn_causal_model(data, models = "gvar", ...)

#' Convenience wrapper: fit only the CausalGNN model
#'
#' Calls \code{\link{gnn_causal_model}} with \code{models = "causalgnn"}.
#'
#' @inheritParams gnn_causal_model
#' @return Object of class \code{gnn_causal_model}.
#' @seealso \code{\link{gnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # causal_gnn_model(x, lag = 10L)
#' }
#' @export
causal_gnn_model <- function(data, ...) gnn_causal_model(data, models = "causalgnn", ...)

#' Convenience wrapper: fit only the CUTS+ model
#'
#' Calls \code{\link{gnn_causal_model}} with \code{models = "cuts"}.
#'
#' @inheritParams gnn_causal_model
#' @return Object of class \code{gnn_causal_model}.
#' @seealso \code{\link{gnn_causal_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # cuts_model(x, lag = 10L, miss_rate = 0.15)
#' }
#' @export
cuts_model <- function(data, ...) gnn_causal_model(data, models = "cuts", ...)

#' CamelCase alias: gnnCausalModel
#' @rdname gnn_causal_model
#' @export
gnnCausalModel <- function(data, ...) gnn_causal_model(data, ...)

#' CamelCase alias: GVARModel
#' @rdname gvar_model
#' @export
GVARModel <- function(data, ...) gvar_model(data, ...)

#' CamelCase alias: CausalGNNModel
#' @rdname causal_gnn_model
#' @export
CausalGNNModel <- function(data, ...) causal_gnn_model(data, ...)

#' CamelCase alias: CUTSModel
#' @rdname cuts_model
#' @export
CUTSModel <- function(data, ...) cuts_model(data, ...)

# ===========================================================================
# Counterfactual / Potential Outcomes Models (v0.4.0)
# ===========================================================================
#
#   • DeepSynth  — Neural Synthetic Control: GRU encoder, attention-weighted
#                  donor series form counterfactual baseline; factual and
#                  counterfactual heads produce ITE estimates.
#                  Public API: deep_synth_model() / DeepSynthModel()
#   • CRN        — Counterfactual Recurrent Network: GRU encoder with an
#                  adversarial treatment-balancing discriminator (GradRev-
#                  style); decoder conditioned on representation + do(T).
#                  Public API: crn_model() / CRNModel()
#   • G-Net      — Deep G-Computation: GRU backbone, covariate transition
#                  head and outcome head; marginal counterfactual prediction
#                  via sequential substitution of intervened treatment.
#                  Public API: gnet_model() / GNetModel()
#   Unified entry point:   counterfactual_model() / CounterfactualModel()
#   ATE/ITE accessors:     ate_counterfactual(), ite_counterfactual()
#   Prediction:            predict.counterfactual_model()
#
# References:
#   Abadie, A., & Gardeazabal, J. (2003). The economic costs of conflict:
#     A case study of the Basque Country. Am. Econ. Rev. 93(1), 113-132.
#   Bica, I., Alaa, A. M., Jordon, J., & van der Schaar, M. (2020).
#     Estimating counterfactual treatment outcomes over time through
#     adversarially balanced representations. ICLR 2020.
#   Li, R., & van der Schaar, M. (2021). G-Net: a recurrent network approach
#     to G-computation for counterfactual prediction under a dynamic
#     treatment regime. ML4H Workshop NeurIPS 2021.
# ===========================================================================

# ---------------------------------------------------------------------------
# Internal helper: build counterfactual dataset
# ---------------------------------------------------------------------------

.deepnet_cf_build_dataset <- function(data_np, treatment, outcome, lag) {
  lag <- as.integer(lag)
  n   <- nrow(data_np)
  if (n <= lag + 1L)
    stop("Not enough time points for lag = ", lag)

  idx <- seq_len(n - lag - 1L)
  X      <- array(0.0, dim = c(length(idx), lag, ncol(data_np)))
  T_hist <- matrix(0.0, nrow = length(idx), ncol = lag)
  T_next <- numeric(length(idx))
  Y_next <- numeric(length(idx))

  for (k in seq_along(idx)) {
    t <- lag + k - 1L
    X[k, , ]    <- data_np[(t - lag + 1L):t, , drop = FALSE]
    T_hist[k, ] <- treatment[(t - lag + 1L):t]
    T_next[k]   <- treatment[t + 1L]
    Y_next[k]   <- outcome[t + 1L]
  }
  list(X = X, T_hist = T_hist, T_next = T_next, Y_next = Y_next)
}

# ---------------------------------------------------------------------------
# Internal helper: training loop for counterfactual models
# ---------------------------------------------------------------------------

.deepnet_cf_train <- function(model, X_tr, Th_tr, Tn_tr, Y_tr,
                               X_val, Th_val, Tn_val, Y_val,
                               model_name = "model",
                               epochs     = 80L,
                               lr         = 3e-4,
                               patience   = 15L,
                               batch_size = 64L,
                               device     = "cpu",
                               verbose    = FALSE) {
  epochs     <- as.integer(epochs)
  patience   <- as.integer(patience)
  batch_size <- as.integer(batch_size)

  model <- model$to(device = device)
  opt   <- torch::optim_adam(model$parameters, lr = lr, weight_decay = 1e-4)
  sched <- torch::lr_cosine_annealing(opt, T_max = epochs)

  n_tr  <- nrow(X_tr)
  n_val <- nrow(X_val)

  best_val <- Inf
  best_wts <- NULL
  no_imp   <- 0L
  hist_tr  <- numeric(0)
  hist_val <- numeric(0)

  Xt_val  <- torch::torch_tensor(X_val,  dtype = torch::torch_float32(), device = device)
  Tht_val <- torch::torch_tensor(Th_val, dtype = torch::torch_float32(), device = device)
  Tnt_val <- torch::torch_tensor(Tn_val, dtype = torch::torch_float32(), device = device)
  Yt_val  <- torch::torch_tensor(Y_val,  dtype = torch::torch_float32(), device = device)

  for (ep in seq_len(epochs)) {
    model$train()
    idx_perm <- sample(n_tr)
    batches  <- split(idx_perm, ceiling(seq_along(idx_perm) / batch_size))
    ep_loss  <- 0.0

    coro::loop(for (bidx in batches) {
      Xb  <- torch::torch_tensor(X_tr[bidx, , , drop = FALSE],
                                  dtype = torch::torch_float32(), device = device)
      Thb <- torch::torch_tensor(Th_tr[bidx, , drop = FALSE],
                                  dtype = torch::torch_float32(), device = device)
      Tnb <- torch::torch_tensor(Tn_tr[bidx],
                                  dtype = torch::torch_float32(), device = device)
      Yb  <- torch::torch_tensor(Y_tr[bidx],
                                  dtype = torch::torch_float32(), device = device)
      opt$zero_grad()
      out  <- model$forward(Xb, Thb, Tnb)
      loss <- model$cf_loss(out, Yb)
      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      opt$step()
      ep_loss <- ep_loss + loss$item()
    })
    sched$step()
    ep_loss <- ep_loss / length(batches)

    model$eval()
    v_loss <- torch::with_no_grad({
      out_v <- model$forward(Xt_val, Tht_val, Tnt_val)
      model$cf_loss(out_v, Yt_val)$item()
    })

    hist_tr  <- c(hist_tr,  ep_loss)
    hist_val <- c(hist_val, v_loss)

    if (verbose && ep %% 20L == 0L)
      message(sprintf("[%-12s] epoch %03d | train %.5f | val %.5f",
                      model_name, ep, ep_loss, v_loss))

    if (v_loss < best_val) {
      best_val <- v_loss
      best_wts <- lapply(model$state_dict(), function(p) p$clone()$detach())
      no_imp   <- 0L
    } else {
      no_imp <- no_imp + 1L
      if (no_imp >= patience) {
        if (verbose)
          message(sprintf("[%s] early stop at epoch %d", model_name, ep))
        break
      }
    }
  }

  if (!is.null(best_wts)) model$load_state_dict(best_wts)
  list(train = hist_tr, val = hist_val, best_val = best_val)
}

# ---------------------------------------------------------------------------
# Internal helper: evaluate MSE on a loader
# ---------------------------------------------------------------------------

.deepnet_cf_eval_mse <- function(model, X, Th, Tn, Y, batch_size = 256L, device = "cpu") {
  model$eval()
  batch_size <- as.integer(batch_size)
  n <- nrow(X)
  total <- 0.0
  n_total <- 0L
  torch::with_no_grad({
    start <- 1L
    while (start <= n) {
      end <- min(start + batch_size - 1L, n)
      bidx <- start:end
      Xb  <- torch::torch_tensor(X[bidx, , , drop = FALSE],
                                  dtype = torch::torch_float32(), device = device)
      Thb <- torch::torch_tensor(Th[bidx, , drop = FALSE],
                                  dtype = torch::torch_float32(), device = device)
      Tnb <- torch::torch_tensor(Tn[bidx],
                                  dtype = torch::torch_float32(), device = device)
      Yb  <- torch::torch_tensor(Y[bidx],
                                  dtype = torch::torch_float32(), device = device)
      pred <- model$forward(Xb, Thb, Tnb)$pred
      total   <- total + torch::nnf_mse_loss(pred, Yb, reduction = "sum")$item()
      n_total <- n_total + length(bidx)
      start <- end + 1L
    }
  })
  total / max(n_total, 1L)
}

# ---------------------------------------------------------------------------
# Internal helper: collect ITE over a split
# ---------------------------------------------------------------------------

.deepnet_cf_collect_ite <- function(model, X, Th, Tn, batch_size = 256L, device = "cpu") {
  model$eval()
  batch_size <- as.integer(batch_size)
  n    <- nrow(X)
  ites <- numeric(0)
  torch::with_no_grad({
    start <- 1L
    while (start <= n) {
      end  <- min(start + batch_size - 1L, n)
      bidx <- start:end
      Xb  <- torch::torch_tensor(X[bidx, , , drop = FALSE],
                                  dtype = torch::torch_float32(), device = device)
      Thb <- torch::torch_tensor(Th[bidx, , drop = FALSE],
                                  dtype = torch::torch_float32(), device = device)
      Tnb <- torch::torch_tensor(Tn[bidx],
                                  dtype = torch::torch_float32(), device = device)
      out  <- model$forward(Xb, Thb, Tnb)
      ites <- c(ites, as.numeric(out$ite$detach()$cpu()))
      start <- end + 1L
    }
  })
  ites
}

# ---------------------------------------------------------------------------
# Module: SharedEncoder (GRU + FC) — used by DeepSynth
# ---------------------------------------------------------------------------

.deepnet_cf_shared_encoder <- function(input_dim, latent_dim = 32L) {
  input_dim  <- as.integer(input_dim)
  latent_dim <- as.integer(latent_dim)
  torch::nn_module(
    "SharedEncoder",
    initialize = function() {
      self$rnn <- torch::nn_gru(
        input_size  = input_dim,
        hidden_size = latent_dim,
        num_layers  = 2L,
        batch_first = TRUE,
        dropout     = 0.1
      )
      self$fc <- torch::nn_linear(latent_dim, latent_dim)
    },
    forward = function(x) {
      h_all <- self$rnn(x)
      h     <- h_all[[2L]][2L, , ]
      torch::nnf_relu(self$fc(h))
    }
  )
}

# ---------------------------------------------------------------------------
# Module: DeepSynth (Neural Synthetic Control)
# ---------------------------------------------------------------------------

.deepnet_cf_deepsynth_module <- function(d, lag, treat_idx, out_idx,
                                          latent_dim = 32L, dropout = 0.15) {
  d          <- as.integer(d)
  lag        <- as.integer(lag)
  treat_idx  <- as.integer(treat_idx)   # 0-based
  out_idx    <- as.integer(out_idx)     # 0-based
  latent_dim <- as.integer(latent_dim)
  scale      <- sqrt(latent_dim)

  torch::nn_module(
    "DeepSynth",
    initialize = function() {
      self$d         <- d
      self$treat_idx <- treat_idx
      self$out_idx   <- out_idx
      self$scale     <- scale
      self$encoder   <- .deepnet_cf_shared_encoder(d, latent_dim)()
      self$q_proj    <- torch::nn_linear(latent_dim, latent_dim)
      self$k_proj    <- torch::nn_linear(latent_dim, latent_dim)
      self$factual_head <- torch::nn_sequential(
        torch::nn_linear(latent_dim + 1L, 32L),
        torch::nn_relu(),
        torch::nn_dropout(dropout),
        torch::nn_linear(32L, 1L)
      )
      self$cf_head <- torch::nn_sequential(
        torch::nn_linear(latent_dim, 32L),
        torch::nn_relu(),
        torch::nn_dropout(dropout),
        torch::nn_linear(32L, 1L)
      )
    },
    forward = function(x, t_hist, t_next) {
      keep <- setdiff(0L:(self$d - 1L), c(self$treat_idx, self$out_idx))
      z_treated <- self$encoder(x)

      z_donors <- torch::torch_stack(
        lapply(keep, function(j) {
          xj <- x[, , j + 1L, drop = FALSE]$`repeat`(c(1L, 1L, self$d))
          self$encoder(xj)
        }),
        dim = 2L
      )

      q      <- self$q_proj(z_treated)$unsqueeze(2L)
      k      <- self$k_proj(z_donors)
      scores <- torch::torch_bmm(q, k$transpose(2L, 3L)) / self$scale
      w      <- torch::nnf_softmax(scores, dim = -1L)
      z_cf   <- torch::torch_bmm(w, z_donors)$squeeze(2L)

      y_fact    <- self$factual_head(
        torch::torch_cat(list(z_treated, t_next$unsqueeze(-1L)), dim = -1L)
      )$squeeze(-1L)
      y_counter <- self$cf_head(z_cf)$squeeze(-1L)

      list(
        pred          = y_fact,
        ycounter      = y_counter,
        ite           = y_fact - y_counter,
        donor_weights = w$squeeze(2L)$detach()
      )
    },
    cf_loss = function(out, y_true) {
      torch::nnf_mse_loss(out$pred, y_true)
    }
  )
}

# ---------------------------------------------------------------------------
# Module: TreatmentDiscriminator — used by CRN
# ---------------------------------------------------------------------------

.deepnet_cf_treatment_discriminator <- function(rep_dim, hidden = 32L) {
  rep_dim <- as.integer(rep_dim)
  hidden  <- as.integer(hidden)
  torch::nn_module(
    "TreatmentDiscriminator",
    initialize = function() {
      self$net <- torch::nn_sequential(
        torch::nn_linear(rep_dim, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, 1L)
      )
    },
    forward = function(z) {
      self$net(z)$squeeze(-1L)
    }
  )
}

# ---------------------------------------------------------------------------
# Module: CRN (Counterfactual Recurrent Network)
# ---------------------------------------------------------------------------

.deepnet_cf_crn_module <- function(d, lag, rep_dim = 32L, hidden = 64L,
                                    lam_adv = 0.2, dropout = 0.2) {
  d       <- as.integer(d)
  lag     <- as.integer(lag)
  rep_dim <- as.integer(rep_dim)
  hidden  <- as.integer(hidden)

  torch::nn_module(
    "CRN",
    initialize = function() {
      self$lam_adv <- lam_adv
      self$encoder <- torch::nn_gru(
        input_size  = d + 2L,
        hidden_size = hidden,
        num_layers  = 2L,
        batch_first = TRUE,
        dropout     = dropout
      )
      self$rep_proj <- torch::nn_sequential(
        torch::nn_linear(hidden, rep_dim),
        torch::nn_layer_norm(rep_dim),
        torch::nn_tanh()
      )
      self$discriminator <- .deepnet_cf_treatment_discriminator(rep_dim, 32L)()
      self$decoder <- torch::nn_sequential(
        torch::nn_linear(rep_dim + 1L, hidden),
        torch::nn_relu(),
        torch::nn_dropout(dropout),
        torch::nn_linear(hidden, hidden %/% 2L),
        torch::nn_relu(),
        torch::nn_linear(hidden %/% 2L, 1L)
      )
    },
    encode = function(x, t_hist) {
      b <- x$size(1L); l <- x$size(2L)
      t_feat <- t_hist$unsqueeze(-1L)
      y_lag  <- torch::torch_zeros(c(b, l, 1L), device = x$device)
      aug    <- torch::torch_cat(list(x, t_feat, y_lag), dim = -1L)
      h      <- self$encoder(aug)[[2L]][2L, , ]
      self$rep_proj(h)
    },
    forward = function(x, t_hist, t_next) {
      r    <- self$encode(x, t_hist)
      pred <- self$decoder(
        torch::torch_cat(list(r, t_next$unsqueeze(-1L)), dim = -1L)
      )$squeeze(-1L)
      disc_logits <- self$discriminator(r)
      adv_loss    <- torch::nnf_binary_cross_entropy_with_logits(disc_logits, t_next)
      z0 <- torch::torch_zeros_like(t_next)$unsqueeze(-1L)
      z1 <- torch::torch_ones_like(t_next)$unsqueeze(-1L)
      y0 <- self$decoder(torch::torch_cat(list(r, z0), dim = -1L))$squeeze(-1L)
      y1 <- self$decoder(torch::torch_cat(list(r, z1), dim = -1L))$squeeze(-1L)
      list(
        pred        = pred,
        disc_logits = disc_logits,
        adv_loss    = adv_loss,
        y0          = y0,
        y1          = y1,
        ite         = y1 - y0
      )
    },
    cf_loss = function(out, y_true) {
      pred_loss <- torch::nnf_mse_loss(out$pred, y_true)
      pred_loss - self$lam_adv * out$adv_loss
    }
  )
}

# ---------------------------------------------------------------------------
# Module: G-Net (Deep G-Computation)
# ---------------------------------------------------------------------------

.deepnet_cf_gnet_module <- function(d, hidden = 64L, dropout = 0.15) {
  d      <- as.integer(d)
  hidden <- as.integer(hidden)

  torch::nn_module(
    "GNet",
    initialize = function() {
      self$d        <- d
      self$backbone <- torch::nn_gru(
        input_size  = d + 1L,
        hidden_size = hidden,
        num_layers  = 2L,
        batch_first = TRUE,
        dropout     = dropout
      )
      self$x_head <- torch::nn_sequential(
        torch::nn_linear(hidden + 1L, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, d)
      )
      self$y_head <- torch::nn_sequential(
        torch::nn_linear(hidden + d + 1L, hidden),
        torch::nn_relu(),
        torch::nn_linear(hidden, 1L)
      )
    },
    encode = function(x, t_hist) {
      aug <- torch::torch_cat(list(x, t_hist$unsqueeze(-1L)), dim = -1L)
      self$backbone(aug)[[2L]][2L, , ]
    },
    forward = function(x, t_hist, t_next) {
      h       <- self$encode(x, t_hist)
      tn_e    <- t_next$unsqueeze(-1L)
      x_next  <- self$x_head(torch::torch_cat(list(h, tn_e), dim = -1L))
      y_hat   <- self$y_head(torch::torch_cat(list(h, x_next, tn_e), dim = -1L))$squeeze(-1L)

      z0    <- torch::torch_zeros_like(t_next)$unsqueeze(-1L)
      z1    <- torch::torch_ones_like(t_next)$unsqueeze(-1L)
      x0    <- self$x_head(torch::torch_cat(list(h, z0), dim = -1L))
      x1    <- self$x_head(torch::torch_cat(list(h, z1), dim = -1L))
      y0    <- self$y_head(torch::torch_cat(list(h, x0, z0), dim = -1L))$squeeze(-1L)
      y1    <- self$y_head(torch::torch_cat(list(h, x1, z1), dim = -1L))$squeeze(-1L)

      list(
        pred       = y_hat,
        x_next_hat = x_next$detach(),
        y0         = y0,
        y1         = y1,
        ite        = y1 - y0
      )
    },
    cf_loss = function(out, y_true) {
      torch::nnf_mse_loss(out$pred, y_true)
    }
  )
}

# ---------------------------------------------------------------------------
# Public API: counterfactual_model()
# ---------------------------------------------------------------------------

#' Counterfactual / Potential Outcomes Models
#'
#' Fits one or more deep counterfactual models to a multivariate time-series
#' with a binary treatment indicator and a scalar outcome.  Three architectures
#' are available:
#'
#' \describe{
#'   \item{\code{"deepsynth"}}{
#'     \strong{DeepSynth — Neural Synthetic Control.}
#'     A GRU encoder maps the treated unit's covariate history to a latent
#'     query vector.  A set of donor variables (all columns except the
#'     treatment and outcome columns) are encoded the same way and treated as
#'     keys/values in a scaled dot-product attention layer.  The attention
#'     weights define a soft synthetic control; a factual head (query +
#'     \eqn{T_{t+1}}) and a counterfactual head (weighted donor summary)
#'     produce the factual prediction and the counterfactual baseline,
#'     enabling ITE estimation.
#'   }
#'   \item{\code{"crn"}}{
#'     \strong{CRN — Counterfactual Recurrent Network.}
#'     A two-layer GRU encodes the covariate-treatment history into a
#'     balanced representation via a treatment discriminator loss term
#'     \eqn{-\lambda_{\text{adv}} \mathcal{L}_{\text{BCE}}(D(r), T)}.
#'     A decoder conditioned on \eqn{r} and \eqn{\text{do}(T) \in \{0,1\}}
#'     produces \eqn{\hat{Y}(0)} and \eqn{\hat{Y}(1)}, giving ITE and ATE.
#'   }
#'   \item{\code{"gnet"}}{
#'     \strong{G-Net — Deep G-Computation.}
#'     A two-layer GRU backbone encodes covariate + treatment history.  A
#'     covariate transition head \eqn{\hat{X}_{t+1}} and an outcome head
#'     \eqn{\hat{Y}_{t+1}(\bar{a})} are jointly learned.  Counterfactual
#'     outcomes under \eqn{\text{do}(T=0)} and \eqn{\text{do}(T=1)} are
#'     computed by substituting the intervened treatment value in both heads.
#'   }
#' }
#'
#' @param data      Numeric matrix or data frame \code{(T x d)}: multivariate
#'   time-series (rows = time steps, cols = variables).
#' @param treatment Numeric or integer vector of length \eqn{T}: binary
#'   treatment indicator (0/1) at each time step.
#' @param outcome   Numeric vector of length \eqn{T}: scalar outcome at each
#'   time step.  One-step-ahead prediction is used: \eqn{Y_{t+1}} is the
#'   target for window ending at \eqn{t}.
#' @param treat_col Integer (1-indexed) column of \code{data} that corresponds
#'   to the treatment variable.  Used by DeepSynth to exclude it from the
#'   donor pool.  Default \code{1L}.
#' @param out_col   Integer (1-indexed) column of \code{data} that corresponds
#'   to the outcome variable.  Used by DeepSynth to exclude it from the
#'   donor pool.  Default \code{2L}.
#' @param lag       Integer lag window fed to models (default \code{20L}).
#' @param models    Character vector subset of
#'   \code{c("deepsynth","crn","gnet")} (default all three).
#' @param latent_dim Integer latent dimension for DeepSynth encoder
#'   (default \code{32L}).
#' @param rep_dim   Integer representation dimension for CRN encoder
#'   (default \code{32L}).
#' @param hidden    Integer hidden dimension for CRN and G-Net GRUs
#'   (default \code{64L}).
#' @param lam_adv   Adversarial balancing weight \eqn{\lambda_{\text{adv}}}
#'   in the CRN loss (default \code{0.2}).
#' @param dropout   Dropout probability for all models (default \code{0.15}).
#' @param epochs    Maximum training epochs (default \code{80L}).
#' @param lr        Adam learning rate (default \code{3e-4}).
#' @param patience  Early-stopping patience (default \code{15L}).
#' @param batch_size Mini-batch size (default \code{64L}).
#' @param val_split Validation fraction in (0, 1) (default \code{0.2}).
#' @param device    Torch device string (\code{"cpu"} or \code{"cuda"}).
#'   \code{NULL} auto-selects.
#' @param verbose   Print per-epoch progress (default \code{FALSE}).
#' @param ...       Ignored.
#'
#' @return Object of class \code{counterfactual_model} containing:
#' \describe{
#'   \item{\code{models}}{Named list of fitted torch modules.}
#'   \item{\code{histories}}{Named list of train/val loss vectors per model.}
#'   \item{\code{val_mse}}{Named numeric vector of final validation MSE.}
#'   \item{\code{ate}}{Named numeric vector of estimated Average Treatment
#'     Effect (ATE) on the validation split.}
#'   \item{\code{ite_val}}{Named list of ITE vectors on the validation split.}
#'   \item{\code{lag}}{Integer lag used.}
#'   \item{\code{d}}{Number of variables.}
#'   \item{\code{var_names}}{Character vector of variable names.}
#'   \item{\code{val_data}}{List with \code{X}, \code{T_hist}, \code{T_next},
#'     \code{Y_next} arrays for the validation split.}
#'   \item{\code{device}}{Torch device string.}
#' }
#'
#' @references
#'   Abadie, A., & Gardeazabal, J. (2003). The economic costs of conflict.
#'   \emph{Am. Econ. Rev.} 93(1), 113-132.
#'
#'   Bica, I., Alaa, A. M., Jordon, J., & van der Schaar, M. (2020).
#'   Estimating counterfactual treatment outcomes over time through
#'   adversarially balanced representations. \emph{ICLR 2020}.
#'
#'   Li, R., & van der Schaar, M. (2021). G-Net: a recurrent network approach
#'   to G-computation for counterfactual prediction under a dynamic
#'   treatment regime. \emph{ML4H @ NeurIPS 2021}.
#'
#' @seealso \code{\link{deep_synth_model}}, \code{\link{crn_model}},
#'   \code{\link{gnet_model}}, \code{\link{predict.counterfactual_model}},
#'   \code{\link{ate_counterfactual}}, \code{\link{ite_counterfactual}},
#'   \code{\link{RCausalML-package}}
#'
#' @examples
#' \dontrun{
#' set.seed(42)
#' T_len <- 500L; d <- 6L
#' data_mat  <- matrix(rnorm(T_len * d), T_len, d)
#' treatment <- as.numeric(data_mat[, 1] > 0)
#' outcome   <- data_mat[, 2]
#' fit <- counterfactual_model(data_mat, treatment, outcome,
#'                             treat_col = 1L, out_col = 2L,
#'                             lag = 10L, models = c("deepsynth", "crn", "gnet"),
#'                             epochs = 20L, verbose = TRUE)
#' print(fit$val_mse)
#' print(fit$ate)
#' }
#' @export
counterfactual_model <- function(
    data,
    treatment,
    outcome,
    treat_col  = 1L,
    out_col    = 2L,
    lag        = 20L,
    models     = c("deepsynth", "crn", "gnet"),
    latent_dim = 32L,
    rep_dim    = 32L,
    hidden     = 64L,
    lam_adv    = 0.2,
    dropout    = 0.15,
    epochs     = 80L,
    lr         = 3e-4,
    patience   = 15L,
    batch_size = 64L,
    val_split  = 0.2,
    device     = NULL,
    verbose    = FALSE,
    ...) {

  if (!requireNamespace("torch", quietly = TRUE))
    stop("counterfactual_model() requires package 'torch'.", call. = FALSE)
  if (!requireNamespace("coro", quietly = TRUE))
    stop("counterfactual_model() requires package 'coro'.", call. = FALSE)

  x_mat <- as.matrix(data)
  if (!is.numeric(x_mat)) stop("`data` must be numeric.", call. = FALSE)
  treatment <- as.numeric(treatment)
  outcome   <- as.numeric(outcome)
  n_obs     <- nrow(x_mat)
  if (length(treatment) != n_obs || length(outcome) != n_obs)
    stop("`treatment` and `outcome` must have length nrow(data).", call. = FALSE)

  valid_models <- c("deepsynth", "crn", "gnet")
  req_models   <- unique(tolower(models))
  bad <- setdiff(req_models, valid_models)
  if (length(bad)) stop("Unknown models: ", paste(bad, collapse = ", "), call. = FALSE)

  dev <- if (is.null(device))
    (if (torch::cuda_is_available()) "cuda" else "cpu")
  else device

  d         <- ncol(x_mat)
  lag       <- as.integer(lag)
  treat_col <- as.integer(treat_col)
  out_col   <- as.integer(out_col)
  var_names <- if (!is.null(colnames(x_mat))) colnames(x_mat) else paste0("V", seq_len(d))

  treat_idx0 <- treat_col - 1L
  out_idx0   <- out_col   - 1L

  ds    <- .deepnet_cf_build_dataset(x_mat, treatment, outcome, lag)
  n_ds  <- nrow(ds$X)
  split_idx <- max(1L, min(n_ds - 1L, floor((1 - val_split) * n_ds)))

  X_tr   <- ds$X[seq_len(split_idx), , , drop = FALSE]
  Th_tr  <- ds$T_hist[seq_len(split_idx), , drop = FALSE]
  Tn_tr  <- ds$T_next[seq_len(split_idx)]
  Y_tr   <- ds$Y_next[seq_len(split_idx)]

  X_val  <- ds$X[(split_idx + 1L):n_ds, , , drop = FALSE]
  Th_val <- ds$T_hist[(split_idx + 1L):n_ds, , drop = FALSE]
  Tn_val <- ds$T_next[(split_idx + 1L):n_ds]
  Y_val  <- ds$Y_next[(split_idx + 1L):n_ds]

  fit_models   <- list()
  histories    <- list()
  val_mse_list <- list()
  ate_list     <- list()
  ite_val_list <- list()

  for (m in req_models) {
    model_obj <- switch(m,
      deepsynth = .deepnet_cf_deepsynth_module(
        d = d, lag = lag,
        treat_idx  = treat_idx0, out_idx = out_idx0,
        latent_dim = as.integer(latent_dim), dropout = dropout)(),
      crn = .deepnet_cf_crn_module(
        d = d, lag = lag,
        rep_dim = as.integer(rep_dim), hidden = as.integer(hidden),
        lam_adv = lam_adv, dropout = dropout)(),
      gnet = .deepnet_cf_gnet_module(
        d = d, hidden = as.integer(hidden), dropout = dropout)()
    )
    model_obj <- model_obj$to(device = dev)

    hist <- .deepnet_cf_train(
      model      = model_obj,
      X_tr = X_tr, Th_tr = Th_tr, Tn_tr = Tn_tr, Y_tr = Y_tr,
      X_val = X_val, Th_val = Th_val, Tn_val = Tn_val, Y_val = Y_val,
      model_name = m,
      epochs     = as.integer(epochs),
      lr         = lr,
      patience   = as.integer(patience),
      batch_size = as.integer(batch_size),
      device     = dev,
      verbose    = verbose
    )

    vmse <- .deepnet_cf_eval_mse(model_obj, X_val, Th_val, Tn_val, Y_val,
                                  batch_size = 256L, device = dev)
    ites <- .deepnet_cf_collect_ite(model_obj, X_val, Th_val, Tn_val,
                                     batch_size = 256L, device = dev)

    fit_models[[m]]   <- model_obj
    histories[[m]]    <- hist
    val_mse_list[[m]] <- vmse
    ate_list[[m]]     <- mean(ites)
    ite_val_list[[m]] <- ites
  }

  val_mse <- unlist(val_mse_list)
  ate      <- unlist(ate_list)

  structure(
    list(
      models    = fit_models,
      histories = histories,
      val_mse   = val_mse,
      ate       = ate,
      ite_val   = ite_val_list,
      lag       = lag,
      d         = d,
      var_names = var_names,
      val_data  = list(X = X_val, T_hist = Th_val, T_next = Tn_val, Y_next = Y_val),
      device    = dev
    ),
    class = "counterfactual_model"
  )
}

#' Predict from a fitted counterfactual model
#'
#' Runs the forward pass of one fitted model in a
#' \code{counterfactual_model} object on new data.
#'
#' @param object   Fitted \code{counterfactual_model} object.
#' @param model    One of \code{"deepsynth"}, \code{"crn"}, \code{"gnet"}.
#' @param X        3-D numeric array \code{(N x lag x d)}: covariate history.
#' @param T_hist   2-D numeric array \code{(N x lag)}: treatment history.
#' @param T_next   Numeric vector length \code{N}: treatment at next step.
#' @param ...      Ignored.
#' @return List with \code{pred}, \code{y0}, \code{y1}, and \code{ite}
#'   numeric vectors of length \code{N}.
#' @seealso \code{\link{counterfactual_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # predict(fit, model = "crn", X = X_new, T_hist = Th_new, T_next = Tn_new)
#' }
#' @export
predict.counterfactual_model <- function(
    object,
    model  = c("deepsynth", "crn", "gnet"),
    X,
    T_hist,
    T_next,
    ...) {
  model <- match.arg(model)
  if (is.null(object$models[[model]]))
    stop("Model not fitted: ", model, call. = FALSE)
  fit <- object$models[[model]]
  fit$eval()
  dev <- if (is.null(object$device)) "cpu" else object$device
  torch::with_no_grad({
    Xt  <- torch::torch_tensor(X,      dtype = torch::torch_float32(), device = dev)
    Tht <- torch::torch_tensor(T_hist, dtype = torch::torch_float32(), device = dev)
    Tnt <- torch::torch_tensor(T_next, dtype = torch::torch_float32(), device = dev)
    out <- fit$forward(Xt, Tht, Tnt)
    list(
      pred    = as.numeric(out$pred$detach()$cpu()),
      y0      = as.numeric(out$y0$detach()$cpu()),
      y1      = as.numeric(out$y1$detach()$cpu()),
      ite     = as.numeric(out$ite$detach()$cpu())
    )
  })
}

#' Extract Average Treatment Effect (ATE) estimates
#'
#' Returns a named numeric vector of ATE estimates computed on the
#' validation split during training.
#'
#' @param object Fitted \code{counterfactual_model} object.
#' @return Named numeric vector (one entry per fitted model).
#' @seealso \code{\link{counterfactual_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # ate_counterfactual(fit)
#' }
#' @export
ate_counterfactual <- function(object) {
  stopifnot(inherits(object, "counterfactual_model"))
  object$ate
}

#' Extract Individual Treatment Effect (ITE) vectors
#'
#' Returns the ITE vector for one model on the validation split.
#'
#' @param object Fitted \code{counterfactual_model} object.
#' @param model  One of \code{"deepsynth"}, \code{"crn"}, \code{"gnet"}.
#' @return Numeric vector of ITE estimates for each validation observation.
#' @seealso \code{\link{counterfactual_model}}, \code{\link{ate_counterfactual}},
#'   \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # ite_counterfactual(fit, model = "crn")
#' }
#' @export
ite_counterfactual <- function(object,
                                model = c("deepsynth", "crn", "gnet")) {
  stopifnot(inherits(object, "counterfactual_model"))
  model <- match.arg(model)
  ites  <- object$ite_val[[model]]
  if (is.null(ites)) stop("Model not fitted: ", model, call. = FALSE)
  ites
}

#' Convenience wrapper: fit only the DeepSynth model
#'
#' Calls \code{\link{counterfactual_model}} with \code{models = "deepsynth"}.
#'
#' @inheritParams counterfactual_model
#' @return Object of class \code{counterfactual_model}.
#' @seealso \code{\link{counterfactual_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # deep_synth_model(data_mat, treatment, outcome, lag = 10L)
#' }
#' @export
deep_synth_model <- function(data, treatment, outcome, ...)
  counterfactual_model(data, treatment, outcome, models = "deepsynth", ...)

#' Convenience wrapper: fit only the CRN model
#'
#' Calls \code{\link{counterfactual_model}} with \code{models = "crn"}.
#'
#' @inheritParams counterfactual_model
#' @return Object of class \code{counterfactual_model}.
#' @seealso \code{\link{counterfactual_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # crn_model(data_mat, treatment, outcome, lag = 10L)
#' }
#' @export
crn_model <- function(data, treatment, outcome, ...)
  counterfactual_model(data, treatment, outcome, models = "crn", ...)

#' Convenience wrapper: fit only the G-Net model
#'
#' Calls \code{\link{counterfactual_model}} with \code{models = "gnet"}.
#'
#' @inheritParams counterfactual_model
#' @return Object of class \code{counterfactual_model}.
#' @seealso \code{\link{counterfactual_model}}, \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # gnet_model(data_mat, treatment, outcome, lag = 10L)
#' }
#' @export
gnet_model <- function(data, treatment, outcome, ...)
  counterfactual_model(data, treatment, outcome, models = "gnet", ...)

#' CamelCase alias: CounterfactualModel
#' @rdname counterfactual_model
#' @export
CounterfactualModel <- function(data, treatment, outcome, ...)
  counterfactual_model(data, treatment, outcome, ...)

#' CamelCase alias: DeepSynthModel
#' @rdname deep_synth_model
#' @export
DeepSynthModel <- function(data, treatment, outcome, ...)
  deep_synth_model(data, treatment, outcome, ...)

#' CamelCase alias: CRNModel
#' @rdname crn_model
#' @export
CRNModel <- function(data, treatment, outcome, ...)
  crn_model(data, treatment, outcome, ...)

#' CamelCase alias: GNetModel
#' @rdname gnet_model
#' @export
GNetModel <- function(data, treatment, outcome, ...)
  gnet_model(data, treatment, outcome, ...)
