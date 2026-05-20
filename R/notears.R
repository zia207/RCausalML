###############################################################
# NOTEARS IMPLEMENTATION IN R
# Converted from notears-master/notears/notears.py
# Linear NOTEARS (pure R): L-BFGS-B + augmented Lagrangian
# Nonlinear NOTEARS (torch for R): MLP + Sobolev, dual ascent
# Includes utilities, SEM simulation, accuracy metrics, demos
###############################################################

# Dependencies: expm, igraph, torch (for nonlinear only).
# Nonlinear NOTEARS: h_func() uses R expm (no autograd through acyclicity);
# loss + L1/L2 on weights are still optimized.

#' Internal imports for NOTEARS utilities
#' @name notears-imports
#' @keywords internal
#' @importFrom expm expm
#' @importFrom igraph graph_from_adjacency_matrix is_dag topo_sort neighbors
#' @importFrom igraph sample_gnm sample_pa as_adj as_adjacency_matrix sample_bipartite
NULL

###############################################################
# UTILITIES
###############################################################

#' Set random seed for reproducibility (R and torch)
#' @param seed Integer seed.
#' @return
#' Object returned by \code{set_random_seed}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # set_random_seed(...)
#' }
#' @export
set_random_seed <- function(seed) {
  set.seed(seed)
  if (requireNamespace("torch", quietly = TRUE)) torch::torch_manual_seed(seed)
}

#' Check if weighted adjacency matrix W defines a DAG
#' @param W Square numeric matrix (weighted adjacency).
#' @return Logical.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # is_dag(...)
#' }
#' @export
is_dag <- function(W) {
  g <- igraph::graph_from_adjacency_matrix(
    (W != 0) * 1L, mode = "directed", diag = FALSE)
  igraph::is_dag(g)
}

#' Simulate random DAG with expected number of edges
#' @param d Number of nodes.
#' @param s0 Expected number of edges.
#' @param graph_type "ER" (Erdos-Renyi), "SF" (scale-free).
#' @return Binary adjacency matrix (d x d).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # simulate_dag(...)
#' }
#' @export
simulate_dag <- function(d, s0, graph_type = "ER") {
  if (graph_type == "ER") {
    prob <- s0 / (d * (d - 1) / 2)
    B_low <- matrix(0L, d, d)
    for (i in 2:d)
      for (j in 1:(i - 1))
        if (runif(1) < prob) B_low[i, j] <- 1L
    perm <- sample(d)
    B_perm <- B_low[perm, ][, perm]
    return(B_perm)
  } else if (graph_type == "SF") {
    g <- igraph::sample_pa(d, directed = TRUE)
    B <- as.matrix(igraph::as_adjacency_matrix(g))
    perm <- sample(d)
    B_perm <- B[perm, ][, perm]
    B_perm <- B_perm * upper.tri(B_perm)
    return(B_perm)
  } else {
    stop("Unknown graph_type: ", graph_type)
  }
}

#' Simulate SEM edge weights for a DAG
#' @param B Binary adjacency matrix (d x d).
#' @param w_ranges List of (low, high) ranges for weights.
#' @return Weighted adjacency matrix (d x d).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # simulate_parameter(...)
#' }
#' @export
simulate_parameter <- function(B, w_ranges = list(c(-2.0, -0.5), c(0.5, 2.0))) {
  d  <- nrow(B)
  nr <- length(w_ranges)
  W  <- matrix(0, d, d)
  S  <- matrix(sample(0L:(nr - 1L), d * d, replace = TRUE), d, d)
  for (i in seq_along(w_ranges)) {
    U <- matrix(runif(d * d, w_ranges[[i]][1], w_ranges[[i]][2]), d, d)
    W <- W + B * (S == (i - 1L)) * U
  }
  W
}

#' Simulate samples from linear SEM
#' @param W Weighted adjacency matrix (d x d) of a DAG.
#' @param n Number of samples.
#' @param sem_type "gauss", "exp", "uniform".
#' @param noise_scale Scalar or length-d vector; default 1.
#' @return Sample matrix (n x d).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # simulate_linear_sem(...)
#' }
#' @export
simulate_linear_sem <- function(W, n, sem_type = "gauss", noise_scale = NULL) {
  d <- nrow(W)
  if (is.null(noise_scale)) noise_scale <- rep(1.0, d)
  else if (length(noise_scale) == 1L) noise_scale <- rep(noise_scale, d)
  if (!is_dag(W)) stop("W must represent a DAG")
  g     <- igraph::graph_from_adjacency_matrix((W != 0)*1L, mode = "directed", diag = FALSE)
  order <- as.integer(igraph::topo_sort(g))
  X     <- matrix(0.0, n, d)
  for (j in order) {
    parents <- which(W[j, ] != 0)
    pa_val  <- if (length(parents) == 0) rep(0, n) else
                 as.vector(X[, parents, drop = FALSE] %*% W[j, parents])
    X[, j] <- pa_val + switch(sem_type,
      gauss   = rnorm(n,   sd  = noise_scale[j]),
      exp     = rexp(n,    rate = 1 / noise_scale[j]),
      uniform = runif(n, -noise_scale[j], noise_scale[j]),
      stop("Unknown sem_type: ", sem_type)
    )
  }
  X
}

#' Simulate samples from nonlinear SEM (mlp, mim; no GP)
#' @param B Binary adjacency matrix (d x d).
#' @param n Number of samples.
#' @param sem_type "mlp" or "mim".
#' @param noise_scale Per-variable scale; default ones.
#' @return Sample matrix (n x d).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # simulate_nonlinear_sem(...)
#' }
#' @export
simulate_nonlinear_sem <- function(B, n, sem_type = "mlp", noise_scale = NULL) {
  if (!is_dag(B)) stop("B must be a DAG")
  d <- nrow(B)
  if (is.null(noise_scale)) noise_scale <- rep(1, d)
  g <- igraph::graph_from_adjacency_matrix(B != 0, mode = "directed")
  order <- as.integer(igraph::topo_sort(g)$name)
  X <- matrix(0, n, d)
  sigmoid <- function(z) 1 / (1 + exp(-z))
  for (j in order) {
    parents <- as.integer(igraph::neighbors(g, j, mode = "in"))
    pa_val <- if (length(parents) == 0) rep(0, n) else X[, parents, drop = FALSE]
    scale_j <- noise_scale[j]
    z <- rnorm(n, sd = scale_j)
    if (length(parents) == 0) {
      X[, j] <- z
      next
    }
    pa_size <- length(parents)
    if (sem_type == "mlp") {
      hidden <- 100
      W1 <- matrix(runif(pa_size * hidden, 0.5, 2), pa_size, hidden)
      W1[sample(pa_size * hidden, floor(pa_size * hidden / 2))] <- -W1[sample(pa_size * hidden, floor(pa_size * hidden / 2))]
      W2 <- runif(hidden, 0.5, 2)
      W2[sample(hidden, floor(hidden / 2))] <- -W2[sample(hidden, floor(hidden / 2))]
      X[, j] <- (sigmoid(pa_val %*% W1) %*% W2) + z
    } else if (sem_type == "mim") {
      w1 <- runif(pa_size, 0.5, 2)
      w1[sample(pa_size, floor(pa_size / 2))] <- -w1[sample(pa_size, floor(pa_size / 2))]
      w2 <- runif(pa_size, 0.5, 2)
      w2[sample(pa_size, floor(pa_size / 2))] <- -w2[sample(pa_size, floor(pa_size / 2))]
      w3 <- runif(pa_size, 0.5, 2)
      w3[sample(pa_size, floor(pa_size / 2))] <- -w3[sample(pa_size, floor(pa_size / 2))]
      X[, j] <- tanh(pa_val %*% w1) + cos(pa_val %*% w2) + sin(pa_val %*% w3) + z
    } else {
      stop("Only sem_type 'mlp' and 'mim' supported (no GP)")
    }
  }
  X
}

#' Accuracy metrics: estimated graph vs ground truth
#' @param B_true Ground truth binary adjacency (d x d).
#' @param B_est Estimated binary adjacency (d x d).
#' @return List with fdr, tpr, fpr, shd, nnz.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # count_accuracy(...)
#' }
#' @export
count_accuracy <- function(B_true, B_est) {
  B_true     <- (B_true != 0) * 1L
  B_est      <- (B_est != 0) * 1L
  d          <- nrow(B_true)
  pred_edges <- which(B_est   == 1)
  cond       <- which(B_true  == 1)
  cond_rev   <- which(t(B_true) == 1)
  cond_skel  <- union(cond, cond_rev)
  tp         <- intersect(pred_edges, cond)
  fp         <- setdiff(pred_edges, cond_skel)
  rev_       <- intersect(setdiff(pred_edges, cond), cond_rev)
  pred_size  <- length(pred_edges)
  cond_neg   <- 0.5 * d * (d - 1) - length(cond)
  fdr <- (length(fp) + length(rev_)) / max(pred_size, 1)
  tpr <- length(tp) / max(length(cond), 1)
  fpr <- (length(fp) + length(rev_)) / max(cond_neg,  1)
  pred_lower <- which(lower.tri(B_est  + t(B_est))  & (B_est  + t(B_est))  > 0)
  cond_lower <- which(lower.tri(B_true + t(B_true)) & (B_true + t(B_true)) > 0)
  shd <- length(setdiff(pred_lower, cond_lower)) +
         length(setdiff(cond_lower, pred_lower)) + length(rev_)
  list(fdr = fdr, tpr = tpr, fpr = fpr, shd = shd, nnz = pred_size)
}

###############################################################
# LINEAR NOTEARS (PURE R) — Augmented Lagrangian with L-BFGS-B
###############################################################

#' Linear NOTEARS: min L(W;X) + lambda1||W||_1 s.t. h(W)=0
#' @param X Data matrix (n x d).
#' @param lambda1 L1 penalty.
#' @param loss_type "l2", "logistic", "poisson".
#' @param max_iter Max dual ascent steps.
#' @param h_tol Stop if |h(W)| <= h_tol.
#' @param rho_max Stop if rho >= rho_max.
#' @param w_threshold Zero edges with |weight| < threshold.
#' @return Estimated weighted adjacency (d x d).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # notears_linear(...)
#' }
#' @export
notears_linear <- function(X, lambda1 = 0.1, loss_type = "l2",
                           max_iter = 100L, h_tol = 1e-8,
                           rho_max = 1e16, w_threshold = 0.3) {
  n <- nrow(X); d <- ncol(X)
  if (loss_type == "l2") X <- sweep(X, 2, colMeans(X))
  sigmoid <- function(z) 1 / (1 + exp(-z))
  loss_fn <- function(W) {
    M <- X %*% W
    switch(loss_type,
      l2 = {
        R <- X - M
        list(loss = 0.5 / n * sum(R^2),
             grad = -(1/n) * t(X) %*% R)
      },
      logistic = list(
        loss = (1/n) * sum(log1p(exp(M)) - X * M),
        grad = (1/n) * t(X) %*% (sigmoid(M) - X)
      ),
      poisson = {
        S <- exp(M)
        list(loss = (1/n) * sum(S - X * M),
             grad = (1/n) * t(X) %*% (S - X))
      },
      stop("Unknown loss_type: ", loss_type)
    )
  }
  h_fn <- function(W) {
    E <- expm::expm(W * W)
    list(h = sum(diag(E)) - d, grad = t(E) * W * 2)
  }
  toW  <- function(w) matrix(w[1:(d*d)] - w[(d*d+1):(2*d*d)], d, d)
  fn_g <- function(wv) {
    W  <- toW(wv)
    L  <- loss_fn(W); H <- h_fn(W)
    obj <- L$loss + 0.5 * rho * H$h^2 + alpha * H$h + lambda1 * sum(wv)
    Gs  <- L$grad + (rho * H$h + alpha) * H$grad
    list(obj = obj, grad = c(as.vector(Gs) + lambda1, -as.vector(Gs) + lambda1))
  }
  w  <- rep(0, 2 * d * d)
  lb <- rep(0, 2 * d * d); ub <- rep(Inf, 2 * d * d)
  for (i in seq_len(d)) {
    dg <- (i - 1) * d + i
    lb[dg] <- ub[dg] <- lb[d*d + dg] <- ub[d*d + dg] <- 0
  }
  rho <- 1.0; alpha <- 0.0; h_old <- Inf
  for (iter in seq_len(max_iter)) {
    repeat {
      sol <- optim(w,
                   fn = function(wv) fn_g(wv)$obj,
                   gr = function(wv) fn_g(wv)$grad,
                   method = "L-BFGS-B", lower = lb, upper = ub,
                   control = list(maxit = 1000L, factr = 1e7))
      w_new <- sol$par
      h_new <- h_fn(toW(w_new))$h
      if (h_new > 0.25 * h_old && rho < rho_max) rho <- rho * 10
      else break
    }
    w <- w_new; h_old <- h_new
    alpha <- alpha + rho * h_old
    if (abs(h_old) <= h_tol || rho >= rho_max) break
  }
  W_est <- toW(w)
  W_est[abs(W_est) < w_threshold] <- 0
  W_est
}

###############################################################
# NONLINEAR NOTEARS (R‑TORCH) — MLP and Sobolev, custom autograd for trace(expm)
###############################################################

if (requireNamespace("torch", quietly = TRUE)) {

  # Custom differentiable trace(expm(A)) via autograd
  trace_expm_fn <- torch::autograd_function(
    forward = function(ctx, A) {
      A_r <- as.matrix(A$detach()$cpu())
      E_r <- expm::expm(A_r)
      f   <- sum(diag(E_r))
      E_t <- torch::torch_tensor(E_r, dtype = A$dtype, device = A$device)
      ctx$save_for_backward(E_t)
      torch::torch_tensor(f, dtype = A$dtype, device = A$device)
    },
    backward = function(ctx, grad_output) {
      E <- ctx$saved_variables[[1]]
      list(grad_output$reshape(c(1L, 1L)) * E$t())
    }
  )

  # --- LocallyConnected layer (per-node linear map) ---
  LocallyConnected <- torch::nn_module(
    "LocallyConnected",
    initialize = function(num_linear, in_features, out_features, bias = TRUE) {
      self$num_linear  <- num_linear
      self$in_features <- in_features
      self$weight <- torch::nn_parameter(
        torch::torch_randn(num_linear, in_features, out_features))
      if (bias) {
        self$bias <- torch::nn_parameter(torch::torch_zeros(num_linear, out_features))
      } else {
        self$bias <- NULL
      }
    },
    forward = function(x) {
      out <- torch::torch_matmul(x$unsqueeze(3L), self$weight$unsqueeze(1L))$squeeze(3L)
      if (!is.null(self$bias)) out <- out + self$bias
      out
    }
  )

  # --- NotearsMLP ---
  NotearsMLP <- torch::nn_module(
    "NotearsMLP",
    initialize = function(d, hidden, bias = TRUE) {
      self$d      <- d
      self$hidden <- hidden
      self$fc1_pos <- torch::nn_linear(d, d * hidden, bias = bias)
      self$fc1_neg <- torch::nn_linear(d, d * hidden, bias = bias)
      self$lc1     <- LocallyConnected(d, hidden, 1L, bias = bias)
    },
    forward = function(x) {
      n <- x$size(1L)
      z <- self$fc1_pos(x) - self$fc1_neg(x)
      z <- z$view(c(n, self$d, self$hidden))
      z <- torch::nnf_sigmoid(z)
      z <- self$lc1(z)
      z$squeeze(3L)
    },
    h_func = function() {
      w <- (self$fc1_pos$weight - self$fc1_neg$weight)$view(c(self$d, self$hidden, self$d))
      A <- torch::torch_sum(w * w, dim = 2L)$t()
      trace_expm_fn(A) - self$d
    },
    fc1_l1_reg = function() {
      torch::torch_sum(self$fc1_pos$weight + self$fc1_neg$weight)
    },
    l2_reg = function() {
      w_diff <- self$fc1_pos$weight - self$fc1_neg$weight
      reg <- torch::torch_sum(w_diff^2)
      reg <- reg + torch::torch_sum(self$lc1$weight^2)
      reg
    },
    fc1_to_adj = function() {
      w <- (self$fc1_pos$weight - self$fc1_neg$weight)$view(c(self$d, self$hidden, self$d))
      A <- torch::torch_sum(w * w, dim = 2L)$t()
      as.matrix(torch::torch_sqrt(A)$detach()$cpu())
    }
  )

  # --- NotearsSobolev ---
  NotearsSobolev <- torch::nn_module(
    "NotearsSobolev",
    initialize = function(d, k = 5L, bias = FALSE) {
      self$d <- d; self$k <- k
      self$fc1_pos <- torch::nn_linear(d * k, d, bias = bias)
      self$fc1_neg <- torch::nn_linear(d * k, d, bias = bias)
      torch::nn_init_zeros_(self$fc1_pos$weight)
      torch::nn_init_zeros_(self$fc1_neg$weight)
    },
    sobolev_basis = function(x) {
      seq_list <- vector("list", self$k)
      for (kk in seq_len(self$k)) {
        mu  <- 2 / ((2 * (kk - 1) + 1) * pi)
        psi <- mu * torch::torch_sin(x / mu)
        seq_list[[kk]] <- psi
      }
      torch::torch_stack(seq_list, dim = 3L)$view(c(x$size(1L), self$d * self$k))
    },
    forward = function(x) {
      bases <- self$sobolev_basis(x)
      self$fc1_pos(bases) - self$fc1_neg(bases)
    },
    h_func = function() {
      w <- (self$fc1_pos$weight - self$fc1_neg$weight)$view(c(self$d, self$d, self$k))
      A <- torch::torch_sum(w * w, dim = 3L)$t()
      trace_expm_fn(A) - self$d
    },
    fc1_l1_reg = function() {
      torch::torch_sum(self$fc1_pos$weight + self$fc1_neg$weight)
    },
    l2_reg = function() {
      torch::torch_sum((self$fc1_pos$weight - self$fc1_neg$weight)^2)
    },
    fc1_to_adj = function() {
      w <- (self$fc1_pos$weight - self$fc1_neg$weight)$view(c(self$d, self$d, self$k))
      A <- torch::torch_sum(w * w, dim = 3L)$t()
      as.matrix(torch::torch_sqrt(A)$detach()$cpu())
    }
  )

  # --- Nonlinear NOTEARS training (Adam + LBFGS) ---
  #' Nonlinear NOTEARS via Adam then L-BFGS (MLP or Sobolev)
  #' @name notears_nonlinear
  #' @param model NotearsMLP or NotearsSobolev (torch nn_module).
  #' @param X Data matrix (n x d).
  #' @param lambda1 L1 on fc1 weights.
  #' @param lambda2 L2 on weights.
  #' @param max_iter Max Adam iterations.
  #' @param lbfgs_iter L-BFGS fine-tuning steps.
  #' @param w_threshold Zero edges below threshold.
  #' @param lr Learning rate for Adam.
  #' @param verbose Print progress every 50 iterations.
  #' @return Estimated weighted adjacency (d x d), with attr "train_losses" if from Adam phase.
  #' @export
  notears_nonlinear <- function(model, X, lambda1 = 0.005, lambda2 = 0.01,
                                max_iter = 300L, lbfgs_iter = 10L,
                                w_threshold = 0.3, lr = 1e-2, verbose = TRUE) {
    n      <- nrow(X)
    X_t    <- torch::torch_tensor(X, dtype = torch::torch_double())
    model  <- model$to(dtype = torch::torch_double())
    opt    <- torch::optim_adam(model$parameters, lr = lr)
    train_losses <- numeric(max_iter)
    for (it in seq_len(max_iter)) {
      model$train(TRUE)
      opt$zero_grad()
      X_hat <- model(X_t)
      loss  <- 0.5 / n * torch::torch_sum((X_t - X_hat)^2)
      reg   <- lambda1 * model$fc1_l1_reg() + lambda2 * model$l2_reg()
      h     <- model$h_func()
      total <- loss + reg + 100.0 * h * h
      total$backward()
      opt$step()
      train_losses[it] <- as.numeric(loss)
      if (verbose && it %% 50L == 0L)
        message("  Iter ", it, " | loss: ", round(as.numeric(loss), 5),
                " | h: ", round(as.numeric(h), 6))
    }
    opt_lbfgs <- torch::optim_lbfgs(model$parameters, lr = 1.0, max_iter = 20L)
    closure <- function() {
      opt_lbfgs$zero_grad()
      X_hat <- model(X_t)
      loss  <- 0.5 / n * torch::torch_sum((X_t - X_hat)^2)
      reg   <- lambda1 * model$fc1_l1_reg() + lambda2 * model$l2_reg()
      h     <- model$h_func()
      total <- loss + reg + 100.0 * h * h
      total$backward()
      total
    }
    for (i in seq_len(lbfgs_iter)) opt_lbfgs$step(closure)
    W_est        <- model$fc1_to_adj()
    W_est[abs(W_est) < w_threshold] <- 0.0
    attr(W_est, "train_losses") <- train_losses
    W_est
  }
}

###############################################################
# DEMOS
###############################################################

#' Demo: linear NOTEARS on synthetic data
#' @return
#' Object returned by \code{demo_linear}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # demo_linear(...)
#' }
#' @export
demo_linear <- function() {
  set_random_seed(1)
  n <- 100
  d <- 20
  s0 <- 20
  B <- simulate_dag(d, s0)
  W <- simulate_parameter(B)
  X <- simulate_linear_sem(W, n)
  W_est <- notears_linear(X, 0.1, "l2")
  # B is row=child,col=parent; notears_linear returns row=parent,col=child → use t()
  print(count_accuracy(B, (t(W_est) != 0) + 0))
}

#' Demo: nonlinear NOTEARS (MLP) on synthetic data
#' @return
#' Object returned by \code{demo_nonlinear_mlp}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # demo_nonlinear_mlp(...)
#' }
#' @export
demo_nonlinear_mlp <- function() {
  set_random_seed(123)
  n <- 200
  d <- 5
  s0 <- 9
  B <- simulate_dag(d, s0)
  X <- simulate_nonlinear_sem(B, n, sem_type = "mim")
  model <- NotearsMLP(d = d, hidden = 10)
  W_est <- notears_nonlinear(model, X, lambda1 = 0.01, lambda2 = 0.01)
  print(count_accuracy(B, (W_est != 0) + 0))
}

#' Demo: nonlinear NOTEARS (Sobolev) on random data
#' @return
#' Object returned by \code{demo_nonlinear_sobolev}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # demo_nonlinear_sobolev(...)
#' }
#' @export
demo_nonlinear_sobolev <- function() {
  set_random_seed(1)
  n <- 200
  d <- 5
  X <- matrix(rnorm(n * d), n, d)
  model <- NotearsSobolev(d = d, k = 5)
  W_est <- notears_nonlinear(model, X)
  print(W_est)
}
