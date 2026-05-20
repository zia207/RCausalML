# ==============================================================================
# temporalCausaDiscovery.R
# R port of TCDF-master/TCDF.py (+ model.py, depthwise.py) — Temporal Causal
# Discovery Framework (Nauta et al.). Requires the suggested package `torch`.
# Ref: https://github.com/M-Nauta/TCDF
# ==============================================================================

# ------------------------------------------------------------------------------
# Depthwise causal TCN building blocks (depthwise.py)
# ------------------------------------------------------------------------------

if (requireNamespace("torch", quietly = TRUE)) {
TCDF_chomp1d <- torch::nn_module(
  classname = "TCDF_chomp1d",
  initialize = function(chomp_size) {
    self$chomp_size <- as.integer(chomp_size)
  },
  forward = function(x) {
    if (self$chomp_size <= 0L) return(x)
    sz <- x$size()
    n_time <- as.integer(sz[length(sz)])
    keep <- n_time - self$chomp_size
    if (keep < 1L) return(x)
    x[, , seq_len(keep), drop = FALSE]
  }
)

TCDF_first_block <- torch::nn_module(
  classname = "TCDF_first_block",
  initialize = function(target, n_inputs, n_outputs, kernel_size, stride,
                        dilation, padding) {
    self$target <- as.integer(target)
    self$conv1 <- torch::nn_conv1d(
      n_inputs, n_outputs, kernel_size,
      stride = stride, padding = padding, dilation = dilation, groups = n_outputs
    )
    self$chomp1 <- TCDF_chomp1d(padding)
    self$net <- torch::nn_sequential(self$conv1, self$chomp1)
    self$relu <- torch::nn_prelu(n_inputs)
    w <- self$conv1$weight
    torch::nn_init_normal_(w, mean = 0, std = 0.1)
  },
  forward = function(x) {
    out <- self$net(x)
    self$relu(out)
  }
)

TCDF_temporal_block <- torch::nn_module(
  classname = "TCDF_temporal_block",
  initialize = function(n_inputs, n_outputs, kernel_size, stride,
                        dilation, padding) {
    self$conv1 <- torch::nn_conv1d(
      n_inputs, n_outputs, kernel_size,
      stride = stride, padding = padding, dilation = dilation, groups = n_outputs
    )
    self$chomp1 <- TCDF_chomp1d(padding)
    self$net <- torch::nn_sequential(self$conv1, self$chomp1)
    self$relu <- torch::nn_prelu(n_inputs)
    w <- self$conv1$weight
    torch::nn_init_normal_(w, mean = 0, std = 0.1)
  },
  forward = function(x) {
    out <- self$net(x)
    self$relu(out + x)
  }
)

TCDF_last_block <- torch::nn_module(
  classname = "TCDF_last_block",
  initialize = function(n_inputs, n_outputs, kernel_size, stride,
                        dilation, padding) {
    self$conv1 <- torch::nn_conv1d(
      n_inputs, n_outputs, kernel_size,
      stride = stride, padding = padding, dilation = dilation, groups = n_outputs
    )
    self$chomp1 <- TCDF_chomp1d(padding)
    self$net <- torch::nn_sequential(self$conv1, self$chomp1)
    self$linear <- torch::nn_linear(n_inputs, n_inputs)
    w <- self$conv1$weight
    torch::nn_init_normal_(w, mean = 0, std = 0.1)
    torch::nn_init_normal_(self$linear$weight, mean = 0, std = 0.01)
  },
  forward = function(x) {
    out <- self$net(x)
    self$linear(out$transpose(2, 3) + x$transpose(2, 3))$transpose(2, 3)
  }
)

TCDF_depthwise_net <- torch::nn_module(
  classname = "TCDF_depthwise_net",
  initialize = function(target, num_inputs, num_levels, kernel_size = 2L,
                        dilation_c = 2L) {
    blocks <- list()
    for (l in seq_len(num_levels) - 1L) {
      dilation_size <- dilation_c^l
      pad <- (kernel_size - 1L) * dilation_size
      if (l == 0L) {
        blocks[[length(blocks) + 1L]] <- TCDF_first_block(
          target, num_inputs, num_inputs, kernel_size, 1L,
          dilation_size, pad
        )
      } else if (l == num_levels - 1L) {
        blocks[[length(blocks) + 1L]] <- TCDF_last_block(
          num_inputs, num_inputs, kernel_size, 1L,
          dilation_size, pad
        )
      } else {
        blocks[[length(blocks) + 1L]] <- TCDF_temporal_block(
          num_inputs, num_inputs, kernel_size, 1L,
          dilation_size, pad
        )
      }
    }
    self$network <- do.call(torch::nn_sequential, blocks)
  },
  forward = function(x) {
    self$network(x)
  }
)

# ------------------------------------------------------------------------------
# ADDSTCN (model.py)
# ------------------------------------------------------------------------------

TCDF_ADDSTCN <- torch::nn_module(
  classname = "TCDF_ADDSTCN",
  initialize = function(target, input_size, num_levels, kernel_size, cuda,
                      dilation_c) {
    self$target <- as.integer(target)
    self$dwn <- TCDF_depthwise_net(
      self$target, input_size, num_levels,
      kernel_size = kernel_size, dilation_c = dilation_c
    )
    self$pointwise <- torch::nn_conv1d(input_size, 1L, 1L)
    att <- torch::torch_ones(input_size, 1L)
    self$fs_attention <- torch::nn_parameter(att)
    torch::nn_init_normal_(self$pointwise$weight, mean = 0, std = 0.1)
  },
  forward = function(x) {
    w <- torch::nnf_softmax(self$fs_attention, dim = 1L)
    y1 <- self$dwn(x * w)
    y1 <- self$pointwise(y1)
    y1$transpose(2, 3)
  }
)

# ------------------------------------------------------------------------------
# TCDF.py — data prep, training, discovery
# ------------------------------------------------------------------------------

#' @keywords internal
TCDF_prepare_data <- function(file, target) {
  df <- utils::read.csv(file, check.names = FALSE)
  if (!target %in% names(df)) {
    stop("Column not found: ", target, call. = FALSE)
  }
  df_y <- df[, target, drop = FALSE]
  df_x <- df
  lag_col <- c(NA_real_, df_y[[1]][-nrow(df_y)])
  lag_col[is.na(lag_col)] <- 0
  df_x[[target]] <- lag_col
  xm <- t(as.matrix(df_x))
  ym <- t(as.matrix(df_y))
  storage.mode(xm) <- "double"
  storage.mode(ym) <- "double"
  x <- torch::torch_tensor(xm, dtype = torch::torch_float())
  y <- torch::torch_tensor(ym, dtype = torch::torch_float())
  list(x = x, y = y)
}

#' @keywords internal
TCDF_train_one_epoch <- function(epoch, traindata, traintarget, model, optimizer,
                                 log_interval, epochs) {
  model$train()
  x <- traindata[1:1, , , drop = FALSE]
  y <- traintarget[1:1, , , drop = FALSE]
  optimizer$zero_grad()
  epochpercentage <- (epoch / as.double(epochs)) * 100
  output <- model(x)
  attentionscores <- model$fs_attention
  loss <- torch::nnf_mse_loss(output, y)
  loss$backward()
  optimizer$step()
  if (epoch %% log_interval == 0L || epoch %% epochs == 0L || epoch == 1L) {
    message(sprintf(
      "Epoch: %2d [%4.0f%%] \tLoss: %.6f",
      as.integer(epoch), epochpercentage, as.numeric(loss)
    ))
  }
  list(attentionscores = attentionscores, loss = loss)
}

#' @keywords internal
TCDF_optimizer <- function(name, parameters, lr) {
  switch(
    name,
    "Adam" = torch::optim_adam(parameters, lr = lr),
    "AdamW" = torch::optim_adamw(parameters, lr = lr),
    "RMSprop" = torch::optim_rmsprop(parameters, lr = lr),
    "SGD" = torch::optim_sgd(parameters, lr = lr),
    stop("Unknown optimizer: ", name, " (try Adam, RMSprop, SGD)", call. = FALSE)
  )
}

#' @keywords internal
TCDF_device <- function(cuda) {
  if (isTRUE(cuda) && torch::cuda_is_available()) {
    torch::torch_device("cuda")
  } else {
    torch::torch_device("cpu")
  }
}

#' Discover potential causes for one target time series (TCDF)
#'
#' R equivalent of \code{TCDF.findcauses}. Trains ADDSTCN with attention,
#' interprets attention for candidate parents, applies the permutation (PIVM)
#' check, and estimates delays from depthwise kernel weights.
#'
#' @param target Name of the target column in \code{file}.
#' @param cuda Logical; use CUDA if available.
#' @param epochs Integer, training epochs.
#' @param kernel_size Convolution kernel size.
#' @param layers Number of temporal blocks in the depthwise net.
#' @param log_interval Print every this many epochs.
#' @param lr Learning rate.
#' @param optimizername Optimizer class name: \code{"Adam"}, \code{"AdamW"},
#'   \code{"RMSprop"}, or \code{"SGD"} (Python \code{torch.optim} names).
#' @param seed Integer random seed.
#' @param dilation_c Dilation base per level.
#' @param significance Multiplier for the PIVM loss-difference threshold.
#' @param file Path to CSV (rows = time, columns = series).
#' @return A list with \code{validated} (1-based cause column indices),
#'   \code{causeswithdelay} (\code{data.frame} with \code{cause}, \code{delay}),
#'   \code{realloss}, and \code{scores} (attention weights per channel).
#' @references Nauta et al., TCDF: Temporal Causal Discovery Framework.
TCDF_find_causes <- function(target, cuda, epochs, kernel_size, layers,
                             log_interval, lr, optimizername, seed,
                             dilation_c, significance, file) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Package 'torch' is required for TCDF_find_causes()", call. = FALSE)
  }
  message("\nAnalysis started for target: ", target)
  torch::torch_manual_seed(seed)

  prep <- TCDF_prepare_data(file, target)
  X_train <- prep$x$unsqueeze(1)$contiguous()
  Y_train <- prep$y$unsqueeze(3)$contiguous()

  input_channels <- as.integer(X_train$size(2))
  df_header <- utils::read.csv(file, check.names = FALSE, nrows = 0L)
  targetidx <- match(target, names(df_header))
  if (is.na(targetidx)) stop("target not in columns", call. = FALSE)

  device <- TCDF_device(cuda)
  model <- TCDF_ADDSTCN(
    targetidx - 1L, input_channels, layers,
    kernel_size = kernel_size, cuda = cuda, dilation_c = dilation_c
  )
  model <- model$to(device = device)
  X_train <- X_train$to(device = device)
  Y_train <- Y_train$to(device = device)

  optimizer <- TCDF_optimizer(optimizername, model$parameters, lr = lr)

  out1 <- TCDF_train_one_epoch(1L, X_train, Y_train, model, optimizer,
                               log_interval, epochs)
  scores <- out1$attentionscores
  firstloss <- as.numeric(out1$loss$cpu())
  for (ep in seq_len(epochs - 1L) + 1L) {
    out1 <- TCDF_train_one_epoch(ep, X_train, Y_train, model, optimizer,
                                 log_interval, epochs)
  }
  realloss <- as.numeric(out1$loss$cpu())
  scores_vec <- as.numeric(out1$attentionscores$view(-1L)$cpu()$detach())

  s <- sort(scores_vec, decreasing = TRUE)
  indices <- order(-scores_vec)

  if (length(s) <= 5L) {
    potentials <- indices[scores_vec[indices] > 1]
  } else {
    potentials <- integer(0)
    gaps <- numeric(0)
    for (i in seq_len(length(s) - 1L)) {
      if (s[i] < 1) break
      gaps <- c(gaps, s[i] - s[i + 1L])
    }
    sortgaps <- sort(gaps, decreasing = TRUE)
    ind <- -1L
    for (i in seq_along(sortgaps)) {
      largestgap <- sortgaps[i]
      index <- match(largestgap, gaps)
      if (is.na(index)) next
      if (index < (length(s) - 1L) / 2) {
        if (index > 1L) {
          ind <- index
          break
        }
      }
    }
    if (ind < 0L) ind <- 0L
    take <- ind + 1L
    potentials <- indices[seq_len(min(take, length(indices)))]
  }
  message("Potential causes: ", paste(potentials, collapse = " "))

  validated <- potentials

  diff_improve <- firstloss - realloss
  for (idx in potentials) {
    set.seed(seed)
    shuffled <- torch::torch_clone(X_train)
    ch <- as.numeric(shuffled[1, idx, ]$cpu())
    ch <- sample(ch)
    new_ch <- torch::torch_tensor(ch, dtype = shuffled$dtype)$to(device = device)
    shuffled[1, idx, ] <- new_ch
    model$train(FALSE)
    output <- model(shuffled)
    testloss <- as.numeric(torch::nnf_mse_loss(output, Y_train)$cpu())
    testdiff <- firstloss - testloss
    if (testdiff > (diff_improve * significance)) {
      validated <- validated[validated != idx]
    }
  }

  weights <- list()
  for (layer in seq_len(layers)) {
    net_l <- model$dwn$network[[layer]]
    conv1 <- net_l$net[[1]]
    wraw <- conv1$weight$abs()
    d0 <- as.integer(wraw$size(1))
    d2 <- as.integer(wraw$size(3))
    w <- wraw$view(c(d0, d2))
    weights[[layer]] <- w
  }

  causes_rows <- list()
  for (v in validated) {
    totaldelay <- 0
    for (k in seq_along(weights)) {
      w <- weights[[k]]
      row <- w[v, ]
      rv <- as.numeric(row$cpu())
      ord <- order(rv, decreasing = TRUE)
      twolargest <- rv[ord[seq_len(min(2L, length(rv)))]]
      m <- twolargest[1]
      m2 <- if (length(twolargest) >= 2L) twolargest[2] else m
      if (m > m2) {
        idx_max <- which.max(rv)
        index_max <- length(rv) - idx_max
      } else {
        index_max <- 0L
      }
      delay <- index_max * (dilation_c^(k - 1L))
      totaldelay <- totaldelay + delay
    }
    dlay <- if (targetidx != v) totaldelay else totaldelay + 1L
    causes_rows[[length(causes_rows) + 1L]] <- data.frame(
      cause = v,
      delay = as.integer(dlay),
      stringsAsFactors = FALSE
    )
  }
  if (length(causes_rows)) {
    causeswithdelay <- do.call(rbind, causes_rows)
  } else {
    causeswithdelay <- data.frame(cause = integer(), delay = integer())
  }
  message("Validated causes: ", paste(validated, collapse = " "))

  list(
    validated = validated,
    causeswithdelay = causeswithdelay,
    realloss = realloss,
    scores = scores_vec
  )
}

# Python name mapping (this file):
#   preparedata  -> TCDF_prepare_data()
#   train        -> TCDF_train_one_epoch()
#   findcauses   -> TCDF_find_causes()
# Column indices in outputs are 1-based (R). Optimizers: Adam, AdamW, RMSprop, SGD.

}
