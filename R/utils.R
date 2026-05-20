# CausalML-R: Internal utilities

#' Run lapply in parallel when n_cores > 1 (uses parallel::mclapply on Unix/mac; 1 core on Windows).
#' @param X vector to iterate over
#' @param FUN function
#' @param n_cores number of cores (1 = sequential)
#' @param ... passed to mclapply/lapply
#' @noRd
parallel_lapply <- function(X, FUN, n_cores = 1L, ...) {
  n_cores <- as.integer(n_cores)[1L]
  if (n_cores <= 1L) return(lapply(X, FUN, ...))
  parallel::mclapply(X, FUN, ..., mc.cores = min(n_cores, length(X)))
}

#' Check treatment vector
#' @param treatment numeric or factor treatment vector
#' @param control_name value indicating control
#' @noRd
check_treatment_vector <- function(treatment, control_name = 0) {
  treatment <- as.vector(treatment)
  if (!is.numeric(treatment) && !is.factor(treatment))
    stop("treatment must be numeric or factor")
  uniq <- unique(na.omit(treatment))
  if (!control_name %in% uniq)
    stop("control_name must appear in treatment")
  invisible(TRUE)
}

#' Validate numeric continuous treatment (positive variability, finite values)
#' @noRd
check_continuous_treatment <- function(treatment) {
  d <- as.numeric(as.vector(treatment))
  if (any(!is.finite(d)))
    stop("Continuous treatment must be finite numeric.")
  s <- stats::sd(d, na.rm = TRUE)
  if (!is.finite(s) || s <= sqrt(.Machine$double.eps))
    stop("Continuous treatment must have strictly positive variability.")
  invisible(TRUE)
}

#' Convert inputs to matrix/numeric for internal use
#' @noRd
convert_to_numeric <- function(X, treatment, y) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  if (inherits(treatment, "data.frame")) treatment <- treatment[[1L]]
  if (inherits(y, "data.frame")) y <- y[[1L]]
  if (is.character(treatment)) treatment <- as.factor(treatment)
  if (is.factor(treatment)) {
    treatment <- as.numeric(treatment)
  } else {
    treatment <- as.numeric(as.vector(treatment))
  }
  y <- as.numeric(as.vector(y))
  list(X = X, treatment = treatment, y = y)
}

#' Weighted variance (like Python get_weighted_variance)
#' @param x numeric vector
#' @param sample_weight numeric vector of weights (same length as x)
#' @return weighted variance
#' @noRd
get_weighted_variance <- function(x, sample_weight) {
  x <- as.numeric(x)
  sample_weight <- as.numeric(sample_weight)
  if (length(x) != length(sample_weight))
    stop("x and sample_weight must have the same length")
  avg <- stats::weighted.mean(x, sample_weight, na.rm = TRUE)
  sum(sample_weight * (x - avg)^2, na.rm = TRUE) / sum(sample_weight, na.rm = TRUE)
}

#' Regression metrics (MSE, MAE, R2) by treatment group
#' @param y observed outcome
#' @param yhat predicted outcome
#' @param w treatment indicator (0/1)
#' @return list with mse, mae, r2 for control and treated
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # regression_metrics(...)
#' }
#' @export
regression_metrics <- function(y, yhat, w) {
  w <- as.integer(w)
  mse0 <- mean((y[w == 0] - yhat[w == 0])^2)
  mse1 <- mean((y[w == 1] - yhat[w == 1])^2)
  mae0 <- mean(abs(y[w == 0] - yhat[w == 0]))
  mae1 <- mean(abs(y[w == 1] - yhat[w == 1]))
  r2_ <- function(y, yh) 1 - sum((y - yh)^2) / sum((y - mean(y))^2)
  r20 <- r2_(y[w == 0], yhat[w == 0])
  r21 <- r2_(y[w == 1], yhat[w == 1])
  list(mse = c(control = mse0, treated = mse1),
       mae = c(control = mae0, treated = mae1),
       r2 = c(control = r20, treated = r21))
}

#' Classification metrics (AUC proxy, accuracy) by treatment group
#' @param y observed binary outcome
#' @param yhat predicted probability
#' @param w treatment indicator
#' @return
#' Object returned by \code{classification_metrics}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # classification_metrics(...)
#' }
#' @export
classification_metrics <- function(y, yhat, w) {
  w <- as.integer(w)
  acc0 <- mean((yhat[w == 0] > 0.5) == y[w == 0])
  acc1 <- mean((yhat[w == 1] > 0.5) == y[w == 1])
  list(accuracy = c(control = acc0, treated = acc1))
}

# --- Uplift / Qini metrics (mirror Python causalml.metrics) ---

#' Qini curve: cumulative true treatment effect when units ordered by predicted CATE (descending)
#' @param tau true treatment effect vector
#' @param pred predicted CATE vector
#' @return cumulative sum of tau in order of pred (descending)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # qini_curve(...)
#' }
#' @export
qini_curve <- function(tau, pred) {
  ord <- order(pred, decreasing = TRUE)
  cumsum(tau[ord])
}

#' Qini score: area under Qini curve (trapezoidal)
#' @param tau true treatment effect vector
#' @param pred predicted CATE vector
#' @return scalar
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # qini_score_vec(...)
#' }
#' @export
qini_score_vec <- function(tau, pred) {
  n <- length(tau)
  if (n < 2) return(0)
  qc <- qini_curve(tau, pred)
  sum(diff(seq_len(n)) * (qc[-1] + qc[-n]) / 2)
}

#' Qini score for each model column in a results data frame (like Python qini_score)
#' @param data data.frame with outcome_col, treatment_col, treatment_effect_col, and model prediction columns
#' @param outcome_col name of outcome column
#' @param treatment_col name of treatment column (0/1)
#' @param treatment_effect_col name of true treatment effect column
#' @param model_cols optional character vector of column names to score; if NULL, all columns except outcome, treatment, treatment_effect
#' @return named vector of Qini scores (sorted descending by default in caller)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # qini_score(...)
#' }
#' @export
qini_score <- function(data, outcome_col = "outcome", treatment_col = "is_treated",
                       treatment_effect_col = "treatment_effect", model_cols = NULL) {
  skip <- c(outcome_col, treatment_col, treatment_effect_col)
  if (is.null(model_cols))
    model_cols <- setdiff(names(data), skip)
  tau <- data[[treatment_effect_col]]
  scores <- setNames(
    vapply(model_cols, function(col) qini_score_vec(tau, data[[col]]), numeric(1)),
    model_cols
  )
  scores
}

#' Plot Qini chart: cumulative true treatment effect vs population fraction (like Python plot_qini)
#' @param data data.frame with treatment_effect_col and model prediction columns
#' @param outcome_col name of outcome column (unused but for API compatibility)
#' @param treatment_col name of treatment column (unused but for API compatibility)
#' @param treatment_effect_col name of true treatment effect column
#' @param model_cols optional character vector of model columns; if NULL, inferred
#' @param figsize width and height of plot (inches, for par pin)
#' @param main title
#' @param ... passed to plot
#' @return
#' Object returned by \code{plot_qini}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # plot_qini(...)
#' }
#' @export
plot_qini <- function(data, outcome_col = "outcome", treatment_col = "is_treated",
                      treatment_effect_col = "treatment_effect", model_cols = NULL,
                      figsize = c(5, 5), main = "Qini", ...) {
  skip <- c(outcome_col, treatment_col, treatment_effect_col)
  if (is.null(model_cols))
    model_cols <- setdiff(names(data), skip)
  tau <- data[[treatment_effect_col]]
  n <- length(tau)
  frac <- (1:n) / n
  cols <- seq_along(model_cols)
  ylim_q <- range(0, qini_curve(tau, tau))
  if (!all(is.finite(ylim_q))) ylim_q <- c(0, 1)
  opar <- par(mfrow = c(1, 1), mar = c(4, 4, 2, 1))
  on.exit(par(opar))
  plot(NULL, xlim = c(0, 1), ylim = ylim_q, type = "n",
       xlab = "Population", ylab = "Qini", main = main, ...)
  for (i in seq_along(model_cols)) {
    qc <- qini_curve(tau, data[[model_cols[i]]])
    lines(frac, qc, col = cols[i], lwd = 2)
  }
  legend("bottomright", legend = model_cols, col = cols, lwd = 2, bty = "n", cex = 0.8)
  invisible(NULL)
}

#' Gain curve: cumulative sum of true tau (normalized to 1) when ordered by pred
#' @param tau true treatment effect vector
#' @param pred predicted CATE vector
#' @return normalized cumulative gain
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # gain_curve(...)
#' }
#' @export
gain_curve <- function(tau, pred) {
  pred <- as.numeric(unlist(pred))
  if (length(pred) != length(tau)) pred <- rep(pred, length.out = length(tau))
  ord <- order(pred, decreasing = TRUE)
  cs <- cumsum(tau[ord])
  if (max(cs) > 0) cs / max(cs) else cs
}

#' Get cumulative lift (Python get_cumlift). Average uplift in cumulative population when sorted by pred descending.
#' @param data data.frame with outcome_col, treatment_col, and model columns (predicted uplift)
#' @param outcome_col name of outcome column
#' @param treatment_col name of treatment column (0/1)
#' @param treatment_effect_col optional; if provided use true tau for lift
#' @return list with lift (matrix, rows = 0..n), index 0..n
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # get_cumlift(...)
#' }
#' @export
get_cumlift <- function(data, outcome_col = "outcome", treatment_col = "is_treated",
                       treatment_effect_col = NULL) {
  y <- data[[outcome_col]]
  w <- data[[treatment_col]]
  n_actual <- length(y)
  skip <- c(outcome_col, treatment_col, if (!is.null(treatment_effect_col)) treatment_effect_col)
  model_cols <- setdiff(names(data), skip)
  lift_list <- lapply(model_cols, function(col) {
    pred <- data[[col]]
    ord <- order(pred, decreasing = TRUE)
    sorted_y <- y[ord]
    sorted_w <- w[ord]
    if (!is.null(treatment_effect_col)) {
      tau_sorted <- data[[treatment_effect_col]][ord]
      lift_vec <- cumsum(tau_sorted) / (1:n_actual)
    } else {
      cumsum_tr <- cumsum(sorted_w)
      cumsum_ct <- (1:n_actual) - cumsum_tr
      cumsum_y_tr <- cumsum(sorted_y * sorted_w)
      cumsum_y_ct <- cumsum(sorted_y * (1 - sorted_w))
      lift_vec <- numeric(n_actual)
      for (i in seq_len(n_actual)) {
        if (cumsum_tr[i] > 0 && cumsum_ct[i] > 0)
          lift_vec[i] <- (cumsum_y_tr[i] / cumsum_tr[i]) - (cumsum_y_ct[i] / cumsum_ct[i])
        else
          lift_vec[i] <- if (i > 1) lift_vec[i - 1] else 0
      }
    }
    c(0, lift_vec)
  })
  lift <- do.call(cbind, lift_list)
  colnames(lift) <- model_cols
  list(lift = lift, index = 0:n_actual)
}

#' Get cumulative gain (Python get_cumgain). gain = lift * index (cumulative gain at each position).
#' @param data data.frame with outcome_col, treatment_col, and model columns
#' @param outcome_col name of outcome column
#' @param treatment_col name of treatment column
#' @param treatment_effect_col optional true treatment effect column
#' @param normalize if TRUE normalize gain so max = 1 (in absolute value)
#' @return matrix with rows 0..n, columns = model_cols
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # get_cumgain(...)
#' }
#' @export
get_cumgain <- function(data, outcome_col = "outcome", treatment_col = "is_treated",
                        treatment_effect_col = NULL, normalize = FALSE) {
  cl <- get_cumlift(data, outcome_col, treatment_col, treatment_effect_col)
  n <- nrow(cl$lift)
  idx <- 0:(n - 1)
  gain <- cl$lift * idx
  if (normalize) {
    last <- gain[n, , drop = FALSE]
    for (j in seq_len(ncol(gain)))
      gain[, j] <- gain[, j] / max(1e-10, abs(last[1, j]))
  }
  gain
}

#' Plot gain chart (Python plot_gain). Uses get_cumgain for observed uplift so curve matches Python.
#' @param data data.frame with outcome, treatment, and optionally treatment_effect columns plus model columns
#' @param outcome_col name of outcome column
#' @param treatment_col name of treatment column
#' @param treatment_effect_col if provided, plot cumulative gain of true tau; else use get_cumgain (observed)
#' @param n number of points to plot (downsample); NULL = use all
#' @param model_cols optional; if NULL use all columns except outcome, treatment, treatment_effect
#' @param main title
#' @param normalize if TRUE (when treatment_effect_col set) normalize y to 1
#' @param plot_chance_level if TRUE draw random baseline line (Python default)
#' @param ... passed to plot
#' @return
#' Object returned by \code{plot_gain}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # plot_gain(...)
#' }
#' @export
plot_gain <- function(data, outcome_col = "outcome", treatment_col = "is_treated",
                      treatment_effect_col = NULL, n = NULL,
                      model_cols = NULL, main = "Cumulative gain", normalize = FALSE,
                      plot_chance_level = TRUE, ...) {
  skip <- c(outcome_col, treatment_col, if (!is.null(treatment_effect_col)) treatment_effect_col)
  if (is.null(model_cols))
    model_cols <- setdiff(names(data), skip)
  opar <- par(mfrow = c(1, 1), mar = c(4, 4, 2, 1))
  on.exit(par(opar))
  n_actual <- nrow(data)
  y <- data[[outcome_col]]
  w <- data[[treatment_col]]
  if (!is.null(treatment_effect_col)) {
    tau <- data[[treatment_effect_col]]
    frac <- (1:n_actual) / n_actual
    plot(NULL, xlim = c(0, 1), ylim = c(0, 1), xlab = "Population", ylab = "Cumulative gain (normalized)", main = main, ...)
    cols <- seq_along(model_cols)
    for (i in seq_along(model_cols)) {
      g <- gain_curve(tau, data[[model_cols[i]]])
      if (normalize && max(g) > 0) g <- g / max(g)
      lines(frac, g, col = cols[i], lwd = 2)
    }
    legend("bottomright", legend = model_cols, col = cols, lwd = 2, bty = "n", cex = 0.8)
  } else {
    gain <- get_cumgain(data, outcome_col, treatment_col, treatment_effect_col, normalize = FALSE)
    n_pts <- nrow(gain)
    frac <- (0:(n_pts - 1)) / (n_pts - 1)
    if (!is.null(n) && n < n_pts) {
      idx <- unique(round(seq(1, n_pts, length.out = n)))
      frac <- (idx - 1) / (n_pts - 1)
      gain <- gain[idx, , drop = FALSE]
    }
    ylim <- range(0, gain, na.rm = TRUE)
    plot(NULL, xlim = c(0, 1), ylim = ylim, type = "n",
         xlab = "Population", ylab = "Cumulative gain", main = main, ...)
    cols <- seq_along(model_cols)
    for (i in seq_along(model_cols)) {
      lines(frac, gain[, model_cols[i]], col = cols[i], lwd = 2)
    }
    if (plot_chance_level && nrow(gain) > 1) {
      end_gain <- gain[nrow(gain), 1]
      lines(c(0, 1), c(0, end_gain), col = "black", lty = 2)
      legend("topleft", legend = c(model_cols, "Random"), col = c(cols, "black"), lwd = c(rep(2, length(model_cols)), 1), lty = c(rep(1, length(model_cols)), 2), bty = "n", cex = 0.8)
    } else {
      legend("topleft", legend = model_cols, col = cols, lwd = 2, bty = "n", cex = 0.8)
    }
  }
  invisible(NULL)
}
