# CausalML-R: Synthetic data generation (mirrors causalml.dataset)
# Reference: Nie & Wager (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'

#' Synthetic data with difficult nuisance and easy treatment (Setup A)
#' @param n number of observations
#' @param p number of covariates (>= 5)
#' @param sigma standard deviation of error
#' @param adj propensity adjustment (higher -> propensity shifted toward 0)
#' @return list with y, X, w, tau, b, e
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # simulate_nuisance_and_easy_treatment(...)
#' }
#' @export
simulate_nuisance_and_easy_treatment <- function(n = 1000, p = 5, sigma = 1, adj = 0) {
  X <- matrix(stats::runif(n * p), n, p)
  b <- sin(pi * X[, 1] * X[, 2]) + 2 * (X[, 3] - 0.5)^2 + X[, 4] + 0.5 * X[, 5]
  eta <- 0.1
  e <- pmax(eta, pmin(sin(pi * X[, 1] * X[, 2]), 1 - eta))
  e <- plogis(qlogis(e) - adj)
  tau <- (X[, 1] + X[, 2]) / 2
  w <- stats::rbinom(n, 1, e)
  y <- b + (w - 0.5) * tau + sigma * stats::rnorm(n)
  list(y = y, X = X, w = w, tau = tau, b = b, e = e)
}

#' Synthetic data: randomized trial (Setup B)
#' @return
#' Object returned by \code{simulate_randomized_trial}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # simulate_randomized_trial(...)
#' }
#' @export
simulate_randomized_trial <- function(n = 1000, p = 5, sigma = 1, adj = 0) {
  X <- matrix(stats::rnorm(n * p), n, p)
  b <- pmax(0, X[, 1] + X[, 2], X[, 3]) + pmax(0, X[, 4] + X[, 5])
  e <- rep(0.5, n)
  tau <- X[, 1] + log1p(exp(X[, 2]))
  w <- stats::rbinom(n, 1, e)
  y <- b + (w - 0.5) * tau + sigma * stats::rnorm(n)
  list(y = y, X = X, w = w, tau = tau, b = b, e = e)
}

#' Synthetic data: easy propensity, difficult baseline (Setup C)
#' @return
#' Object returned by \code{simulate_easy_propensity_difficult_baseline}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # simulate_easy_propensity_difficult_baseline(...)
#' }
#' @export
simulate_easy_propensity_difficult_baseline <- function(n = 1000, p = 5, sigma = 1, adj = 0) {
  X <- matrix(stats::rnorm(n * p), n, p)
  b <- 2 * log1p(exp(X[, 1] + X[, 2] + X[, 3]))
  e <- 1 / (1 + exp(X[, 2] + X[, 3]))
  tau <- rep(1, n)
  w <- stats::rbinom(n, 1, e)
  y <- b + (w - 0.5) * tau + sigma * stats::rnorm(n)
  list(y = y, X = X, w = w, tau = tau, b = b, e = e)
}

#' Synthetic data: unrelated treatment and control (Setup D)
#' @return
#' Object returned by \code{simulate_unrelated_treatment_control}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # simulate_unrelated_treatment_control(...)
#' }
#' @export
simulate_unrelated_treatment_control <- function(n = 1000, p = 5, sigma = 1, adj = 0) {
  X <- matrix(stats::rnorm(n * p), n, p)
  b <- (pmax(0, X[, 1] + X[, 2] + X[, 3]) + pmax(0, X[, 4] + X[, 5])) / 2
  e <- 1 / (1 + exp(-X[, 1]) + exp(-X[, 2]))
  e <- plogis(qlogis(e) - adj)
  tau <- pmax(0, X[, 1] + X[, 2] + X[, 3]) - pmax(0, X[, 4] + X[, 5])
  w <- stats::rbinom(n, 1, e)
  y <- b + (w - 0.5) * tau + sigma * stats::rnorm(n)
  list(y = y, X = X, w = w, tau = tau, b = b, e = e)
}

#' Synthetic data: hidden confounder (Louizos et al.)
#' @return
#' Object returned by \code{simulate_hidden_confounder}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # simulate_hidden_confounder(...)
#' }
#' @export
simulate_hidden_confounder <- function(n = 10000, p = 5, sigma = 1, adj = 0) {
  z <- stats::rbinom(n, 1, 0.5)
  X <- vapply(seq_len(p), function(j) stats::rnorm(n, mean = z, sd = 5 * z + 3 * (1 - z)), numeric(n))
  if (p == 1) X <- matrix(X, ncol = 1)
  e <- 0.75 * z + 0.25 * (1 - z)
  w <- stats::rbinom(n, 1, e)
  b <- plogis(3 * (z + 2 * (2 * w - 2)))
  y <- stats::rbinom(n, 1, b)
  tau <- plogis(3 * (z + 4)) - plogis(3 * (z - 4))
  list(y = y, X = X, w = w, tau = tau, b = b, e = e)
}

#' Main entry: generate synthetic data by mode
#' @param mode 1=nuisance+easy treatment, 2=RCT, 3=easy propensity, 4=unrelated groups, 5=hidden confounder
#' @param n sample size
#' @param p number of covariates
#' @param sigma error sd
#' @param adj propensity adjustment (modes 1,4)
#' @return list with y, X, w, tau, b, e (or as separate vectors if requested)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # synthetic_data(...)
#' }
#' @export
synthetic_data <- function(mode = 1, n = 1000, p = 5, sigma = 1, adj = 0) {
  catalog <- list(
    `1` = simulate_nuisance_and_easy_treatment,
    `2` = simulate_randomized_trial,
    `3` = simulate_easy_propensity_difficult_baseline,
    `4` = simulate_unrelated_treatment_control,
    `5` = simulate_hidden_confounder
  )
  if (!as.character(mode) %in% names(catalog))
    stop("Invalid mode. Use 1, 2, 3, 4, or 5.")
  catalog[[as.character(mode)]](n = n, p = p, sigma = sigma, adj = adj)
}

#' Uplift classification dataset
#' @param treatment_name character vector of treatment group names (first = control)
#' @param y_name name of outcome column
#' @param n_samples total sample size
#' @param n_classification_features number of features
#' @param n_classification_informative number of informative features
#' @param n_uplift_increase_dict named list: treatment -> number of positive uplift features
#' @param n_uplift_decrease_dict named list: treatment -> number of negative uplift features
#' @param delta_uplift_increase_dict named list: treatment -> uplift delta for increase
#' @param delta_uplift_decrease_dict named list: treatment -> uplift delta for decrease
#' @param random_seed seed
#' @return data.frame with outcome and treatment columns plus feature columns; second element is feature names
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # make_uplift_classification(...)
#' }
#' @export
make_uplift_classification <- function(treatment_name = c("control", "treatment1"),
                                       y_name = "conversion",
                                       n_samples = 10000,
                                       n_classification_features = 10,
                                       n_classification_informative = 5,
                                       n_classification_repeated = 0,
                                       n_uplift_increase_dict = list(treatment1 = 5),
                                       n_uplift_decrease_dict = list(treatment1 = 3),
                                       delta_uplift_increase_dict = list(treatment1 = 0.1),
                                       delta_uplift_decrease_dict = list(treatment1 = -0.1),
                                       random_seed = NULL) {
  if (!is.null(random_seed)) set.seed(random_seed)
  n_treatments <- length(treatment_name) - 1L
  n_informative <- n_classification_informative
  n_total_features <- n_classification_features + n_classification_repeated

  X <- matrix(stats::rnorm(n_samples * n_total_features), n_samples, n_total_features)
  colnames(X) <- paste0("X", seq_len(n_total_features))
  X_names <- colnames(X)

  # Baseline outcome (control) from first few features
  beta_base <- stats::rnorm(n_informative)
  b0 <- as.vector(X[, seq_len(n_informative), drop = FALSE] %*% beta_base)
  prob_control <- plogis(b0)

  # Treatment assignment (equal prob for simplicity)
  treatment_idx <- sample(seq_len(length(treatment_name)), n_samples, replace = TRUE)
  treatment_col <- treatment_name[treatment_idx]

  # Uplift by treatment
  prob <- prob_control
  for (k in seq_len(n_treatments)) {
    trt_name <- treatment_name[k + 1]
    n_inc <- if (trt_name %in% names(n_uplift_increase_dict)) n_uplift_increase_dict[[trt_name]] else 0
    n_dec <- if (trt_name %in% names(n_uplift_decrease_dict)) n_uplift_decrease_dict[[trt_name]] else 0
    delta_inc <- if (trt_name %in% names(delta_uplift_increase_dict)) delta_uplift_increase_dict[[trt_name]] else 0
    delta_dec <- if (trt_name %in% names(delta_uplift_decrease_dict)) delta_uplift_decrease_dict[[trt_name]] else 0
    idx_inc <- seq_len(n_inc)
    idx_dec <- seq_len(n_dec) + n_informative
    uplift <- rep(0, n_samples)
    if (length(idx_inc)) uplift <- uplift + delta_inc * rowMeans(X[, idx_inc, drop = FALSE])
    if (length(idx_dec)) uplift <- uplift + delta_dec * rowMeans(X[, idx_dec, drop = FALSE])
    prob <- prob + (treatment_col == trt_name) * uplift
  }
  prob <- pmax(0.001, pmin(0.999, prob))
  y <- stats::rbinom(n_samples, 1, prob)

  df <- as.data.frame(X)
  df[[y_name]] <- y
  df[["treatment_group_key"]] <- treatment_col
  list(data = df, X_names = X_names)
}

#' Uplift regression dataset (continuous outcome)
#'
#' Mirrors \code{\link{make_uplift_classification}}: same feature and treatment
#' structure, but the outcome is Gaussian with mean equal to a linear baseline
#' plus treatment-specific uplift terms.
#'
#' @param treatment_name character vector of treatment group names (first = control)
#' @param y_name name of outcome column
#' @param n_samples total sample size
#' @param n_regression_features number of features
#' @param n_regression_informative number of features entering the baseline mean
#' @param n_regression_repeated reserved for symmetry with classification (currently unused extra columns if >0)
#' @param n_uplift_increase_dict named list: treatment -> count of positive uplift features
#' @param n_uplift_decrease_dict named list: treatment -> count of negative uplift features
#' @param delta_uplift_increase_dict named list: treatment -> coefficient scale on mean uplift
#' @param delta_uplift_decrease_dict named list: treatment -> coefficient scale (often negative)
#' @param sigma standard deviation of Gaussian noise
#' @param random_seed seed
#' @return list with \code{data} (data.frame) and \code{X_names}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # make_uplift_regression(...)
#' }
#' @export
make_uplift_regression <- function(treatment_name = c("control", "treatment1"),
                                   y_name = "outcome",
                                   n_samples = 10000,
                                   n_regression_features = 10,
                                   n_regression_informative = 5,
                                   n_regression_repeated = 0,
                                   n_uplift_increase_dict = list(treatment1 = 5),
                                   n_uplift_decrease_dict = list(treatment1 = 3),
                                   delta_uplift_increase_dict = list(treatment1 = 0.5),
                                   delta_uplift_decrease_dict = list(treatment1 = -0.3),
                                   sigma = 1,
                                   random_seed = NULL) {
  if (!is.null(random_seed)) set.seed(random_seed)
  n_treatments <- length(treatment_name) - 1L
  n_informative <- n_regression_informative
  n_total_features <- n_regression_features + n_regression_repeated

  X <- matrix(stats::rnorm(n_samples * n_total_features), n_samples, n_total_features)
  colnames(X) <- paste0("X", seq_len(n_total_features))
  X_names <- colnames(X)

  beta_base <- stats::rnorm(n_informative)
  mu <- as.vector(X[, seq_len(n_informative), drop = FALSE] %*% beta_base)

  treatment_idx <- sample(seq_len(length(treatment_name)), n_samples, replace = TRUE)
  treatment_col <- treatment_name[treatment_idx]

  for (k in seq_len(n_treatments)) {
    trt_name <- treatment_name[k + 1]
    n_inc <- if (trt_name %in% names(n_uplift_increase_dict)) n_uplift_increase_dict[[trt_name]] else 0
    n_dec <- if (trt_name %in% names(n_uplift_decrease_dict)) n_uplift_decrease_dict[[trt_name]] else 0
    delta_inc <- if (trt_name %in% names(delta_uplift_increase_dict)) delta_uplift_increase_dict[[trt_name]] else 0
    delta_dec <- if (trt_name %in% names(delta_uplift_decrease_dict)) delta_uplift_decrease_dict[[trt_name]] else 0
    idx_inc <- seq_len(n_inc)
    idx_dec <- seq_len(n_dec) + n_informative
    uplift <- rep(0, n_samples)
    if (length(idx_inc)) uplift <- uplift + delta_inc * rowMeans(X[, idx_inc, drop = FALSE])
    if (length(idx_dec)) uplift <- uplift + delta_dec * rowMeans(X[, idx_dec, drop = FALSE])
    mu <- mu + (treatment_col == trt_name) * uplift
  }

  y <- mu + sigma * stats::rnorm(n_samples)

  df <- as.data.frame(X)
  df[[y_name]] <- y
  df[["treatment_group_key"]] <- treatment_col
  list(data = df, X_names = X_names)
}
