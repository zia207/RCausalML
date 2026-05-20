# MASS is in Imports (mvrnorm)

#' Generate synthetic data similar to causalml.dataset.synthetic_data(mode=2)
#'
#' @param mode Integer. \code{1} = difficult nuisance + easy treatment (see \code{\link{simulate_nuisance_and_easy_treatment}}); \code{2} = nonlinear baseline + strong heterogeneity (MVN covariates).
#' @param n Integer. Sample size
#' @param p Integer. Number of covariates (mode 1 requires \code{p >= 5}; mode 2 requires \code{p >= 11})
#' @param sigma Numeric. Noise standard deviation for outcome
#'
#' @return A named list with:
#'   y: outcome vector (n)
#'   X: covariate matrix (n × p)
#'   w: treatment vector (0/1) (n)
#'   tau: true individual treatment effect (n)
#'   b: baseline function value E[Y| X, W=0] (n)
#'   e: propensity score P(W=1 | X) (n)
#'
synthetic_data <- function(mode = 2, n = 15000, p = 20, sigma = 5.5) {
  if (mode == 1L) {
    if (p < 5L)
      stop("synthetic_data(mode = 1) requires p >= 5 (baseline and tau use X[,1] through X[,5]).")
    return(simulate_nuisance_and_easy_treatment(n = n, p = p, sigma = sigma, adj = 0))
  }

  # Covariates ~ multivariate normal, correlated
  mu <- rep(0, p)
  Sigma <- 0.5 * diag(p) + 0.5 * matrix(1, p, p)   # mild correlation
  X <- MASS::mvrnorm(n, mu = mu, Sigma = Sigma)
  
  # Helper: logistic sigmoid
  sigmoid <- function(z) 1 / (1 + exp(-z))
  
  if (mode == 2) {
    if (p < 11L) {
      stop("synthetic_data(mode = 2) requires p >= 11 (baseline and tau use X[,1] through X[,11]).")
    }
    # Baseline function b(x) — quite nonlinear / wiggly
    b <- 5 * sin(pi * X[,1] * X[,2]) + 
         4 * (X[,3] - 0.5)^2 + 
         3 * X[,4] + 
         2 * X[,5] * X[,6] +
         X[,7] - 0.5
    
    # True treatment effect tau(x) — strong heterogeneity, interacts with many variables
    tau <- (0.5 * X[,1] + X[,2] + 0.8 * X[,3] - 1.2 * X[,4] +
            0.6 * X[,5] - X[,6] + 1.5 * (X[,7] > 0) +
            2 * sin(2 * pi * X[,8]) +
            1.2 * (X[,9] - 0.5)^2 +
            0.7 * X[,10] * X[,11]) / 3
    
    # Propensity score e(x) — moderate variation
    e_raw <- 0.15 + 0.7 * sigmoid(1.5 * (X[,1] + X[,2] - X[,3] + 0.5 * X[,4] - X[,5]))
    e <- pmax(pmin(e_raw, 0.95), 0.05)   # keep away from 0/1
    
  } else {
    stop("synthetic_data: use mode = 1 or mode = 2.")
  }
  
  # Treatment assignment (randomized given X)
  w <- rbinom(n, size = 1, prob = e)
  
  # Outcome
  y <- b + tau * w + rnorm(n, mean = 0, sd = sigma)
  
  return(list(
    y   = y,
    X   = X,
    w   = w,
    tau = tau,
    b   = b,
    e   = e
  ))
}


# ────────────────────────────────────────────────
#   Example usage 

set.seed(123)

dat <- synthetic_data(mode = 2, n = 15000, p = 20, sigma = 5.5)

y   <- dat$y
X   <- dat$X
w   <- dat$w
tau <- dat$tau
b   <- dat$b
e   <- dat$e

# Build the data frame (base R)
feature_names <- paste0("feature_", seq_len(ncol(X)))

df <- as.data.frame(X)
colnames(df) <- feature_names

df$outcome         <- y
df$treatment       <- w
df$treatment_effect <- tau

# Quick check
dim(df)                     # should be 15000 × (20 + 3)
summary(df$treatment_effect) # should show strong spread
table(df$treatment)         # roughly balanced but not exactly 50/50