# CausalML-R: Instrumental variables (2SLS, LATE)

#' Two-stage least squares (2SLS) for causal effect with instrumental variable
#' First stage: W ~ Z + X. Second stage: Y ~ W_hat + X.
#' @param Y outcome vector
#' @param W treatment/endogenous variable
#' @param Z instrument(s)
#' @param X covariates (optional)
#' @return list with coef (treatment effect), se, fitted model
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # iv_2sls(...)
#' }
#' @export
iv_2sls <- function(Y, W, Z, X = NULL) {
  Y <- as.numeric(Y)
  W <- as.numeric(W)
  Z <- as.matrix(Z)
  n <- length(Y)
  if (is.null(X)) X <- matrix(1, n, 1)
  else X <- cbind(1, as.matrix(X))
  # First stage: W = Z * gamma_z + X * gamma_x
  Omega <- cbind(Z, X)
  gamma_hat <- solve(crossprod(Omega), crossprod(Omega, W))
  W_hat <- as.vector(Omega %*% gamma_hat)
  # Second stage: Y = W_hat * alpha + X * beta
  Xi <- cbind(W_hat, X)
  theta_hat <- solve(crossprod(Xi), crossprod(Xi, Y))
  res <- Y - Xi %*% theta_hat
  sigma2 <- sum(res^2) / (n - ncol(Xi))
  vcov <- sigma2 * solve(crossprod(Xi))
  se <- sqrt(diag(vcov))
  list(coefficients = theta_hat, treatment_effect = theta_hat[1], se = se, vcov = vcov)
}

#' Local Average Treatment Effect (LATE) via IV
#' LATE = (E[Y|Z=1] - E[Y|Z=0]) / (E[W|Z=1] - E[W|Z=0])
#' @param Y outcome
#' @param W actual treatment (may differ from assignment due to noncompliance)
#' @param Z assignment (instrument)
#' @return scalar LATE estimate
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # late_iv(...)
#' }
#' @export
late_iv <- function(Y, W, Z) {
  Y <- as.numeric(Y)
  W <- as.numeric(W)
  Z <- as.numeric(Z)
  nom <- mean(Y[Z == 1]) - mean(Y[Z == 0])
  denom <- mean(W[Z == 1]) - mean(W[Z == 0])
  if (abs(denom) < 1e-8) stop("Instrument has no effect on treatment (denominator ~ 0)")
  nom / denom
}

#' Doubly Robust Instrumental Variable (DRIV) learner for conditional LATE
#' 3-fold cross-fit: fit e0(x), e1(x), m0(x), m1(x); then fit tau(X) minimizing DR IV loss.
#' @param X covariates
#' @param Y outcome
#' @param W treatment (endogenous)
#' @param Z instrument (assignment)
#' @param p_z assignment probability P(Z=1); if NULL, estimated from data
#' @param learner "ranger" or "lm" for nuisance and tau
#' @param n_fold number of splits (3 for cross-fit)
#' @return object with predict method for conditional LATE
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # driv_learner(...)
#' }
#' @export
driv_learner <- function(X, Y, W, Z, p_z = NULL, learner = "ranger", n_fold = 3) {
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  W <- as.numeric(W)
  Z <- as.numeric(Z)
  n <- length(Y)
  if (is.null(p_z)) p_z <- mean(Z)
  p_z <- max(0.01, min(0.99, p_z))
  idx <- sample(n)
  n1 <- floor(n / 3)
  n2 <- floor(2 * n / 3)
  I1 <- idx[1:n1]
  I2 <- idx[(n1 + 1):n2]
  I3 <- idx[(n2 + 1):n]
  # Stage 1: e0(x)=P(W=1|Z=0,X), e1(x)=P(W=1|Z=1,X), m0(x)=E[Y|Z=0,X], m1(x)=E[Y|Z=1,X]
  # Using Z as group: e_z(x) = P(W=1|Z=z,X), m_z(x) = E[Y|Z=z,X]
  fit_nuisance <- function(X_tr, W_tr, Y_tr, Z_tr, X_te) {
    df_w0 <- as.data.frame(X_tr[Z_tr == 0, , drop = FALSE])
    df_w0$W <- W_tr[Z_tr == 0]
    df_w1 <- as.data.frame(X_tr[Z_tr == 1, , drop = FALSE])
    df_w1$W <- W_tr[Z_tr == 1]
    df_y0 <- as.data.frame(X_tr[Z_tr == 0, , drop = FALSE])
    df_y0$Y <- Y_tr[Z_tr == 0]
    df_y1 <- as.data.frame(X_tr[Z_tr == 1, , drop = FALSE])
    df_y1$Y <- Y_tr[Z_tr == 1]
    if (learner == "ranger") {
      e0 <- ranger::ranger(W ~ ., data = df_w0, probability = TRUE)
      e1 <- ranger::ranger(W ~ ., data = df_w1, probability = TRUE)
      m0 <- ranger::ranger(Y ~ ., data = df_y0)
      m1 <- ranger::ranger(Y ~ ., data = df_y1)
      # P(W=1): ranger probability matrix can have 1 column if only one class in training
      prob0 <- predict(e0, data = as.data.frame(X_te))$predictions
      prob1 <- predict(e1, data = as.data.frame(X_te))$predictions
      e0_te <- if (is.matrix(prob0) && ncol(prob0) >= 2) prob0[, 2] else (1 - drop(prob0))
      e1_te <- if (is.matrix(prob1) && ncol(prob1) >= 2) prob1[, 2] else (1 - drop(prob1))
      m0_te <- predict(m0, data = as.data.frame(X_te))$predictions
      m1_te <- predict(m1, data = as.data.frame(X_te))$predictions
    } else {
      e0 <- stats::glm(W ~ ., data = df_w0, family = stats::binomial)
      e1 <- stats::glm(W ~ ., data = df_w1, family = stats::binomial)
      m0 <- stats::lm(Y ~ ., data = df_y0)
      m1 <- stats::lm(Y ~ ., data = df_y1)
      e0_te <- predict(e0, newdata = as.data.frame(X_te), type = "response")
      e1_te <- predict(e1, newdata = as.data.frame(X_te), type = "response")
      m0_te <- predict(m0, newdata = as.data.frame(X_te))
      m1_te <- predict(m1, newdata = as.data.frame(X_te))
    }
    list(e0 = e0_te, e1 = e1_te, m0 = m0_te, m1 = m1_te)
  }
  # Pseudo-outcome for tau(X): (m1-m0 + Z(Y-m1)/p_z - (1-Z)(Y-m0)/(1-p_z)) - (e1-e0 + ...)*tau
  # So target = (m1-m0 + Z(Y-m1)/p_z - (1-Z)(Y-m0)/(1-p_z)) / (e1-e0 + Z(W-e1)/p_z - (1-Z)(W-e0)/(1-p_z))
  # and fit tau(X) by regressing target on X (or minimizing squared error).
  nuis1 <- fit_nuisance(X[I1, , drop = FALSE], W[I1], Y[I1], Z[I1], X[I3, , drop = FALSE])
  Y3 <- Y[I3]; W3 <- W[I3]; Z3 <- Z[I3]
  num <- (nuis1$m1 - nuis1$m0) + Z3 * (Y3 - nuis1$m1) / p_z - (1 - Z3) * (Y3 - nuis1$m0) / (1 - p_z)
  denom <- (nuis1$e1 - nuis1$e0) + Z3 * (W3 - nuis1$e1) / p_z - (1 - Z3) * (W3 - nuis1$e0) / (1 - p_z)
  denom <- pmax(0.01, abs(denom)) * sign(denom)
  pseudo <- num / denom
  df_tau <- as.data.frame(X[I3, , drop = FALSE])
  df_tau$pseudo <- pseudo
  if (learner == "ranger") {
    model_tau <- ranger::ranger(pseudo ~ ., data = df_tau)
  } else {
    model_tau <- stats::lm(pseudo ~ ., data = df_tau)
  }
  structure(list(model_tau = model_tau, learner = learner, p_z = p_z, X_names = colnames(X)),
            class = "driv_learner")
}

#' Predict conditional LATE from DRIV learner
#' @return
#' Object returned by \code{predict.driv_learner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.driv_learner(...)
#' }
#' @export
predict.driv_learner <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (object$learner == "ranger") {
    predict(object$model_tau, data = as.data.frame(newdata))$predictions
  } else {
    as.vector(predict(object$model_tau, newdata = as.data.frame(newdata)))
  }
}
