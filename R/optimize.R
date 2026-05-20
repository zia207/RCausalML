# CausalML-R: Treatment optimization (Counterfactual Value Estimator, Counterfactual Unit Selection)

#' Counterfactual Unit Selection: select units to maximize
#' beta*P(complier|X) + gamma*P(always-taker|X) + theta*P(never-taker|X) + delta*P(defier|X).
#' Requires estimates of P(favourable|X,W=0) and P(favourable|X,W=1) (e.g. from outcome model).
#' Complier: favourable iff treated. Always: favourable either way. Never: never favourable. Defier: favourable iff not treated.
#' @param X covariates (matrix or data.frame)
#' @param y_fit_control predicted P(favourable | X, W=0)
#' @param y_fit_treated predicted P(favourable | X, W=1)
#' @param beta benefit for complier (default 1)
#' @param gamma benefit for always-taker (default 0)
#' @param theta benefit for never-taker (default 0)
#' @param delta benefit for defier (default 0)
#' @return vector of unit selection scores (higher = better to treat)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # counterfactual_unit_selection(...)
#' }
#' @export
counterfactual_unit_selection <- function(X,
                                          y_fit_control,
                                          y_fit_treated,
                                          beta = 1,
                                          gamma = 0,
                                          theta = 0,
                                          delta = 0) {
  # Under consistency and binary W,Y: P(complier) ~ P(Y=1|W=1,X)-P(Y=1|W=0,X) when no defiers; etc.
  # Simplified: complier prob ~ max(0, uplift); always ~ min(p0,p1); never ~ 1-max(p0,p1); defier ~ max(0, p0-p1).
  p0 <- pmax(0, pmin(1, y_fit_control))
  p1 <- pmax(0, pmin(1, y_fit_treated))
  uplift <- p1 - p0
  p_complier <- pmax(0, uplift)   # simplified
  p_always <- pmin(p0, p1)
  p_never <- 1 - pmax(p0, p1)
  p_never <- pmax(0, p_never)
  p_defier <- pmax(0, -uplift)
  beta * p_complier + gamma * p_always + theta * p_never + delta * p_defier
}

# --- Counterfactual Value Estimator (existing) ---

#' Counterfactual value estimator: E[(v - cc_w) * Y_w - ic_w]
#' @param treatment vector of assigned treatment (group names or 0/1)
#' @param control_name name of control
#' @param treatment_names names of treatment arms
#' @param y_proba predicted probability of conversion (or outcome) under each treatment; matrix n x n_treatments or list
#' @param cate CATE estimates (optional) for value
#' @param value conversion value per unit
#' @param conversion_cost cost per conversion by treatment
#' @param impression_cost cost per impression by treatment
#' @return object with predict_best method
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # counterfactual_value_estimator(...)
#' }
#' @export
counterfactual_value_estimator <- function(treatment,
                                           control_name = "control",
                                           treatment_names,
                                           y_proba,
                                           cate = NULL,
                                           value = 1,
                                           conversion_cost = 0,
                                           impression_cost = 0) {
  if (is.vector(value) && length(value) == 1) value <- rep(value, length(treatment))
  if (is.vector(conversion_cost) && length(conversion_cost) == 1) conversion_cost <- rep(conversion_cost, length(treatment))
  if (is.vector(impression_cost) && length(impression_cost) == 1) impression_cost <- rep(impression_cost, length(treatment))
  n <- length(treatment)
  n_trt <- length(treatment_names)
  if (is.matrix(y_proba)) {
    stopifnot(nrow(y_proba) == n, ncol(y_proba) >= n_trt)
    y_proba <- y_proba[, seq_len(n_trt), drop = FALSE]
  }
  # Expected value for unit i under treatment k: (value - cc_k) * y_proba[i,k] - ic_k
  structure(list(
    treatment = treatment,
    control_name = control_name,
    treatment_names = treatment_names,
    y_proba = y_proba,
    cate = cate,
    value = value,
    conversion_cost = conversion_cost,
    impression_cost = impression_cost,
    n = n,
    n_trt = n_trt
  ), class = "counterfactual_value_estimator")
}

#' Predict best treatment for each unit
#' @return
#' Object returned by \code{predict_best_treatment}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict_best_treatment(...)
#' }
#' @export
predict_best_treatment <- function(obj, ...) {
  UseMethod("predict_best_treatment")
}

#' Predict best treatment from value estimator
#'
#' @return
#' Object returned by \code{predict_best_treatment.counterfactual_value_estimator}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict_best_treatment.counterfactual_value_estimator(...)
#' }
#' @export
predict_best_treatment.counterfactual_value_estimator <- function(obj, ...) {
  v <- obj$value
  cc <- obj$conversion_cost
  ic <- obj$impression_cost
  yp <- obj$y_proba
  # Matrix of value: (v - cc) * yp - ic for each treatment
  if (length(v) == 1) v <- rep(v, obj$n)
  if (length(cc) == 1) cc <- rep(cc, obj$n)
  if (length(ic) == 1) ic <- rep(ic, obj$n)
  val_mat <- matrix(NA_real_, obj$n, obj$n_trt)
  for (k in seq_len(obj$n_trt)) {
    vk <- if (length(v) >= obj$n) v else rep(v[1], obj$n)
    # Per-treatment costs: length n_trt → use cc[k] for column k; else per-unit or scalar
    if (length(cc) == obj$n_trt) {
      cck <- rep(cc[k], obj$n)
    } else {
      cck <- if (length(cc) >= obj$n) cc else rep(cc[1], obj$n)
    }
    if (length(ic) == obj$n_trt) {
      ick <- rep(ic[k], obj$n)
    } else {
      ick <- if (length(ic) >= obj$n) ic else rep(ic[1], obj$n)
    }
    val_mat[, k] <- (vk - cck) * yp[, k] - ick
  }
  max.col(val_mat, ties.method = "first")
}
