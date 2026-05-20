# CausalML-R: Meta-learners (S, T, X, R, DR) for CATE estimation
# This file defines S/T/X/R/DR learners, fit/predict generics, and estimate_ate.
# Reference: Kunzel et al. (2019) Metalearners for estimating heterogeneous treatment effects.
# API aligned with causalml.inference.meta (BaseLearner, S/T/X/R-learners, TMLE).

#' Check that xgboost is available (for learner = "xgb")
#' @noRd
check_xgboost <- function() {
  if (!requireNamespace("xgboost", quietly = TRUE))
    stop("Package 'xgboost' is required for learner = 'xgb'. Install with install.packages('xgboost').")
}

#' @noRd
check_xgboost_dr <- check_xgboost

#' Build args for xgboost() so nrounds can be overridden via ... without duplicate argument error
#' @noRd
.xgb_reg_args <- function(data, label, ..., default_nrounds = 50L, weight = NULL) {
  dots <- list(...)
  nrounds <- if ("nrounds" %in% names(dots)) dots$nrounds else default_nrounds
  dots <- dots[setdiff(names(dots), "nrounds")]
  out <- c(list(data = data, label = label, nrounds = nrounds, verbosity = 0L, objective = "reg:squarederror"), dots)
  if (!is.null(weight)) out$weight <- weight
  out
}

#' Format propensity p into list by treatment group (like Python BaseLearner._format_p)
#' @param p vector or list of propensity scores
#' @param t_groups treatment group names/ids
#' @noRd
meta_format_p <- function(p, t_groups) {
  if (is.null(p)) return(NULL)
  t_groups <- sort(unique(t_groups))
  if (is.list(p)) {
    return(lapply(p, as.numeric))
  }
  setNames(list(as.numeric(p)), as.character(t_groups[1L]))
}

#' S-Learner: single model with treatment as feature
#' CATE(x) = mu(x,1) - mu(x,0). Aligned with causalml BaseSLearner.
#' With \code{treatment_type = "continuous"}, fits one surface \eqn{\mu(X, D)} and
#' predicts \eqn{\mu(X, d_1) - \mu(X, d_0)} using \code{dose_values} (default: 25th and
#' 75th percentiles of training \eqn{D}).
#' @param learner character "lm", "glmnet", "ranger", or "xgb"; or a list with fit/predict functions
#' @param control_name value for control group (0 or "control"); ignored when \code{treatment_type = "continuous"}
#' @param ate_alpha confidence level for ATE interval (default 0.05)
#' @param treatment_type \code{"binary"} (default) or \code{"continuous"}
#' @param dose_values length-2 numeric \eqn{(d_0, d_1)} for continuous contrast; if \code{NULL}, set at fit from training quantiles 0.25 and 0.75
#' @return
#' Object returned by \code{SLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # SLearner(...)
#' }
#' @export
SLearner <- function(learner = "ranger", control_name = 0, ate_alpha = 0.05,
                     treatment_type = c("binary", "continuous"), dose_values = NULL) {
  treatment_type <- match.arg(treatment_type)
  structure(list(learner = learner, control_name = control_name, ate_alpha = ate_alpha,
                 models = NULL, t_groups = NULL, treatment_type = treatment_type,
                 dose_values = dose_values),
            class = "SLearner")
}

#' Fit S-Learner
#' @param obj SLearner object
#' @param X covariate matrix
#' @param treatment treatment vector (0/1 or control/treatment)
#' @param y outcome vector
#' @param p optional propensity (unused in S-learner)
#' @return fitted SLearner
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.SLearner(...)
#' }
#' @export
fit.SLearner <- function(obj, X, treatment, y, p = NULL, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X; treatment <- conv$treatment; y <- conv$y
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  is_cont <- identical(obj$treatment_type, "continuous")
  if (is_cont) {
    check_continuous_treatment(treatment)
    if (is.null(obj$dose_values) || length(obj$dose_values) != 2L) {
      obj$dose_values <- as.numeric(stats::quantile(treatment, c(0.25, 0.75), na.rm = TRUE))
    } else {
      obj$dose_values <- as.numeric(obj$dose_values)[1:2]
    }
    obj$t_groups <- 1L
    D <- treatment
    X_aug <- cbind(D = D, X)
    df <- as.data.frame(X_aug)
    df$y <- y
    if (identical(obj$learner, "lm")) {
      m <- stats::lm(y ~ ., data = df)
      obj$models <- list(`1` = list(type = "lm", model = m))
    } else if (identical(obj$learner, "ranger")) {
      m <- ranger::ranger(y ~ ., data = df, ...)
      obj$models <- list(`1` = list(type = "ranger", model = m))
    } else if (identical(obj$learner, "glmnet")) {
      m <- glmnet::cv.glmnet(as.matrix(df[, -which(names(df) == "y"), drop = FALSE]), df$y, nfolds = 5)
      obj$models <- list(`1` = list(type = "glmnet", model = m))
    } else if (identical(obj$learner, "xgb")) {
      check_xgboost()
      x_mat <- as.matrix(df[, -which(names(df) == "y"), drop = FALSE])
      m <- do.call(xgboost::xgboost, .xgb_reg_args(x_mat, df$y, ..., default_nrounds = 50L))
      obj$models <- list(`1` = list(type = "xgb", model = m))
    } else {
      stop("learner must be 'lm', 'ranger', 'glmnet', or 'xgb'")
    }
    obj$X_names <- colnames(X)
    return(obj)
  }
  check_treatment_vector(treatment, obj$control_name)
  t_groups <- sort(unique(treatment[treatment != obj$control_name]))
  obj$t_groups <- t_groups
  obj$models <- list()
  for (g in t_groups) {
    mask <- (treatment == g) | (treatment == obj$control_name)
    X_filt <- X[mask, , drop = FALSE]
    y_filt <- y[mask]
    w_filt <- as.integer(treatment[mask] == g)
    X_new <- cbind(w = w_filt, X_filt)
    df <- as.data.frame(X_new)
    df$y <- y_filt
    if (identical(obj$learner, "lm")) {
      m <- stats::lm(y ~ ., data = df)
      obj$models[[as.character(g)]] <- list(type = "lm", model = m)
    } else if (identical(obj$learner, "ranger")) {
      m <- ranger::ranger(y ~ ., data = df, ...)
      obj$models[[as.character(g)]] <- list(type = "ranger", model = m)
    } else if (identical(obj$learner, "glmnet")) {
      m <- glmnet::cv.glmnet(as.matrix(df[, -which(names(df) == "y")]), df$y, nfolds = 5)
      obj$models[[as.character(g)]] <- list(type = "glmnet", model = m)
    } else if (identical(obj$learner, "xgb")) {
      check_xgboost()
      x_mat <- as.matrix(df[, -which(names(df) == "y")])
      m <- do.call(xgboost::xgboost, .xgb_reg_args(x_mat, df$y, ..., default_nrounds = 50L))
      obj$models[[as.character(g)]] <- list(type = "xgb", model = m)
    } else {
      stop("learner must be 'lm', 'ranger', 'glmnet', or 'xgb'")
    }
  }
  obj$X_names <- colnames(X)
  obj
}

#' Predict CATE from S-Learner
#' @param object fitted SLearner
#' @param newdata covariate matrix
#' @param treatment optional treatment vector (for metrics if y also provided)
#' @param y optional outcome (for metrics if treatment provided)
#' @param return_components if TRUE return list(te, yhat_cs, yhat_ts)
#' @return
#' Object returned by \code{predict.SLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.SLearner(...)
#' }
#' @export
predict.SLearner <- function(object, newdata, treatment = NULL, y = NULL,
                             return_components = FALSE, verbose = TRUE, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  x_names <- object$X_names
  if (is.null(x_names)) x_names <- paste0("X", seq_len(ncol(newdata)))
  n <- nrow(newdata)
  if (identical(object$treatment_type, "continuous")) {
    dv <- object$dose_values
    if (is.null(dv) || length(dv) != 2L) stop("Fitted continuous SLearner must have dose_values of length 2.")
    d0 <- dv[1L]; d1 <- dv[2L]
    m <- object$models[["1"]]
    if (is.null(m)) stop("Model not fitted.")
    X0 <- cbind(D = rep(d0, n), newdata)
    X1 <- cbind(D = rep(d1, n), newdata)
    colnames(X0) <- colnames(X1) <- c("D", x_names)
    if (m$type == "lm") {
      yc <- predict(m$model, newdata = as.data.frame(X0))
      yt <- predict(m$model, newdata = as.data.frame(X1))
    } else if (m$type == "ranger") {
      yc <- predict(m$model, data = as.data.frame(X0))$predictions
      yt <- predict(m$model, data = as.data.frame(X1))$predictions
    } else if (m$type == "xgb") {
      yc <- as.vector(predict(m$model, newdata = X0))
      yt <- as.vector(predict(m$model, newdata = X1))
    } else {
      yc <- as.vector(predict(m$model, newx = X0, s = "lambda.1se"))
      yt <- as.vector(predict(m$model, newx = X1, s = "lambda.1se"))
    }
    te <- matrix(yt - yc, ncol = 1L)
    if (return_components) {
      return(list(te = te, yhat_cs = setNames(list(yc), "1"), yhat_ts = setNames(list(yt), "1")))
    }
    if (verbose && !is.null(y) && !is.null(treatment)) {
      treatment <- as.numeric(treatment)
      Xobs <- cbind(D = treatment, newdata)
      colnames(Xobs) <- c("D", x_names)
      if (m$type == "lm") yhat <- predict(m$model, newdata = as.data.frame(Xobs))
      else if (m$type == "ranger") yhat <- predict(m$model, data = as.data.frame(Xobs))$predictions
      else if (m$type == "xgb") yhat <- as.vector(predict(m$model, newdata = Xobs))
      else yhat <- as.vector(predict(m$model, newx = Xobs, s = "lambda.1se"))
      message("Continuous S-Learner: MSE vs surface at observed D = ", round(mean((y - yhat)^2), 4))
    }
    return(drop(te))
  }
  n_t <- length(object$t_groups)
  te <- matrix(NA_real_, n, n_t)
  yhat_cs <- list()
  yhat_ts <- list()
  for (i in seq_along(object$t_groups)) {
    g <- object$t_groups[i]
    m <- object$models[[as.character(g)]]
    X0 <- cbind(w = 0, newdata)
    X1 <- cbind(w = 1, newdata)
    colnames(X0) <- colnames(X1) <- c("w", x_names)
    if (m$type == "lm") {
      yc <- predict(m$model, newdata = as.data.frame(X0))
      yt <- predict(m$model, newdata = as.data.frame(X1))
    } else if (m$type == "ranger") {
      yc <- predict(m$model, data = as.data.frame(X0))$predictions
      yt <- predict(m$model, data = as.data.frame(X1))$predictions
    } else if (m$type == "xgb") {
      yc <- as.vector(predict(m$model, newdata = X0))
      yt <- as.vector(predict(m$model, newdata = X1))
    } else {
      yc <- as.vector(predict(m$model, newx = X0, s = "lambda.1se"))
      yt <- as.vector(predict(m$model, newx = X1, s = "lambda.1se"))
    }
    yhat_cs[[as.character(g)]] <- yc
    yhat_ts[[as.character(g)]] <- yt
    te[, i] <- yt - yc
    if (verbose && !is.null(y) && !is.null(treatment)) {
      treatment <- as.numeric(treatment)
      mask <- (treatment == g) | (treatment == object$control_name)
      w <- as.integer(treatment[mask] == g)
      y_filt <- y[mask]
      yhat <- numeric(length(y_filt))
      yhat[w == 0] <- yc[mask][w == 0]
      yhat[w == 1] <- yt[mask][w == 1]
      msg <- regression_metrics(y_filt, yhat, w)
      message("Error metrics for group ", g, ": MSE control = ", round(msg$mse[1], 4),
              ", treated = ", round(msg$mse[2], 4))
    }
  }
  if (return_components) return(list(te = te, yhat_cs = yhat_cs, yhat_ts = yhat_ts))
  if (n_t == 1) drop(te) else te
}

#' Fit and predict S-Learner (like Python fit_predict)
#' @param obj SLearner object
#' @param return_ci if TRUE return list(te, te_lower, te_upper) via bootstrap
#' @param n_bootstraps number of bootstrap iterations (default 1000)
#' @param bootstrap_size samples per bootstrap (default 10000)
#' @param n_cores number of cores for bootstrap when return_ci=TRUE (1 = sequential; >1 uses parallel::mclapply on Unix/mac)
#' @param return_components if TRUE include yhat_cs, yhat_ts in output
#' @return
#' Object returned by \code{fit_predict.SLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit_predict.SLearner(...)
#' }
#' @export
fit_predict.SLearner <- function(obj, X, treatment, y, p = NULL,
                                 return_ci = FALSE, n_bootstraps = 1000L,
                                 bootstrap_size = 10000L, n_cores = 1L,
                                 return_components = FALSE,
                                 verbose = TRUE, ...) {
  obj <- fit(obj, X, treatment, y, p = p, ...)
  out <- predict(obj, X, treatment = treatment, y = y, return_components = return_components, verbose = verbose)
  if (return_components) {
    te <- out$te
    yhat_cs <- out$yhat_cs
    yhat_ts <- out$yhat_ts
  } else {
    te <- out
  }
  if (!return_ci) {
    if (return_components) return(list(fit = obj, te = te, yhat_cs = yhat_cs, yhat_ts = yhat_ts))
    return(te)
  }
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- nrow(X)
  size <- min(as.integer(bootstrap_size), n)
  n_boot <- as.integer(n_bootstraps)
  n_t <- length(obj$t_groups)
  t_groups_global <- obj$t_groups
  models_global <- obj$models
  boot_one <- function(b) {
    idx <- sample(n, size = size, replace = TRUE)
    obj_b <- SLearner(learner = obj$learner, control_name = obj$control_name, ate_alpha = obj$ate_alpha,
                      treatment_type = obj$treatment_type, dose_values = obj$dose_values)
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx], ...)
    predict(obj_b, X)
  }
  res_boot <- parallel_lapply(seq_len(n_boot), boot_one, n_cores = n_cores)
  te_boot <- array(NA_real_, c(n, n_t, n_boot))
  for (b in seq_len(n_boot)) te_boot[, , b] <- res_boot[[b]]
  obj$t_groups <- t_groups_global
  obj$models <- models_global
  alpha <- obj$ate_alpha
  te_lower <- apply(te_boot, c(1, 2), quantile, probs = alpha / 2)
  te_upper <- apply(te_boot, c(1, 2), quantile, probs = 1 - alpha / 2)
  if (return_components)
    list(fit = obj, te = te, te_lower = te_lower, te_upper = te_upper, yhat_cs = yhat_cs, yhat_ts = yhat_ts)
  else
    list(te = te, te_lower = te_lower, te_upper = te_upper)
}

#' T-Learner: separate models for control and treatment (aligned with causalml BaseTLearner)
#' With \code{treatment_type = "continuous"}, splits \eqn{D} at \code{dose_split} (default median)
#' into low vs high and fits the binary T-learner on that partition.
#' @param ate_alpha confidence level for ATE interval (default 0.05)
#' @param treatment_type \code{"binary"} or \code{"continuous"}
#' @param dose_split threshold for continuous \eqn{D}; observations with \eqn{D > }\code{dose_split} are "treated"
#' @return
#' Object returned by \code{TLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # TLearner(...)
#' }
#' @export
TLearner <- function(learner = "ranger", control_name = 0, ate_alpha = 0.05,
                     treatment_type = c("binary", "continuous"), dose_split = NULL) {
  treatment_type <- match.arg(treatment_type)
  structure(list(learner = learner, control_name = control_name, ate_alpha = ate_alpha,
                 model_0 = NULL, model_1 = NULL, t_groups = NULL, X_names = NULL,
                 treatment_type = treatment_type, dose_split = dose_split),
            class = "TLearner")
}

fit.TLearner <- function(obj, X, treatment, y, p = NULL, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X; treatment <- conv$treatment; y <- conv$y
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  if (identical(obj$treatment_type, "continuous")) {
    check_continuous_treatment(treatment)
    if (is.null(obj$dose_split)) obj$dose_split <- stats::median(treatment)
    obj$control_name <- 0
    treatment <- as.integer(treatment > obj$dose_split)
    if (length(unique(treatment)) < 2L)
      stop("Continuous T-Learner needs observations on both sides of dose_split; adjust dose_split.")
  } else {
    check_treatment_vector(treatment, obj$control_name)
  }
  obj$t_groups <- sort(unique(treatment[treatment != obj$control_name]))
  w <- as.integer(treatment != obj$control_name)
  df0 <- as.data.frame(X[w == 0, , drop = FALSE])
  df0$y <- y[w == 0]
  df1 <- as.data.frame(X[w == 1, , drop = FALSE])
  df1$y <- y[w == 1]
  if (identical(obj$learner, "lm")) {
    obj$model_0 <- list(type = "lm", model = stats::lm(y ~ ., data = df0))
    obj$model_1 <- list(type = "lm", model = stats::lm(y ~ ., data = df1))
  } else if (identical(obj$learner, "ranger")) {
    obj$model_0 <- list(type = "ranger", model = ranger::ranger(y ~ ., data = df0, ...))
    obj$model_1 <- list(type = "ranger", model = ranger::ranger(y ~ ., data = df1, ...))
  } else if (identical(obj$learner, "glmnet")) {
    obj$model_0 <- list(type = "glmnet", model = glmnet::cv.glmnet(as.matrix(df0[, -ncol(df0)]), df0$y, nfolds = 5))
    obj$model_1 <- list(type = "glmnet", model = glmnet::cv.glmnet(as.matrix(df1[, -ncol(df1)]), df1$y, nfolds = 5))
  } else if (identical(obj$learner, "xgb")) {
    check_xgboost()
    obj$model_0 <- list(type = "xgb", model = do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(df0[, -ncol(df0)]), df0$y, ..., default_nrounds = 50L)))
    obj$model_1 <- list(type = "xgb", model = do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(df1[, -ncol(df1)]), df1$y, ..., default_nrounds = 50L)))
  } else stop("learner must be 'lm', 'ranger', 'glmnet', or 'xgb'")
  obj$X_names <- colnames(X)
  obj
}

#' Predict CATE from T-Learner
#' @param return_components if TRUE return list(te, yhat_cs, yhat_ts)
#' @return
#' Object returned by \code{predict.TLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.TLearner(...)
#' }
#' @export
predict.TLearner <- function(object, newdata, treatment = NULL, y = NULL,
                             return_components = FALSE, verbose = TRUE, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names
  pred0 <- pred_common(object$model_0, newdata)
  pred1 <- pred_common(object$model_1, newdata)
  te <- pred1 - pred0
  if (length(object$t_groups) == 1L) te <- matrix(te, ncol = 1L)
  if (return_components) {
    yhat_cs <- setNames(list(pred0), as.character(object$t_groups[1L]))
    yhat_ts <- setNames(list(pred1), as.character(object$t_groups[1L]))
    return(list(te = te, yhat_cs = yhat_cs, yhat_ts = yhat_ts))
  }
  if (verbose && !is.null(y) && !is.null(treatment) && identical(object$treatment_type, "continuous")) {
    treatment <- as.numeric(treatment)
    w <- as.integer(treatment > object$dose_split)
    yhat <- pred0 * (1 - w) + pred1 * w
    msg <- regression_metrics(y, yhat, w)
    message("Continuous T-Learner (high vs low D): MSE control = ", round(msg$mse[1], 4),
            ", treated = ", round(msg$mse[2], 4))
  }
  if (is.matrix(te) && ncol(te) == 1L) drop(te) else te
}

#' Fit and predict T-Learner (like Python fit_predict)
#' @param n_cores number of cores for bootstrap when return_ci=TRUE (1 = sequential)
#' @return
#' Object returned by \code{fit_predict.TLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit_predict.TLearner(...)
#' }
#' @export
fit_predict.TLearner <- function(obj, X, treatment, y, p = NULL,
                                return_ci = FALSE, n_bootstraps = 1000L,
                                bootstrap_size = 10000L, n_cores = 1L,
                                return_components = FALSE,
                                verbose = TRUE, ...) {
  obj <- fit(obj, X, treatment, y, p = p, ...)
  out <- predict(obj, X, treatment = treatment, y = y, return_components = return_components, verbose = verbose)
  if (return_components) {
    te <- out$te
    yhat_cs <- out$yhat_cs
    yhat_ts <- out$yhat_ts
  } else {
    te <- out
  }
  if (!return_ci) {
    if (return_components) return(list(fit = obj, te = te, yhat_cs = yhat_cs, yhat_ts = yhat_ts))
    return(te)
  }
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- nrow(X)
  size <- min(as.integer(bootstrap_size), n)
  n_boot <- as.integer(n_bootstraps)
  n_t <- length(obj$t_groups)
  if (n_t == 0) n_t <- 1L
  t_groups_global <- obj$t_groups
  model_0_global <- obj$model_0
  model_1_global <- obj$model_1
  boot_one <- function(b) {
    idx <- sample(n, size = size, replace = TRUE)
    obj_b <- TLearner(learner = obj$learner, control_name = obj$control_name, ate_alpha = obj$ate_alpha,
                      treatment_type = obj$treatment_type, dose_split = obj$dose_split)
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx], ...)
    pred_b <- predict(obj_b, X)
    if (is.vector(pred_b)) pred_b <- matrix(pred_b, ncol = 1L)
    pred_b
  }
  res_boot <- parallel_lapply(seq_len(n_boot), boot_one, n_cores = n_cores)
  te_boot <- array(NA_real_, c(n, n_t, n_boot))
  for (b in seq_len(n_boot)) te_boot[, , b] <- res_boot[[b]]
  obj$t_groups <- t_groups_global
  obj$model_0 <- model_0_global
  obj$model_1 <- model_1_global
  alpha <- obj$ate_alpha
  te_lower <- apply(te_boot, c(1, 2), quantile, probs = alpha / 2)
  te_upper <- apply(te_boot, c(1, 2), quantile, probs = 1 - alpha / 2)
  if (return_components)
    list(fit = obj, te = te, te_lower = te_lower, te_upper = te_upper, yhat_cs = yhat_cs, yhat_ts = yhat_ts)
  else
    list(te = te, te_lower = te_lower, te_upper = te_upper)
}

pred_common <- function(m, newdata) {
  if (m$type == "lm") return(as.vector(predict(m$model, newdata = as.data.frame(newdata))))
  if (m$type == "ranger") return(predict(m$model, data = as.data.frame(newdata))$predictions)
  if (m$type == "xgb") return(as.vector(predict(m$model, newdata = as.matrix(newdata))))
  as.vector(predict(m$model, newx = newdata, s = "lambda.1se")[, 1])
}

#' X-Learner (with propensity weighting). Aligned with causalml BaseXLearner.
#' With \code{treatment_type = "continuous"}, dichotomizes \eqn{D} at \code{dose_split} (default median)
#' and fits the binary X-learner on high vs low dose.
#' @param ate_alpha confidence level for ATE interval (default 0.05)
#' @param treatment_type \code{"binary"} or \code{"continuous"}
#' @param dose_split threshold for continuous \eqn{D} (high if \eqn{D > }\code{dose_split})
#' @return
#' Object returned by \code{XLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # XLearner(...)
#' }
#' @export
XLearner <- function(learner = "ranger", control_name = 0, ate_alpha = 0.05,
                     treatment_type = c("binary", "continuous"), dose_split = NULL) {
  treatment_type <- match.arg(treatment_type)
  structure(list(learner = learner, control_name = control_name, ate_alpha = ate_alpha,
                 model_0 = NULL, model_1 = NULL, model_tau0 = NULL, model_tau1 = NULL,
                 propensity_model = NULL, propensity = NULL, vars_c = NULL, vars_t = NULL,
                 treatment_type = treatment_type, dose_split = dose_split),
            class = "XLearner")
}

fit.XLearner <- function(obj, X, treatment, y, p = NULL, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X; treatment <- conv$treatment; y <- conv$y
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  tt <- obj$treatment_type
  dose_s <- obj$dose_split
  if (identical(tt, "continuous")) {
    check_continuous_treatment(treatment)
    if (is.null(dose_s)) dose_s <- stats::median(treatment)
    obj$dose_split <- dose_s
    obj$control_name <- 0
    treatment <- as.integer(treatment > dose_s)
    if (length(unique(treatment)) < 2L)
      stop("Continuous X-Learner needs observations on both sides of dose_split; adjust dose_split.")
  }
  check_treatment_vector(treatment, obj$control_name)
  obj$t_groups <- sort(unique(treatment[treatment != obj$control_name]))
  w <- as.integer(treatment != obj$control_name)
  if (is.null(p)) {
    obj$propensity_model <- glmnet::cv.glmnet(X, w, family = "binomial", nfolds = 5)
    obj$propensity <- as.vector(predict(obj$propensity_model, newx = X, s = "lambda.1se", type = "response"))
    obj$propensity <- setNames(list(obj$propensity), as.character(obj$t_groups[1L]))
  } else {
    obj$propensity_model <- NULL
    obj$propensity <- meta_format_p(p, obj$t_groups)
    if (length(obj$t_groups) == 1L && !is.list(p)) obj$propensity <- setNames(list(as.numeric(p)), as.character(obj$t_groups[1L]))
  }
  p_vec <- obj$propensity[[as.character(obj$t_groups[1L])]]
  if (is.null(p_vec)) p_vec <- rep(0.5, length(y))
  # Stage 1: mu0, mu1
  tl <- TLearner(learner = obj$learner, control_name = obj$control_name, ate_alpha = obj$ate_alpha,
                 treatment_type = "binary", dose_split = NULL)
  obj <- fit.TLearner(tl, X, treatment, y, p, ...)
  class(obj) <- "XLearner"
  obj$treatment_type <- tt
  if (identical(tt, "continuous")) obj$dose_split <- dose_s else obj$dose_split <- NULL
  mu0 <- pred_common(obj$model_0, X)
  mu1 <- pred_common(obj$model_1, X)
  obj$vars_c <- var(y[w == 0] - mu0[w == 0])
  obj$vars_t <- var(y[w == 1] - mu1[w == 1])
  # Imputed treatment effects
  D1 <- y[w == 1] - mu0[w == 1]
  D0 <- mu1[w == 0] - y[w == 0]
  X1 <- X[w == 1, , drop = FALSE]
  X0 <- X[w == 0, , drop = FALSE]
  df1 <- as.data.frame(X1); df1$D <- D1
  df0 <- as.data.frame(X0); df0$D <- D0
  if (identical(obj$learner, "ranger")) {
    obj$model_tau1 <- list(type = "ranger", model = ranger::ranger(D ~ ., data = df1, ...))
    obj$model_tau0 <- list(type = "ranger", model = ranger::ranger(D ~ ., data = df0, ...))
  } else if (identical(obj$learner, "lm")) {
    obj$model_tau1 <- list(type = "lm", model = stats::lm(D ~ ., data = df1))
    obj$model_tau0 <- list(type = "lm", model = stats::lm(D ~ ., data = df0))
  } else if (identical(obj$learner, "xgb")) {
    check_xgboost()
    obj$model_tau1 <- list(type = "xgb", model = do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(X1), D1, ..., default_nrounds = 50L)))
    obj$model_tau0 <- list(type = "xgb", model = do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(X0), D0, ..., default_nrounds = 50L)))
  } else {
    obj$model_tau1 <- list(type = "glmnet", model = glmnet::cv.glmnet(as.matrix(X1), D1, nfolds = 5))
    obj$model_tau0 <- list(type = "glmnet", model = glmnet::cv.glmnet(as.matrix(X0), D0, nfolds = 5))
  }
  obj$p <- if (is.list(obj$propensity)) obj$propensity[[as.character(obj$t_groups[1L])]] else obj$propensity
  if (is.null(obj$p)) obj$p <- rep(0.5, length(y))
  obj
}

#' Predict CATE from X-Learner. When p is NULL and model was fit without p, uses stored propensity or 0.5.
#' @param p optional propensity at newdata; if NULL and fit stored propensity_model, not used (use 0.5 or stored propensity)
#' @param return_components if TRUE return list(te, dhat_cs, dhat_ts)
#' @return
#' Object returned by \code{predict.XLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.XLearner(...)
#' }
#' @export
predict.XLearner <- function(object, newdata, p = NULL, return_components = FALSE, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names
  dhat_c <- pred_common(object$model_tau0, newdata)
  dhat_t <- pred_common(object$model_tau1, newdata)
  if (!is.null(p)) {
    p_vec <- as.numeric(p)
  } else if (!is.null(object$propensity_model)) {
    p_vec <- as.vector(predict(object$propensity_model, newx = newdata, s = "lambda.1se", type = "response"))
  } else if (!is.null(object$p) && length(object$p) == nrow(newdata)) {
    p_vec <- object$p
  } else {
    p_vec <- rep(0.5, nrow(newdata))
  }
  if (length(p_vec) != nrow(newdata)) p_vec <- rep(mean(p_vec, na.rm = TRUE), nrow(newdata))
  p_vec <- pmax(1e-6, pmin(1 - 1e-6, p_vec))
  te <- p_vec * dhat_c + (1 - p_vec) * dhat_t
  if (return_components) return(list(te = te, dhat_cs = setNames(list(dhat_c), as.character(object$t_groups[1L])), dhat_ts = setNames(list(dhat_t), as.character(object$t_groups[1L]))))
  te
}

#' Fit and predict X-Learner (like Python fit_predict)
#' @param n_cores number of cores for bootstrap when return_ci=TRUE (1 = sequential)
#' @return
#' Object returned by \code{fit_predict.XLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit_predict.XLearner(...)
#' }
#' @export
fit_predict.XLearner <- function(obj, X, treatment, y, p = NULL,
                                 return_ci = FALSE, n_bootstraps = 1000L,
                                 bootstrap_size = 10000L, n_cores = 1L,
                                 return_components = FALSE,
                                 verbose = TRUE, ...) {
  obj <- fit(obj, X, treatment, y, p = p, ...)
  out <- predict(obj, X, p = obj$p, return_components = return_components)
  if (return_components) {
    te <- out$te
    dhat_cs <- out$dhat_cs
    dhat_ts <- out$dhat_ts
  } else {
    te <- out
  }
  if (!return_ci) {
    if (return_components) return(list(fit = obj, te = te, dhat_cs = dhat_cs, dhat_ts = dhat_ts))
    return(te)
  }
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- nrow(X)
  size <- min(as.integer(bootstrap_size), n)
  n_boot <- as.integer(n_bootstraps)
  boot_one <- function(b) {
    idx <- sample(n, size = size, replace = TRUE)
    obj_b <- XLearner(learner = obj$learner, control_name = obj$control_name, ate_alpha = obj$ate_alpha,
                      treatment_type = obj$treatment_type, dose_split = obj$dose_split)
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx], p = if (is.null(p)) NULL else p[idx], ...)
    predict(obj_b, X)
  }
  res_boot <- parallel_lapply(seq_len(n_boot), boot_one, n_cores = n_cores)
  te_boot <- matrix(NA_real_, n, n_boot)
  for (b in seq_len(n_boot)) te_boot[, b] <- res_boot[[b]]
  alpha <- obj$ate_alpha
  te_lower <- apply(te_boot, 1, quantile, probs = alpha / 2)
  te_upper <- apply(te_boot, 1, quantile, probs = 1 - alpha / 2)
  if (return_components)
    list(fit = obj, te = te, te_lower = te_lower, te_upper = te_upper, dhat_cs = dhat_cs, dhat_ts = dhat_ts)
  else
    list(te = te, te_lower = te_lower, te_upper = te_upper)
}

# --- Domain adaptation learner (EconML DomainAdaptationLearner) ---
# Ref: econml.metalearners.DomainAdaptationLearner — weighted outcome models and
# arm-specific propensity on {control, treatment_k} to reduce selection bias.

#' @noRd
.dal_fit_propensity <- function(propensity_learner, X, y01) {
  if (identical(propensity_learner, "glmnet")) {
    list(type = "glmnet", model = glmnet::cv.glmnet(as.matrix(X), y01, family = "binomial", nfolds = 5L))
  } else {
    df <- data.frame(y01 = as.integer(y01), as.data.frame(X), check.names = FALSE)
    list(type = "glm", model = stats::glm(y01 ~ ., data = df, family = stats::binomial()))
  }
}

#' @noRd
.dal_predict_propensity <- function(pm, X) {
  Xm <- as.matrix(X)
  if (pm$type == "glmnet") {
    as.vector(predict(pm$model, newx = Xm, s = "lambda.1se", type = "response"))
  } else {
    nd <- data.frame(as.data.frame(X), check.names = FALSE)
    as.vector(stats::predict(pm$model, newdata = nd, type = "response"))
  }
}

#' @noRd
.dal_fit_weighted_outcome <- function(learner, X, y, sample_weight, ...) {
  sample_weight <- as.numeric(sample_weight)
  df <- as.data.frame(X, check.names = FALSE)
  df$y <- y
  if (identical(learner, "lm")) {
    list(type = "lm", model = stats::lm(y ~ ., data = df, weights = sample_weight))
  } else if (identical(learner, "ranger")) {
    list(type = "ranger", model = ranger::ranger(y ~ ., data = df, case.weights = sample_weight, ...))
  } else if (identical(learner, "glmnet")) {
    x_mat <- as.matrix(df[, -ncol(df), drop = FALSE])
    list(type = "glmnet", model = glmnet::cv.glmnet(x_mat, df$y, weights = sample_weight, nfolds = 5L))
  } else if (identical(learner, "xgb")) {
    check_xgboost()
    x_mat <- as.matrix(df[, -ncol(df), drop = FALSE])
    list(type = "xgb", model = do.call(xgboost::xgboost, .xgb_reg_args(x_mat, df$y, ..., default_nrounds = 50L, weight = sample_weight)))
  } else {
    stop("learner must be 'lm', 'ranger', 'glmnet', or 'xgb'")
  }
}

#' @noRd
.dal_fit_outcome <- function(learner, X, y, ...) {
  df <- as.data.frame(X, check.names = FALSE)
  df$y <- y
  if (identical(learner, "lm")) {
    list(type = "lm", model = stats::lm(y ~ ., data = df))
  } else if (identical(learner, "ranger")) {
    list(type = "ranger", model = ranger::ranger(y ~ ., data = df, ...))
  } else if (identical(learner, "glmnet")) {
    x_mat <- as.matrix(df[, -ncol(df), drop = FALSE])
    list(type = "glmnet", model = glmnet::cv.glmnet(x_mat, df$y, nfolds = 5L))
  } else if (identical(learner, "xgb")) {
    check_xgboost()
    x_mat <- as.matrix(df[, -ncol(df), drop = FALSE])
    list(type = "xgb", model = do.call(xgboost::xgboost, .xgb_reg_args(x_mat, df$y, ..., default_nrounds = 50L)))
  } else {
    stop("learner must be 'lm', 'ranger', 'glmnet', or 'xgb'")
  }
}

#' Domain adaptation meta-learner (EconML \code{DomainAdaptationLearner})
#'
#' For each non-control arm, fits propensity of that arm vs control on the pooled
#' subsample, then fits weighted outcome models on controls and treated (weights
#' \eqn{e/(1-e)} and \eqn{(1-e)/e} as in EconML), imputes pseudo-effects, and fits a
#' final regression of imputed effects on \eqn{X}. Supports multiple discrete arms
#' with \code{control_name} as baseline.
#'
#' @param learner character: first-stage outcome learner (\code{lm}, \code{ranger}, \code{glmnet}, \code{xgb}); must support case/observation weights where applicable. Aligns with EconML \code{models} (weighted \eqn{\hat\mu_0}, \eqn{\hat\mu_1}).
#' @param final_learner character: learner regressing imputed pseudo-effects on \eqn{X}; default same as \code{learner}. Aligns with EconML \code{final_models} (one fitted object per non-control arm after \code{fit}).
#' @param propensity_learner \code{glmnet} (default) or \code{glm} for arm-vs-control propensity on each pair. Aligns with EconML \code{propensity_model} (choice of \eqn{\hat e(x)} estimator).
#' @param control_name control level (same convention as other meta-learners)
#' @param ate_alpha confidence level used by \code{\link{estimate_ate}}
#' @return
#' Object returned by \code{DomainAdaptationLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # DomainAdaptationLearner(...)
#' }
#' @export
DomainAdaptationLearner <- function(learner = "ranger", final_learner = NULL,
                                      propensity_learner = c("glmnet", "glm"),
                                      control_name = 0, ate_alpha = 0.05) {
  propensity_learner <- match.arg(propensity_learner)
  if (is.null(final_learner)) final_learner <- learner
  structure(list(learner = learner, final_learner = final_learner,
                 propensity_learner = propensity_learner,
                 control_name = control_name, ate_alpha = ate_alpha,
                 final_models = NULL, propensity_models = NULL,
                 t_groups = NULL, X_names = NULL),
            class = "DomainAdaptationLearner")
}

#' Fit a DomainAdaptationLearner model
#'
#' @return
#' Object returned by \code{fit.DomainAdaptationLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.DomainAdaptationLearner(...)
#' }
#' @export
fit.DomainAdaptationLearner <- function(obj, X, treatment, y, p = NULL, ...) {
  if (!is.null(p))
    warning("DomainAdaptationLearner ignores p; propensity is estimated within each {control, arm} pair.")
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  check_treatment_vector(treatment, obj$control_name)
  t_groups <- sort(unique(treatment[treatment != obj$control_name]))
  if (!length(t_groups)) stop("DomainAdaptationLearner needs at least one non-control treatment level.")
  T_num <- ifelse(treatment == obj$control_name, 0L, match(treatment, t_groups))
  if (anyNA(T_num)) stop("treatment values must be control_name or one of the non-control levels.")
  obj$t_groups <- t_groups
  obj$final_models <- vector("list", length(t_groups))
  obj$propensity_models <- vector("list", length(t_groups))
  for (i in seq_along(t_groups)) {
    idx0 <- which(T_num == 0L)
    idxg <- which(T_num == i)
    if (length(idx0) < 2L || length(idxg) < 2L)
      stop("Need at least 2 control and 2 treated units per arm for arm ", i, ".")
    X0 <- X[idx0, , drop = FALSE]
    y0 <- y[idx0]
    Xg <- X[idxg, , drop = FALSE]
    yg <- y[idxg]
    Xc <- rbind(X0, Xg)
    y01 <- c(rep(0L, length(idx0)), rep(1L, length(idxg)))
    pm <- .dal_fit_propensity(obj$propensity_learner, Xc, y01)
    obj$propensity_models[[i]] <- pm
    pscore <- .dal_predict_propensity(pm, Xc)
    pscore <- pmax(1e-6, pmin(1 - 1e-6, pscore))
    n0 <- nrow(X0)
    ng <- nrow(Xg)
    w_ctrl <- pscore[seq_len(n0)] / (1 - pscore[seq_len(n0)])
    w_trt <- (1 - pscore[n0 + seq_len(ng)]) / pscore[n0 + seq_len(ng)]
    m0 <- .dal_fit_weighted_outcome(obj$learner, X0, y0, w_ctrl, ...)
    mg <- .dal_fit_weighted_outcome(obj$learner, Xg, yg, w_trt, ...)
    imp0 <- pred_common(mg, X0) - y0
    impg <- yg - pred_common(m0, Xg)
    X_imp <- rbind(X0, Xg)
    y_imp <- c(imp0, impg)
    obj$final_models[[i]] <- .dal_fit_outcome(obj$final_learner, X_imp, y_imp, ...)
  }
  obj$X_names <- colnames(X)
  obj
}

#' Predict CATE with a fitted DomainAdaptationLearner
#'
#' @return
#' Object returned by \code{predict.DomainAdaptationLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.DomainAdaptationLearner(...)
#' }
#' @export
predict.DomainAdaptationLearner <- function(object, newdata, treatment = NULL, y = NULL,
                                            return_components = FALSE, verbose = TRUE, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names
  n <- nrow(newdata)
  n_t <- length(object$t_groups)
  te <- matrix(NA_real_, n, n_t)
  colnames(te) <- as.character(object$t_groups)
  for (i in seq_len(n_t)) {
    te[, i] <- pred_common(object$final_models[[i]], newdata)
  }
  if (return_components) {
    fm <- object$final_models
    names(fm) <- as.character(object$t_groups)
    return(list(te = te, final_models = fm))
  }
  if (ncol(te) == 1L) drop(te) else te
}

#' Fit and predict with DomainAdaptationLearner
#'
#' @return
#' Object returned by \code{fit_predict.DomainAdaptationLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit_predict.DomainAdaptationLearner(...)
#' }
#' @export
fit_predict.DomainAdaptationLearner <- function(obj, X, treatment, y, p = NULL,
                                                return_ci = FALSE, n_bootstraps = 1000L,
                                                bootstrap_size = 10000L, n_cores = 1L,
                                                return_components = FALSE,
                                                verbose = TRUE, ...) {
  obj <- fit(obj, X, treatment, y, p = p, ...)
  out <- predict(obj, X, treatment = treatment, y = y, return_components = return_components, verbose = verbose)
  if (return_components) {
    te <- out$te
  } else {
    te <- out
  }
  if (!return_ci) {
    if (return_components) return(list(fit = obj, te = te, final_models = out$final_models))
    return(te)
  }
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- nrow(X)
  size <- min(as.integer(bootstrap_size), n)
  n_boot <- as.integer(n_bootstraps)
  n_t <- length(obj$t_groups)
  if (is.vector(te)) te <- matrix(te, ncol = 1L)
  boot_one <- function(b) {
    idx <- sample(n, size = size, replace = TRUE)
    obj_b <- DomainAdaptationLearner(
      learner = obj$learner, final_learner = obj$final_learner,
      propensity_learner = obj$propensity_learner,
      control_name = obj$control_name, ate_alpha = obj$ate_alpha
    )
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx], ...)
    pred_b <- predict(obj_b, X)
    if (is.vector(pred_b)) pred_b <- matrix(pred_b, ncol = 1L)
    pred_b
  }
  res_boot <- parallel_lapply(seq_len(n_boot), boot_one, n_cores = n_cores)
  te_boot <- array(NA_real_, c(n, n_t, n_boot))
  for (b in seq_len(n_boot)) te_boot[, , b] <- res_boot[[b]]
  alpha <- obj$ate_alpha
  te_lower <- apply(te_boot, c(1, 2), quantile, probs = alpha / 2, na.rm = TRUE)
  te_upper <- apply(te_boot, c(1, 2), quantile, probs = 1 - alpha / 2, na.rm = TRUE)
  if (return_components)
    list(fit = obj, te = te, te_lower = te_lower, te_upper = te_upper, final_models = out$final_models)
  else
    list(te = te, te_lower = te_lower, te_upper = te_upper)
}

#' Estimate ATE for DomainAdaptationLearner
#'
#' @return
#' Object returned by \code{estimate_ate.DomainAdaptationLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # estimate_ate.DomainAdaptationLearner(...)
#' }
#' @export
estimate_ate.DomainAdaptationLearner <- function(obj, X, treatment, y, p = NULL, return_ci = TRUE,
                                                 bootstrap_ci = FALSE, n_bootstraps = 1000L,
                                                 bootstrap_size = 10000L, n_cores = 1L,
                                                 pretrain = FALSE, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- length(y)
  if (!pretrain && is.null(obj$final_models)) obj <- fit(obj, X, treatment, y, p = p, ...)
  te <- predict(obj, X)
  if (is.vector(te)) te <- matrix(te, ncol = 1L)
  t_groups <- obj$t_groups
  n_t <- ncol(te)
  ate <- numeric(n_t)
  ate_lb <- numeric(n_t)
  ate_ub <- numeric(n_t)
  z <- stats::qnorm(1 - obj$ate_alpha / 2)
  for (i in seq_len(n_t)) {
    ate[i] <- mean(te[, i], na.rm = TRUE)
    se <- sqrt(stats::var(te[, i], na.rm = TRUE) / n)
    if (!is.finite(se) || se <= 0) se <- sqrt(stats::var(te[, i], na.rm = TRUE) / max(n, 1L))
    ate_lb[i] <- ate[i] - z * se
    ate_ub[i] <- ate[i] + z * se
  }
  names(ate) <- names(ate_lb) <- names(ate_ub) <- as.character(t_groups)
  if (!return_ci) return(list(ate = ate))
  if (!bootstrap_ci) return(list(ate = ate, ate_lb = ate_lb, ate_ub = ate_ub))
  boot_one <- function(b) {
    idx <- sample(n, min(bootstrap_size, n), replace = TRUE)
    obj_b <- DomainAdaptationLearner(
      learner = obj$learner, final_learner = obj$final_learner,
      propensity_learner = obj$propensity_learner,
      control_name = obj$control_name, ate_alpha = obj$ate_alpha
    )
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx], ...)
    te_b <- predict(obj_b, X)
    if (!is.matrix(te_b)) te_b <- matrix(te_b, ncol = n_t)
    colMeans(te_b, na.rm = TRUE)
  }
  res_boot <- parallel_lapply(seq_len(n_bootstraps), boot_one, n_cores = n_cores)
  ate_boot <- matrix(NA_real_, n_t, n_bootstraps)
  for (b in seq_len(n_bootstraps)) ate_boot[, b] <- res_boot[[b]]
  list(
    ate = ate,
    ate_lb = apply(ate_boot, 1, quantile, probs = obj$ate_alpha / 2, na.rm = TRUE),
    ate_ub = apply(ate_boot, 1, quantile, probs = 1 - obj$ate_alpha / 2, na.rm = TRUE)
  )
}

#' R-Learner: minimize R-loss with cross-fitting. Aligned with causalml BaseRLearner.
#' With \code{treatment_type = "continuous"}, uses the Robinson partial-linear nuisance
#' \eqn{m(X)=\mathbb{E}[Y\mid X]}, \eqn{e(X)=\mathbb{E}[D\mid X]} and weighted effect
#' regression on \eqn{(Y-m)/(D-e)} (cf. \code{\link{r_learner_continuous}} for dose grids).
#' @param ate_alpha confidence level for ATE interval (default 0.05)
#' @param treatment_type \code{"binary"} or \code{"continuous"}
#' @return
#' Object returned by \code{RLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # RLearner(...)
#' }
#' @export
RLearner <- function(learner = "ranger", control_name = 0, n_fold = 5, ate_alpha = 0.05,
                     treatment_type = c("binary", "continuous")) {
  treatment_type <- match.arg(treatment_type)
  structure(list(learner = learner, control_name = control_name, n_fold = n_fold, ate_alpha = ate_alpha,
                 model_m = NULL, model_e = NULL, model_tau = NULL, t_groups = NULL,
                 vars_c = NULL, vars_t = NULL, treatment_type = treatment_type),
            class = "RLearner")
}

fit.RLearner <- function(obj, X, treatment, y, p = NULL, sample_weight = NULL, verbose = TRUE, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X; treatment <- conv$treatment; y <- conv$y
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  n <- length(y)
  if (!is.null(sample_weight)) sample_weight <- as.numeric(sample_weight)
  if (identical(obj$treatment_type, "continuous")) {
    check_continuous_treatment(treatment)
    D <- treatment
    e_hat <- if (is.null(p)) {
      numeric(n)
    } else {
      ev <- as.numeric(p)
      if (is.list(p)) ev <- as.numeric(p[[1L]])
      if (length(ev) != n) stop("For continuous R-Learner, p must be length n (E[D|X]) when supplied.")
      ev
    }
    m_hat <- numeric(n)
    n_fold <- obj$n_fold
    fold_id <- sample(rep(seq_len(n_fold), length.out = n))
    df_m <- as.data.frame(X)
    df_m$y <- y
    df_d <- as.data.frame(X)
    df_d$D <- D
    for (k in seq_len(n_fold)) {
      tr <- fold_id != k
      va <- fold_id == k
      xd_tr <- as.matrix(X[tr, , drop = FALSE])
      xd_va <- as.matrix(X[va, , drop = FALSE])
      if (identical(obj$learner, "ranger")) {
        m_fold <- ranger::ranger(y ~ ., data = df_m[tr, , drop = FALSE], ...)
        m_hat[va] <- predict(m_fold, data = df_m[va, -ncol(df_m), drop = FALSE])$predictions
        if (is.null(p)) {
          e_fold <- ranger::ranger(D ~ ., data = df_d[tr, , drop = FALSE], ...)
          e_hat[va] <- predict(e_fold, data = df_d[va, names(df_d) != "D", drop = FALSE])$predictions
        }
      } else if (identical(obj$learner, "xgb")) {
        check_xgboost()
        m_fold <- do.call(xgboost::xgboost, .xgb_reg_args(xd_tr, y[tr], ..., default_nrounds = 50L))
        m_hat[va] <- as.vector(predict(m_fold, xd_va))
        if (is.null(p)) {
          e_fold <- do.call(xgboost::xgboost, .xgb_reg_args(xd_tr, D[tr], ..., default_nrounds = 50L))
          e_hat[va] <- as.vector(predict(e_fold, xd_va))
        }
      } else if (identical(obj$learner, "glmnet")) {
        m_fold <- glmnet::cv.glmnet(xd_tr, y[tr], nfolds = 5L)
        m_hat[va] <- as.vector(predict(m_fold, newx = xd_va, s = "lambda.1se")[, 1])
        if (is.null(p)) {
          e_fold <- glmnet::cv.glmnet(xd_tr, D[tr], nfolds = 5L)
          e_hat[va] <- as.vector(predict(e_fold, newx = xd_va, s = "lambda.1se")[, 1])
        }
      } else {
        m_fold <- stats::lm(y ~ ., data = df_m[tr, , drop = FALSE])
        m_hat[va] <- predict(m_fold, newdata = df_m[va, -ncol(df_m), drop = FALSE])
        if (is.null(p)) {
          e_fold <- stats::lm(D ~ ., data = df_d[tr, , drop = FALSE])
          e_hat[va] <- predict(e_fold, newdata = df_d[va, names(df_d) != "D", drop = FALSE])
        }
      }
    }
    if (identical(obj$learner, "ranger")) {
      obj$model_m <- ranger::ranger(y ~ ., data = df_m, ...)
      obj$model_e <- ranger::ranger(D ~ ., data = df_d, ...)
    } else if (identical(obj$learner, "xgb")) {
      obj$model_m <- do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(X), y, ..., default_nrounds = 50L))
      obj$model_e <- do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(X), D, ..., default_nrounds = 50L))
    } else if (identical(obj$learner, "glmnet")) {
      obj$model_m <- glmnet::cv.glmnet(as.matrix(X), y, nfolds = 5L)
      obj$model_e <- glmnet::cv.glmnet(as.matrix(X), D, nfolds = 5L)
    } else {
      obj$model_m <- stats::lm(y ~ ., data = df_m)
      obj$model_e <- stats::lm(D ~ ., data = df_d)
    }
    residual_y <- y - m_hat
    residual_w <- D - e_hat
    if (!is.null(sample_weight)) {
      obj$vars_c <- stats::var(residual_y)
      obj$vars_t <- stats::var(residual_w)
      wt <- residual_w^2 * sample_weight
    } else {
      obj$vars_c <- stats::var(residual_y)
      obj$vars_t <- stats::var(residual_w)
      wt <- residual_w^2
    }
    wt[wt < 1e-6] <- 1e-6
    pseudo <- residual_y / residual_w
    pseudo[!is.finite(pseudo) | abs(residual_w) < 0.01] <- 0
    df_tau <- as.data.frame(X)
    df_tau$pseudo <- pseudo
    df_tau$wt <- wt
    if (identical(obj$learner, "ranger")) {
      obj$model_tau <- list(type = "ranger", model = ranger::ranger(pseudo ~ ., data = df_tau[, -ncol(df_tau)], case.weights = df_tau$wt, ...))
    } else if (identical(obj$learner, "xgb")) {
      check_xgboost()
      obj$model_tau <- list(type = "xgb", model = do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(X), pseudo, ..., default_nrounds = 50L, weight = wt)))
    } else if (identical(obj$learner, "glmnet")) {
      obj$model_tau <- list(type = "glmnet", model = glmnet::cv.glmnet(as.matrix(X), pseudo, nfolds = 5L, weights = wt))
    } else {
      obj$model_tau <- list(type = "lm", model = stats::lm(pseudo ~ ., data = df_tau[, -ncol(df_tau)], weights = wt))
    }
    obj$X_names <- colnames(X)
    obj$t_groups <- 1L
    obj$e_hat_train <- e_hat
    obj$m_hat_train <- m_hat
    return(obj)
  }
  check_treatment_vector(treatment, obj$control_name)
  obj$t_groups <- sort(unique(treatment[treatment != obj$control_name]))
  w <- as.integer(treatment != obj$control_name)
  if (is.null(p)) {
    e <- propensity_glmnet(X, w, n_fold = obj$n_fold)
  } else {
    e <- as.numeric(p)
    if (is.list(p)) e <- p[[as.character(obj$t_groups[1L])]]
  }
  e_hat <- pmax(0.01, pmin(0.99, e))
  # Out-of-fold outcome predictions (like Python cross_val_predict)
  n_fold <- obj$n_fold
  fold_id <- sample(rep(seq_len(n_fold), length.out = n))
  m_hat <- numeric(n)
  df_m <- as.data.frame(X)
  df_m$y <- y
  for (k in seq_len(n_fold)) {
    tr <- fold_id != k
    va <- fold_id == k
    if (identical(obj$learner, "ranger")) {
      m_fold <- ranger::ranger(y ~ ., data = df_m[tr, , drop = FALSE], ...)
      m_hat[va] <- predict(m_fold, data = df_m[va, -ncol(df_m), drop = FALSE])$predictions
    } else if (identical(obj$learner, "xgb")) {
      check_xgboost()
      m_fold <- do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(X[tr, , drop = FALSE]), y[tr], ..., default_nrounds = 50L))
      m_hat[va] <- as.vector(predict(m_fold, as.matrix(X[va, , drop = FALSE])))
    } else {
      m_fold <- stats::lm(y ~ ., data = df_m[tr, , drop = FALSE])
      m_hat[va] <- predict(m_fold, newdata = df_m[va, -ncol(df_m), drop = FALSE])
    }
  }
  obj$model_m <- NULL
  df_m_full <- as.data.frame(X); df_m_full$y <- y
  if (identical(obj$learner, "ranger")) obj$model_m <- ranger::ranger(y ~ ., data = df_m_full, ...)
  else if (identical(obj$learner, "xgb")) obj$model_m <- do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(X), y, ..., default_nrounds = 50L))
  else obj$model_m <- stats::lm(y ~ ., data = df_m_full)
  residual_y <- y - m_hat
  residual_w <- w - e_hat
  diff_c <- residual_y[w == 0]
  diff_t <- residual_y[w == 1]
  if (!is.null(sample_weight)) {
    obj$vars_c <- get_weighted_variance(diff_c, sample_weight[w == 0])
    obj$vars_t <- get_weighted_variance(diff_t, sample_weight[w == 1])
    wt <- residual_w^2 * sample_weight
  } else {
    obj$vars_c <- var(diff_c)
    obj$vars_t <- var(diff_t)
    wt <- residual_w^2
  }
  wt[wt < 1e-6] <- 1e-6
  pseudo <- residual_y / residual_w
  pseudo[abs(residual_w) < 0.01] <- 0
  df_tau <- as.data.frame(X)
  df_tau$pseudo <- pseudo
  df_tau$wt <- wt
  if (identical(obj$learner, "ranger")) {
    obj$model_tau <- list(type = "ranger", model = ranger::ranger(pseudo ~ ., data = df_tau[, -ncol(df_tau)], case.weights = df_tau$wt, ...))
  } else if (identical(obj$learner, "xgb")) {
    check_xgboost()
    obj$model_tau <- list(type = "xgb", model = do.call(xgboost::xgboost, .xgb_reg_args(as.matrix(X), pseudo, ..., default_nrounds = 50L, weight = wt)))
  } else {
    obj$model_tau <- list(type = "lm", model = stats::lm(pseudo ~ ., data = df_tau[, -ncol(df_tau)], weights = wt))
  }
  obj$X_names <- colnames(X)
  obj$e_hat_train <- e_hat
  obj$m_hat_train <- m_hat
  obj
}

predict.RLearner <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names
  m <- object$model_tau
  if (m$type == "ranger") return(predict(m$model, data = as.data.frame(newdata))$predictions)
  if (m$type == "xgb") return(as.vector(predict(m$model, newdata = as.matrix(newdata))))
  if (m$type == "glmnet") return(as.vector(predict(m$model, newx = newdata, s = "lambda.1se")[, 1]))
  as.vector(predict(m$model, newdata = as.data.frame(newdata)))
}

#' Fit and predict R-Learner (like Python fit_predict)
#' @param n_cores number of cores for bootstrap when return_ci=TRUE (1 = sequential)
#' @return
#' Object returned by \code{fit_predict.RLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit_predict.RLearner(...)
#' }
#' @export
fit_predict.RLearner <- function(obj, X, treatment, y, p = NULL, sample_weight = NULL,
                                return_ci = FALSE, n_bootstraps = 1000L,
                                bootstrap_size = 10000L, n_cores = 1L,
                                verbose = TRUE, ...) {
  obj <- fit(obj, X, treatment, y, p = p, sample_weight = sample_weight, verbose = verbose, ...)
  te <- predict(obj, X)
  if (!return_ci) return(te)
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- nrow(X)
  size <- min(as.integer(bootstrap_size), n)
  n_boot <- as.integer(n_bootstraps)
  boot_one <- function(b) {
    idx <- sample(n, size = size, replace = TRUE)
    obj_b <- RLearner(learner = obj$learner, control_name = obj$control_name, n_fold = obj$n_fold, ate_alpha = obj$ate_alpha,
                      treatment_type = obj$treatment_type)
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx],
                 p = if (is.null(p)) NULL else if (is.list(p)) lapply(p, function(v) v[idx]) else p[idx], verbose = FALSE, ...)
    predict(obj_b, X)
  }
  res_boot <- parallel_lapply(seq_len(n_boot), boot_one, n_cores = n_cores)
  te_boot <- matrix(NA_real_, n, n_boot)
  for (b in seq_len(n_boot)) te_boot[, b] <- res_boot[[b]]
  alpha <- obj$ate_alpha
  te_lower <- apply(te_boot, 1, quantile, probs = alpha / 2)
  te_upper <- apply(te_boot, 1, quantile, probs = 1 - alpha / 2)
  list(te = te, te_lower = te_lower, te_upper = te_upper)
}

#' Generalized R-learner for continuous treatment (dose-response on a grid)
#'
#' Cross-fits nuisance regressions for \eqn{E[Y \mid D, X]} and \eqn{E[D \mid X]}, then
#' runs weighted local linear regressions of residuals on a \code{dose_grid} (Epanechnikov
#' weights). Base learners match \code{\link{RLearner}}: \code{"ranger"}, \code{"lm"}, or
#' \code{"xgb"}.
#'
#' @param data A \code{data.frame} with outcome, treatment, and covariates.
#' @param outcome Name of the outcome column (default \code{"Y"}).
#' @param treatment Name of the continuous treatment column (default \code{"D"}).
#' @param covariates Character vector of covariate column names. If \code{NULL}, uses
#'   \code{paste0("X", 1:10)} when those columns exist, otherwise all columns except
#'   \code{outcome} and \code{treatment}.
#' @param dose_grid Dose values at which to estimate \eqn{\partial E[Y(d)\mid X]/\partial d}
#'   (local linear slope in \eqn{D}).
#' @param learner \code{"ranger"}, \code{"lm"}, or \code{"xgb"} (same spirit as \code{\link{RLearner}}).
#' @param n_fold Number of cross-fitting folds for nuisance models.
#' @param bandwidth Kernel half-width for local regression; default \code{sd(D) * n^(-1/5)}.
#' @param seed Optional integer; if set, fold assignment is reproducible.
#' @param ... Passed to \code{ranger::ranger} or \code{xgboost::xgboost}.
#' @return A list with \code{dose_grid}, \code{tau_grid} (matrix, length(\code{dose_grid}) by \eqn{n}),
#'   \code{X} (covariate matrix), and cross-fitted \code{mu_hat}, \code{e_hat}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # r_learner_continuous(...)
#' }
#' @export
r_learner_continuous <- function(data,
                                 outcome = "Y",
                                 treatment = "D",
                                 covariates = NULL,
                                 dose_grid = seq(0, 6, by = 0.5),
                                 learner = c("ranger", "lm", "xgb"),
                                 n_fold = 5L,
                                 bandwidth = NULL,
                                 seed = NULL,
                                 ...) {
  learner <- match.arg(learner)
  if (identical(learner, "xgb")) check_xgboost()
  stopifnot(is.data.frame(data))
  if (is.null(covariates)) {
    cn <- names(data)
    if (all(paste0("X", 1:10) %in% cn)) {
      covariates <- paste0("X", 1:10)
    } else {
      covariates <- setdiff(cn, c(outcome, treatment))
    }
    if (!length(covariates)) stop("No covariates found; set covariates explicitly.")
  }
  y <- data[[outcome]]
  D <- data[[treatment]]
  X_mat <- as.matrix(data[, covariates, drop = FALSE])
  n <- nrow(data)
  if (!is.null(seed)) set.seed(as.integer(seed))

  n_fold <- as.integer(n_fold)[1L]
  fold_id <- sample(rep(seq_len(n_fold), length.out = n))
  mu_hat <- numeric(n)
  e_hat <- numeric(n)

  df_y <- data.frame(D = D, data[, covariates, drop = FALSE], y = y,
                     check.names = FALSE)
  df_d <- data.frame(data[, covariates, drop = FALSE], D = D, check.names = FALSE)

  for (k in seq_len(n_fold)) {
    tr <- fold_id != k
    va <- fold_id == k
    if (identical(learner, "lm")) {
      m_y <- stats::lm(y ~ ., data = df_y[tr, , drop = FALSE])
      mu_hat[va] <- as.vector(stats::predict(m_y, newdata = df_y[va, names(df_y) != "y", drop = FALSE]))
      m_d <- stats::lm(D ~ ., data = df_d[tr, , drop = FALSE])
      e_hat[va] <- as.vector(stats::predict(m_d, newdata = df_d[va, names(df_d) != "D", drop = FALSE]))
    } else if (identical(learner, "ranger")) {
      m_y <- ranger::ranger(y ~ ., data = df_y[tr, , drop = FALSE], ...)
      mu_hat[va] <- predict(m_y, data = df_y[va, names(df_y) != "y", drop = FALSE])$predictions
      m_d <- ranger::ranger(D ~ ., data = df_d[tr, , drop = FALSE], ...)
      e_hat[va] <- predict(m_d, data = df_d[va, names(df_d) != "D", drop = FALSE])$predictions
    } else {
      Xy_tr <- as.matrix(df_y[tr, names(df_y) != "y", drop = FALSE])
      Xy_va <- as.matrix(df_y[va, names(df_y) != "y", drop = FALSE])
      m_y <- do.call(xgboost::xgboost, .xgb_reg_args(Xy_tr, df_y$y[tr], ..., default_nrounds = 50L))
      mu_hat[va] <- as.vector(predict(m_y, Xy_va))
      Xd_tr <- as.matrix(df_d[tr, names(df_d) != "D", drop = FALSE])
      Xd_va <- as.matrix(df_d[va, names(df_d) != "D", drop = FALSE])
      m_d <- do.call(xgboost::xgboost, .xgb_reg_args(Xd_tr, df_d$D[tr], ..., default_nrounds = 50L))
      e_hat[va] <- as.vector(predict(m_d, Xd_va))
    }
  }

  Y_resid <- y - mu_hat
  D_resid <- D - e_hat
  if (is.null(bandwidth)) bandwidth <- stats::sd(D) * n^(-1 / 5)

  tau_estimates <- matrix(NA_real_, nrow = length(dose_grid), ncol = n)
  colnames(tau_estimates) <- paste0("unit_", seq_len(n))
  rownames(tau_estimates) <- paste0("dose_", dose_grid)

  for (i in seq_along(dose_grid)) {
    d0 <- dose_grid[i]
    bw <- bandwidth
    wts <- pmax(0, 1 - ((D - d0) / bw)^2)
    # Epanechnikov weights can be all zero (e.g. binary D, interior d0, small bw); widen until some mass
    while (sum(wts > 1e-10) < min(30L, ceiling(0.05 * n)) && bw < 1e6 * pmax(bandwidth, .Machine$double.eps)) {
      bw <- bw * 1.5
      wts <- pmax(0, 1 - ((D - d0) / bw)^2)
    }
    df_local <- data.frame(Y_resid = Y_resid, D_resid = D_resid,
                           data[, covariates, drop = FALSE], check.names = FALSE)
    formula_str <- paste("Y_resid ~ D_resid * (", paste(covariates, collapse = " + "), ")")
    local_fit <- stats::lm(stats::as.formula(formula_str), data = df_local, weights = wts)
    cf <- stats::coef(local_fit)
    beta_d <- if (!is.null(names(cf)) && "D_resid" %in% names(cf)) cf[["D_resid"]] else NA_real_
    tau_estimates[i, ] <- beta_d
  }

  list(
    dose_grid = dose_grid,
    tau_grid = tau_estimates,
    X = X_mat,
    mu_hat = mu_hat,
    e_hat = e_hat
  )
}

#' Generic fit_predict for meta-learners
#' @return
#' Object returned by \code{fit_predict}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit_predict(...)
#' }
#' @export
fit_predict <- function(obj, X, treatment, y, p = NULL, ...) {
  UseMethod("fit_predict")
}

#' Generic fit for meta-learners
#' @return
#' Object returned by \code{fit}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit(...)
#' }
#' @export
fit <- function(obj, X, treatment, y, p = NULL, ...) {
  UseMethod("fit")
}

# --- TMLE (Targeted Maximum Likelihood Estimation) ---
# Ref: Gruber & Van Der Laan (2009). Aligned with causalml.inference.meta.tmle.

#' Simple TMLE step: compute ATE and SE from outcome predictions and propensity (like Python simple_tmle)
#' @param y outcome vector
#' @param w treatment indicator (0/1)
#' @param q0w predicted E[Y|X,W=0]
#' @param q1w predicted E[Y|X,W=1]
#' @param p propensity score
#' @param alpha clipping for predictions (default 1e-4)
#' @return list with ate and se
#' @noRd
simple_tmle <- function(y, w, q0w, q1w, p, alpha = 1e-4) {
  y <- as.numeric(y)
  w <- as.integer(w)
  q0w <- as.numeric(q0w)
  q1w <- as.numeric(q1w)
  p <- as.numeric(p)
  p <- pmax(alpha, pmin(1 - alpha, p))
  r <- range(y, na.rm = TRUE)
  if (diff(r) < 1e-10) r[2] <- r[1] + 1
  ystar <- (y - r[1]) / (r[2] - r[1])
  q0 <- pmax(alpha, pmin(1 - alpha, (q0w - r[1]) / (r[2] - r[1])))
  q1 <- pmax(alpha, pmin(1 - alpha, (q1w - r[1]) / (r[2] - r[1])))
  qaw <- q0 * (1 - w) + q1 * w
  intercept <- qlogis(qaw)
  h1 <- w / p
  h0 <- (1 - w) / (1 - p)
  logit_tmle_fn <- function(eps) {
    qaw_eps <- plogis(intercept + eps[1] * h0 + eps[2] * h1)
    -mean(ystar * log(qaw_eps) + (1 - ystar) * log(1 - qaw_eps), na.rm = TRUE)
  }
  opt <- stats::optim(c(0, 0), logit_tmle_fn, method = "BFGS")
  eps <- opt$par
  qawstar <- plogis(intercept + eps[1] * h0 + eps[2] * h1)
  q0star <- (r[2] - r[1]) * plogis(qlogis(q0) + eps[1] / (1 - p)) + r[1]
  q1star <- (r[2] - r[1]) * plogis(qlogis(q1) + eps[2] / p) + r[1]
  qawstar_orig <- (r[2] - r[1]) * qawstar + r[1]
  ic <- (h1 - h0) * (y - qawstar_orig) + (q1star - q0star) - mean(q1star - q0star)
  ate <- mean(q1star - q0star)
  se <- sqrt(var(ic, na.rm = TRUE) / length(y))
  list(ate = ate, se = se)
}

#' TMLE Learner (Targeted Maximum Likelihood Estimation). Like causalml TMLELearner.
#' @param learner character or model for outcome regression (default "ranger")
#' @param ate_alpha confidence level for ATE interval (default 0.05)
#' @param control_name value for control group
#' @param cv optional; if provided, use K-fold CV for outcome (list with splits or number of folds)
#' @return
#' Object returned by \code{TMLELearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # TMLELearner(...)
#' }
#' @export
TMLELearner <- function(learner = "ranger", ate_alpha = 0.05, control_name = 0, cv = NULL) {
  structure(list(learner = learner, ate_alpha = ate_alpha, control_name = control_name, cv = cv,
                 model_tau = NULL, t_groups = NULL),
            class = "TMLELearner")
}

#' Fit TMLE learner (stores outcome model; TMLE step done in estimate_ate)
#' @return
#' Object returned by \code{fit.TMLELearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.TMLELearner(...)
#' }
#' @export
fit.TMLELearner <- function(obj, X, treatment, y, p = NULL, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  check_treatment_vector(treatment, obj$control_name)
  obj$t_groups <- sort(unique(treatment[treatment != obj$control_name]))
  if (is.null(p)) p <- propensity_glmnet(X, as.integer(treatment != obj$control_name), n_fold = 5)
  else p <- as.numeric(p)
  if (length(obj$t_groups) > 1) stop("TMLELearner currently supports single treatment group only")
  g <- obj$t_groups[1]
  w <- as.integer(treatment == g)
  if (is.null(obj$cv)) {
    X_fit <- cbind(w = w, X)
    if (identical(obj$learner, "ranger")) {
      df <- as.data.frame(X_fit); df$y <- y
      obj$model_tau <- ranger::ranger(y ~ ., data = df, ...)
    } else if (identical(obj$learner, "lm")) {
      obj$model_tau <- stats::lm(y ~ ., data = data.frame(X_fit, y = y))
    } else {
      df <- as.data.frame(X_fit); df$y <- y
      obj$model_tau <- ranger::ranger(y ~ ., data = df, ...)
    }
  }
  obj$X_names <- colnames(X)
  obj$p_fit <- p
  obj
}

#' Predict outcome under control and treatment (for TMLE estimate_ate)
#' @noRd
predict_tmle_yhat <- function(model, X, learner) {
  n <- nrow(X)
  X0 <- cbind(w = 0, X)
  X1 <- cbind(w = 1, X)
  if (inherits(model, "ranger")) {
    yhat_c <- predict(model, data = as.data.frame(X0))$predictions
    yhat_t <- predict(model, data = as.data.frame(X1))$predictions
  } else {
    yhat_c <- as.vector(predict(model, newdata = as.data.frame(X0)))
    yhat_t <- as.vector(predict(model, newdata = as.data.frame(X1)))
  }
  list(yhat_c = yhat_c, yhat_t = yhat_t)
}

#' Estimate ATE for TMLE Learner. Like Python TMLELearner.estimate_ate.
#' @param segment optional segment vector (not implemented; for API compatibility)
#' @return
#' Object returned by \code{estimate_ate.TMLELearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # estimate_ate.TMLELearner(...)
#' }
#' @export
estimate_ate.TMLELearner <- function(obj, X, treatment, y, p = NULL, segment = NULL,
                                    return_ci = TRUE, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  if (is.null(obj$model_tau)) obj <- fit(obj, X, treatment, y, p = p, ...)
  g <- obj$t_groups[1]
  w <- as.integer(treatment == g)
  p_use <- obj$p_fit
  if (is.null(p_use)) p_use <- as.numeric(p)
  pred <- predict_tmle_yhat(obj$model_tau, X, obj$learner)
  res <- simple_tmle(y, w, pred$yhat_c, pred$yhat_t, p_use)
  ate <- res$ate
  se <- res$se
  z <- stats::qnorm(1 - obj$ate_alpha / 2)
  ate_lb <- ate - z * se
  ate_ub <- ate + z * se
  if (return_ci) list(ate = ate, ate_lb = ate_lb, ate_ub = ate_ub) else list(ate = ate)
}

#' LRSRegressor: S-Learner with linear regression (like Python LRSRegressor)
#' @return
#' Object returned by \code{LRSRegressor}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # LRSRegressor(...)
#' }
#' @export
LRSRegressor <- function(control_name = 0) {
  SLearner(learner = "lm", control_name = control_name)
}

#' XGBTRegressor: T-Learner with XGBoost (like Python XGBTRegressor)
#' @param control_name value for control group (0 or "control")
#' @return
#' Object returned by \code{XGBTRegressor}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # XGBTRegressor(...)
#' }
#' @export
XGBTRegressor <- function(control_name = 0) {
  TLearner(learner = "xgb", control_name = control_name)
}

#' XGBSRegressor: S-Learner with XGBoost
#' @param control_name value for control group (0 or "control")
#' @return
#' Object returned by \code{XGBSRegressor}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # XGBSRegressor(...)
#' }
#' @export
XGBSRegressor <- function(control_name = 0) {
  SLearner(learner = "xgb", control_name = control_name)
}

#' XGBXRegressor: X-Learner with XGBoost
#' @param control_name value for control group (0 or "control")
#' @return
#' Object returned by \code{XGBXRegressor}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # XGBXRegressor(...)
#' }
#' @export
XGBXRegressor <- function(control_name = 0) {
  XLearner(learner = "xgb", control_name = control_name)
}

#' XGBRRegressor: R-Learner with XGBoost
#' @param control_name value for control group (0 or "control")
#' @param n_fold number of folds for propensity (if p not provided)
#' @return
#' Object returned by \code{XGBRRegressor}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # XGBRRegressor(...)
#' }
#' @export
XGBRRegressor <- function(control_name = 0, n_fold = 5) {
  RLearner(learner = "xgb", control_name = control_name, n_fold = n_fold)
}

# --- DR-Learner (doubly robust pseudo-outcome with cross-fitting) ---
# Reference: Kennedy (2020) https://arxiv.org/abs/2004.14497
# Port of causalml.inference.meta.dr.BaseDRLearner / BaseDRRegressor

#' Check propensity score conditions (in (0,1), length, names)
#' @param p vector, list by group, or matrix of propensity scores
#' @param t_groups treatment group names/ids
#' @param n expected length
#' @noRd
check_p_conditions <- function(p, t_groups, n) {
  if (is.null(p)) return(invisible(TRUE))
  if (is.list(p)) {
    for (g in names(p)) {
      pg <- as.numeric(p[[g]])
      if (length(pg) != n) stop("Propensity vector length must match number of observations.")
      if (any(pg <= 0 | pg >= 1, na.rm = TRUE))
        stop("Propensity scores must be in (0, 1).")
    }
    if (!all(as.character(t_groups) %in% names(p)))
      stop("Propensity list must have an entry for each treatment group.")
  } else {
    pv <- as.numeric(p)
    if (length(pv) != n) stop("Propensity vector length must match number of observations.")
    if (any(pv <= 0 | pv >= 1, na.rm = TRUE))
      stop("Propensity scores must be in (0, 1).")
  }
  invisible(TRUE)
}

#' DR-Learner: doubly robust pseudo-outcome with 3-fold cross-fitting (causalml-style)
#'
#' Estimates CATE by (1) 3-fold cross-fitting: propensity on fold A, outcome on fold B,
#' DR pseudo-outcome and tau on fold C, with rotation so each observation gets OOS predictions;
#' (2) fitting treatment effect models on the DR pseudo-outcome; (3) predicting by averaging
#' the 3 tau-model predictions. Supports multiple treatment groups.
#' Reference: Kennedy (2020) https://arxiv.org/abs/2004.14497
#'
#' @param learner character "lm", "glmnet", "ranger", or "xgb" for all nuisance and tau models (used when separate learners not set)
#' @param control_outcome_learner optional character; model for control outcome (default: use \code{learner})
#' @param treatment_outcome_learner optional character; model for treatment outcome (default: use \code{learner})
#' @param treatment_effect_learner optional character; model for treatment effect (default: use \code{learner})
#' @param control_name value for control group (0 or "control")
#' @param n_fold number of folds for cross-fitting (default 3, as in causalml)
#' @param ate_alpha confidence level for ATE interval (default 0.05)
#' @param seed random seed for fold splits
#' @param treatment_type \code{"binary"} (default) or \code{"continuous"} (DR pseudo-outcome for
#'   conditional dose effects; \code{p} is optional \eqn{\hat{\mathbb{E}}[D \mid X]} of length \eqn{n})
#' @param continuous_delta_d half-step for numerical \eqn{\partial \mu / \partial d} at \eqn{\hat{\mathbb{E}}[D\mid X]};
#'   default \code{max(1e-4, 0.05 * sd(D))}
#' @return Object of class \code{DRLearner}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # DRLearner(...)
#' }
#' @export
DRLearner <- function(learner = "ranger",
                     control_outcome_learner = NULL,
                     treatment_outcome_learner = NULL,
                     treatment_effect_learner = NULL,
                     control_name = 0,
                     n_fold = 3L,
                     ate_alpha = 0.05,
                     seed = NULL,
                     treatment_type = c("binary", "continuous"),
                     continuous_delta_d = NULL) {
  if (is.null(control_outcome_learner)) control_outcome_learner <- learner
  if (is.null(treatment_outcome_learner)) treatment_outcome_learner <- learner
  if (is.null(treatment_effect_learner)) treatment_effect_learner <- learner
  treatment_type <- match.arg(treatment_type)
  structure(list(
    learner = learner,
    learner_mu_c = control_outcome_learner,
    learner_mu_t = treatment_outcome_learner,
    learner_tau = treatment_effect_learner,
    control_name = control_name,
    n_fold = as.integer(n_fold),
    ate_alpha = ate_alpha,
    seed = seed,
    treatment_type = treatment_type,
    continuous_delta_d = continuous_delta_d,
    models_mu_c = NULL,
    models_mu_t = NULL,
    models_tau = NULL,
    models_e = NULL,
    models_yd = NULL,
    propensity = NULL,
    t_groups = NULL,
    X_names = NULL,
    continuous_dr = FALSE,
    dose_values = NULL,
    dose_split = NULL
  ), class = "DRLearner")
}

# ============== LinearDRLearner, SparseLinearDRLearner, XGBDRLearner, ForestDRLearner ==============

#' Linear DR-Learner (econml-style)
#'
#' Special case of \code{\link{DRLearner}} where the final CATE stage is linear regression (OLS).
#' First-stage propensity and outcome models default to flexible learners (ranger); the final
#' stage regresses the DR pseudo-outcome on features. Enables interpretation via coefficients
#' and optional asymptotic inference on the linear CATE model.
#'
#' @param model_propensity character; learner for propensity (default \code{"ranger"})
#' @param model_regression character; learner for outcome regression (default \code{"ranger"})
#' @param fit_cate_intercept logical; whether the linear CATE model has an intercept (default \code{TRUE})
#' @param control_name value for control group
#' @param n_fold number of folds for cross-fitting (default 3)
#' @param ate_alpha confidence level for ATE interval
#' @param seed random seed
#' @return Object of class \code{c("LinearDRLearner", "DRLearner")}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # LinearDRLearner(...)
#' }
#' @export
LinearDRLearner <- function(model_propensity = "ranger",
                            model_regression = "ranger",
                            fit_cate_intercept = TRUE,
                            control_name = 0,
                            n_fold = 3L,
                            ate_alpha = 0.05,
                            seed = NULL) {
  obj <- DRLearner(learner = model_regression,
                   control_outcome_learner = model_regression,
                   treatment_outcome_learner = model_regression,
                   treatment_effect_learner = "lm",
                   control_name = control_name,
                   n_fold = n_fold,
                   ate_alpha = ate_alpha,
                   seed = seed)
  obj$learner_mu_c <- model_regression
  obj$learner_mu_t <- model_regression
  obj$fit_cate_intercept <- fit_cate_intercept
  # Propensity uses same as first stage or glmnet for speed
  if (identical(model_propensity, "ranger")) obj$learner_propensity <- "glmnet" else obj$learner_propensity <- model_propensity
  class(obj) <- c("LinearDRLearner", "DRLearner")
  obj
}

#' Sparse Linear DR-Learner (econml-style)
#'
#' Special case of \code{\link{DRLearner}} where the final CATE stage is L1-penalized regression
#' (glmnet with \code{alpha=1}). Suited for high-dimensional or sparse linear CATE.
#'
#' @param model_propensity character; learner for propensity (default \code{"ranger"})
#' @param model_regression character; learner for outcome regression (default \code{"ranger"})
#' @param fit_cate_intercept logical; whether the CATE linear model has an intercept (default \code{TRUE})
#' @param alpha L1 penalty for final stage glmnet (default 1 for Lasso)
#' @param control_name value for control group
#' @param n_fold number of folds for cross-fitting (default 3)
#' @param ate_alpha confidence level for ATE interval
#' @param seed random seed
#' @return Object of class \code{c("SparseLinearDRLearner", "DRLearner")}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # SparseLinearDRLearner(...)
#' }
#' @export
SparseLinearDRLearner <- function(model_propensity = "ranger",
                                  model_regression = "ranger",
                                  fit_cate_intercept = TRUE,
                                  alpha = 1,
                                  control_name = 0,
                                  n_fold = 3L,
                                  ate_alpha = 0.05,
                                  seed = NULL) {
  obj <- DRLearner(learner = model_regression,
                   control_outcome_learner = model_regression,
                   treatment_outcome_learner = model_regression,
                   treatment_effect_learner = "glmnet",
                   control_name = control_name,
                   n_fold = n_fold,
                   ate_alpha = ate_alpha,
                   seed = seed)
  obj$learner_mu_c <- model_regression
  obj$learner_mu_t <- model_regression
  obj$fit_cate_intercept <- fit_cate_intercept
  obj$sparse_alpha <- alpha
  class(obj) <- c("SparseLinearDRLearner", "DRLearner")
  obj
}

#' XGBoost DR-Learner
#'
#' \code{\link{DRLearner}} with XGBoost for propensity, outcome, and CATE stages.
#' Alias for \code{DRLearner(learner = "xgb", ...)} with class \code{XGBDRLearner}.
#'
#' @param control_name value for control group
#' @param n_fold number of folds for cross-fitting (default 3)
#' @param ate_alpha confidence level for ATE interval
#' @param seed random seed
#' @param control_outcome_learner,treatment_outcome_learner,treatment_effect_learner optional; default \code{"xgb"}
#' @return Object of class \code{c("XGBDRLearner", "DRLearner")}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # XGBDRLearner(...)
#' }
#' @export
XGBDRLearner <- function(control_name = 0,
                         n_fold = 3L,
                         ate_alpha = 0.05,
                         seed = NULL,
                         control_outcome_learner = NULL,
                         treatment_outcome_learner = NULL,
                         treatment_effect_learner = NULL) {
  obj <- DRLearner(learner = "xgb",
                   control_outcome_learner = control_outcome_learner,
                   treatment_outcome_learner = treatment_outcome_learner,
                   treatment_effect_learner = treatment_effect_learner,
                   control_name = control_name,
                   n_fold = n_fold,
                   ate_alpha = ate_alpha,
                   seed = seed)
  class(obj) <- c("XGBDRLearner", "DRLearner")
  obj
}

#' Forest DR-Learner
#'
#' \code{\link{DRLearner}} with random forest (ranger) for propensity, outcome, and CATE stages.
#' Alias for \code{DRLearner(learner = "ranger", ...)} with class \code{ForestDRLearner}.
#'
#' @param control_name value for control group
#' @param n_fold number of folds for cross-fitting (default 3)
#' @param ate_alpha confidence level for ATE interval
#' @param seed random seed
#' @param control_outcome_learner,treatment_outcome_learner,treatment_effect_learner optional; default \code{"ranger"}
#' @return Object of class \code{c("ForestDRLearner", "DRLearner")}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # ForestDRLearner(...)
#' }
#' @export
ForestDRLearner <- function(control_name = 0,
                            n_fold = 3L,
                            ate_alpha = 0.05,
                            seed = NULL,
                            control_outcome_learner = NULL,
                            treatment_outcome_learner = NULL,
                            treatment_effect_learner = NULL) {
  obj <- DRLearner(learner = "ranger",
                   control_outcome_learner = control_outcome_learner,
                   treatment_outcome_learner = treatment_outcome_learner,
                   treatment_effect_learner = treatment_effect_learner,
                   control_name = control_name,
                   n_fold = n_fold,
                   ate_alpha = ate_alpha,
                   seed = seed)
  class(obj) <- c("ForestDRLearner", "DRLearner")
  obj
}

# Out-of-sample propensity for one group: fit on (X_tr, w_tr), predict for X_te
#' @noRd
dr_propensity_oos <- function(X_tr, w_tr, X_te, learner = "glmnet") {
  w_tr <- as.integer(w_tr)
  if (learner == "glmnet") {
    fit <- glmnet::cv.glmnet(X_tr, w_tr, family = "binomial", nfolds = min(5L, max(1L, length(w_tr) - 1)))
    p <- as.vector(predict(fit, newx = X_te, s = "lambda.1se", type = "response"))
  } else {
    df <- as.data.frame(X_tr)
    df$w <- w_tr
    fit <- stats::glm(w ~ ., data = df, family = stats::binomial)
    p <- as.vector(predict(fit, newdata = as.data.frame(X_te), type = "response"))
  }
  pmax(1e-6, pmin(1 - 1e-6, p))
}

#' @noRd
.dr_col_d <- "D_rcausalml"

#' @noRd
dr_build_df_yd <- function(X, D, y) {
  df <- as.data.frame(X)
  if (is.null(names(df)) || any(names(df) == "")) names(df) <- paste0("X", seq_len(ncol(df)))
  df[[.dr_col_d]] <- D
  df$y <- y
  df
}

#' @noRd
dr_pred_mu_yd <- function(m, X_new, D_vec, learner) {
  df <- as.data.frame(X_new)
  if (is.null(names(df)) || any(names(df) == "")) names(df) <- paste0("X", seq_len(ncol(df)))
  df[[.dr_col_d]] <- D_vec
  dr_pred_outcome_model(m, as.matrix(df), learner)
}

#' Fit DR-Learner
#' @return
#' Object returned by \code{fit.DRLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.DRLearner(...)
#' }
#' @export
fit.DRLearner <- function(obj, X, treatment, y, p = NULL, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- length(y)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  if (identical(obj$treatment_type, "continuous")) {
    check_continuous_treatment(treatment)
    D <- treatment
    obj$t_groups <- 1L
    obj$continuous_dr <- TRUE
    obj$dose_values <- as.numeric(stats::quantile(D, c(0.25, 0.75), na.rm = TRUE))
    obj$dose_split <- stats::median(D, na.rm = TRUE)
    del <- obj$continuous_delta_d
    if (is.null(del)) del <- max(1e-4, 0.05 * stats::sd(D))
    obj$continuous_delta_d <- del
    n_fold <- min(obj$n_fold, 3L)
    if (!is.null(obj$seed)) set.seed(obj$seed)
    folds <- sample(rep(seq_len(n_fold), length.out = n))
    split_indices <- lapply(seq_len(n_fold), function(k) which(folds == k))
    obj$models_mu_c <- NULL
    obj$models_mu_t <- NULL
    obj$propensity <- NULL
    obj$models_e <- vector("list", n_fold)
    obj$models_yd <- vector("list", n_fold)
    obj$models_tau <- list(`1` = vector("list", n_fold))
    for (ifold in seq_len(n_fold)) {
      treatment_idx <- split_indices[[ifold]]
      outcome_idx <- split_indices[[(ifold) %% n_fold + 1L]]
      tau_idx <- split_indices[[(ifold + 1L) %% n_fold + 1L]]
      X_tau <- X[tau_idx, , drop = FALSE]
      D_tau <- D[tau_idx]
      y_tau <- y[tau_idx]
      if (is.null(p)) {
        X_treat <- X[treatment_idx, , drop = FALSE]
        D_treat <- D[treatment_idx]
        df_ed <- as.data.frame(X_treat)
        if (is.null(names(df_ed)) || any(names(df_ed) == "")) names(df_ed) <- paste0("X", seq_len(ncol(df_ed)))
        df_ed$y <- D_treat
        obj$models_e[[ifold]] <- dr_fit_outcome_model(df_ed, obj$learner_mu_c, ...)
        e_tau <- dr_pred_outcome_model(obj$models_e[[ifold]], X_tau, obj$learner_mu_c)
      } else {
        pv <- as.numeric(p)
        if (length(pv) != n) stop("For continuous DR-Learner, p must be a numeric vector of length n (E[D|X]).")
        e_tau <- pv[tau_idx]
      }
      X_out <- X[outcome_idx, , drop = FALSE]
      D_out <- D[outcome_idx]
      y_out <- y[outcome_idx]
      df_yd <- dr_build_df_yd(X_out, D_out, y_out)
      obj$models_yd[[ifold]] <- dr_fit_outcome_model(df_yd, obj$learner_mu_t, ...)
      mu_obs <- dr_pred_mu_yd(obj$models_yd[[ifold]], X_tau, D_tau, obj$learner_mu_t)
      v <- D_tau - e_tau
      sig2 <- mean(v^2, na.rm = TRUE)
      if (!is.finite(sig2) || sig2 < 1e-10) sig2 <- 1e-10
      r <- y_tau - mu_obs
      mu_up <- dr_pred_mu_yd(obj$models_yd[[ifold]], X_tau, e_tau + del, obj$learner_mu_t)
      mu_dn <- dr_pred_mu_yd(obj$models_yd[[ifold]], X_tau, e_tau - del, obj$learner_mu_t)
      tau_add <- (mu_up - mu_dn) / (2 * del)
      dr_vec <- v / sig2 * r + tau_add
      df_dr <- as.data.frame(X_tau)
      df_dr$dr <- dr_vec
      fit_intercept_tau <- if (!is.null(obj$fit_cate_intercept)) obj$fit_cate_intercept else TRUE
      alpha_tau <- if (!is.null(obj$sparse_alpha)) obj$sparse_alpha else NULL
      obj$models_tau[["1"]][[ifold]] <- dr_fit_outcome_model(df_dr, obj$learner_tau,
        fit_cate_intercept = fit_intercept_tau, alpha_glmnet = alpha_tau, ...)
    }
    obj$X_names <- colnames(X)
    return(obj)
  }
  check_treatment_vector(treatment, obj$control_name)
  obj$continuous_dr <- FALSE
  obj$models_e <- NULL
  obj$models_yd <- NULL
  obj$dose_values <- NULL
  obj$dose_split <- NULL
  t_groups <- sort(unique(treatment[treatment != obj$control_name]))
  obj$t_groups <- t_groups
  if (!is.null(p)) check_p_conditions(p, t_groups, n)

  n_fold <- min(obj$n_fold, 3L)
  if (!is.null(obj$seed)) set.seed(obj$seed)
  folds <- sample(rep(seq_len(n_fold), length.out = n))
  split_indices <- lapply(seq_len(n_fold), function(k) which(folds == k))

  obj$models_mu_c <- vector("list", n_fold)
  obj$models_mu_t <- setNames(vector("list", length(t_groups)), as.character(t_groups))
  for (g in t_groups) obj$models_mu_t[[as.character(g)]] <- vector("list", n_fold)
  obj$models_tau <- setNames(vector("list", length(t_groups)), as.character(t_groups))
  for (g in t_groups) obj$models_tau[[as.character(g)]] <- vector("list", n_fold)
  if (is.null(p)) obj$propensity <- setNames(vector("list", length(t_groups)), as.character(t_groups))
  for (g in t_groups) {
    if (is.null(p)) obj$propensity[[as.character(g)]] <- numeric(n)
  }

  for (ifold in seq_len(n_fold)) {
    treatment_idx <- split_indices[[ifold]]
    outcome_idx <- split_indices[[(ifold) %% n_fold + 1L]]
    tau_idx <- split_indices[[(ifold + 1L) %% n_fold + 1L]]

    treatment_treat <- treatment[treatment_idx]
    treatment_out <- treatment[outcome_idx]
    treatment_tau <- treatment[tau_idx]
    y_out <- y[outcome_idx]
    y_tau <- y[tau_idx]
    X_treat <- X[treatment_idx, , drop = FALSE]
    X_out <- X[outcome_idx, , drop = FALSE]
    X_tau <- X[tau_idx, , drop = FALSE]

    cur_p <- list()
    if (is.null(p)) {
      for (g in t_groups) {
        mask <- (treatment_treat == g) | (treatment_treat == obj$control_name)
        if (sum(mask) < 10L) next
        X_filt <- X_treat[mask, , drop = FALSE]
        w_filt <- as.integer(treatment_treat[mask] == g)
        cur_p[[as.character(g)]] <- dr_propensity_oos(X_filt, w_filt, X_tau,
          if (!is.null(obj$learner_propensity)) obj$learner_propensity else obj$learner)
        obj$propensity[[as.character(g)]][tau_idx] <- cur_p[[as.character(g)]]
      }
    } else {
      if (is.vector(p) || (is.matrix(p) && ncol(p) == 1L)) {
        cur_p <- setNames(list(as.numeric(p)[tau_idx]), as.character(t_groups[1L]))
      } else if (is.list(p)) {
        cur_p <- lapply(p, function(prop) as.numeric(prop)[tau_idx])
        names(cur_p) <- names(p)
      } else {
        cur_p <- setNames(list(as.numeric(p)[tau_idx]), as.character(t_groups[1L]))
      }
    }

    # Outcome regressions: mu_c on control, mu_t per group on outcome fold (use per-role learners)
    idx_c <- which(treatment_out == obj$control_name)
    if (length(idx_c) > 0L) {
      df_c <- as.data.frame(X_out[idx_c, , drop = FALSE])
      df_c$y <- y_out[idx_c]
      obj$models_mu_c[[ifold]] <- dr_fit_outcome_model(df_c, obj$learner_mu_c, ...)
    }
    for (g in t_groups) {
      idx_t <- which(treatment_out == g)
      if (length(idx_t) > 0L) {
        df_t <- as.data.frame(X_out[idx_t, , drop = FALSE])
        df_t$y <- y_out[idx_t]
        obj$models_mu_t[[as.character(g)]][[ifold]] <- dr_fit_outcome_model(df_t, obj$learner_mu_t, ...)
      }
    }

    # DR pseudo-outcome and tau fit per group (use treatment_effect_learner)
    for (g in t_groups) {
      p_cur <- if (is.null(p)) cur_p[[as.character(g)]] else cur_p[[as.character(g)]]
      if (is.null(p_cur)) next
      mask <- (treatment_tau == g) | (treatment_tau == obj$control_name)
      treatment_filt <- treatment_tau[mask]
      X_filt <- X_tau[mask, , drop = FALSE]
      y_filt <- y_tau[mask]
      w_filt <- as.integer(treatment_filt == g)
      p_filt <- p_cur[mask]
      mu_c <- dr_pred_outcome_model(obj$models_mu_c[[ifold]], X_filt, obj$learner_mu_c)
      mu_t <- dr_pred_outcome_model(obj$models_mu_t[[as.character(g)]][[ifold]], X_filt, obj$learner_mu_t)
      denom <- p_filt * (1 - p_filt)
      denom[denom < 1e-8] <- 1e-8
      dr <- (w_filt - p_filt) / denom * (y_filt - mu_t * w_filt - mu_c * (1L - w_filt)) + mu_t - mu_c
      df_dr <- as.data.frame(X_filt)
      df_dr$dr <- dr
      fit_intercept_tau <- if (!is.null(obj$fit_cate_intercept)) obj$fit_cate_intercept else TRUE
      alpha_tau <- if (!is.null(obj$sparse_alpha)) obj$sparse_alpha else NULL
      obj$models_tau[[as.character(g)]][[ifold]] <- dr_fit_outcome_model(df_dr, obj$learner_tau,
        fit_cate_intercept = fit_intercept_tau, alpha_glmnet = alpha_tau, ...)
    }
  }

  obj$X_names <- colnames(X)
  obj
}

# Fit outcome/pseudo-outcome model (used for mu_c, mu_t, tau)
#' @param fit_cate_intercept used only when learner is "lm"; if FALSE fit without intercept (dr ~ . - 1)
#' @noRd
dr_fit_outcome_model <- function(df, learner, ..., fit_cate_intercept = TRUE, alpha_glmnet = NULL) {
  resp_idx <- which(names(df) %in% c("y", "dr"))[1]
  resp_name <- names(df)[resp_idx]
  if (learner == "lm") {
    form <- if (isTRUE(fit_cate_intercept)) paste(resp_name, "~ .") else paste(resp_name, "~ . - 1")
    m <- stats::lm(as.formula(form), data = df)
    list(type = "lm", model = m)
  } else if (learner == "ranger") {
    m <- ranger::ranger(as.formula(paste(resp_name, "~ .")), data = df, ...)
    list(type = "ranger", model = m)
  } else if (learner == "glmnet") {
    alpha <- if (is.null(alpha_glmnet)) 1 else alpha_glmnet
    m <- glmnet::cv.glmnet(as.matrix(df[, -resp_idx]), df[[resp_name]], nfolds = 5L, alpha = alpha)
    list(type = "glmnet", model = m)
  } else if (learner == "xgb") {
    check_xgboost_dr()
    x_mat <- as.matrix(df[, -resp_idx])
    y_vec <- df[[resp_name]]
    m <- do.call(xgboost::xgboost, .xgb_reg_args(x_mat, y_vec, ..., default_nrounds = 50L))
    list(type = "xgb", model = m)
  } else {
    stop("learner must be 'lm', 'ranger', 'glmnet', or 'xgb'")
  }
}

#' @noRd
dr_pred_outcome_model <- function(m, X_new, learner) {
  if (is.null(m)) return(numeric(nrow(X_new)))
  if (inherits(X_new, "data.frame")) X_new <- as.matrix(X_new)
  df_new <- as.data.frame(X_new)
  if (m$type == "lm") return(as.vector(predict(m$model, newdata = df_new)))
  if (m$type == "ranger") return(predict(m$model, data = df_new)$predictions)
  if (m$type == "glmnet") return(as.vector(predict(m$model, newx = X_new, s = "lambda.1se")[, 1]))
  if (m$type == "xgb") return(as.vector(predict(m$model, newdata = X_new)))
  numeric(nrow(X_new))
}

#' Predict CATE from DR-Learner
#' @param object fitted DRLearner
#' @param newdata feature matrix
#' @param treatment optional; if provided with \code{y}, used for regression metrics when \code{verbose=TRUE}
#' @param y optional; if provided with \code{treatment}, used for regression metrics when \code{verbose=TRUE}
#' @param return_components if TRUE return list with \code{te}, \code{yhat_cs}, \code{yhat_ts}
#' @param verbose if TRUE and \code{treatment} and \code{y} provided, print regression metrics per group
#' @param ... unused
#' @return
#' Object returned by \code{predict.DRLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.DRLearner(...)
#' }
#' @export
predict.DRLearner <- function(object, newdata, treatment = NULL, y = NULL,
                              return_components = FALSE, verbose = TRUE, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names
  n <- nrow(newdata)
  t_groups <- object$t_groups
  n_t <- length(t_groups)
  if (isTRUE(object$continuous_dr)) {
    d0 <- object$dose_values[1L]
    d1 <- object$dose_values[2L]
    models_tau_g <- object$models_tau[["1"]]
    preds_tau <- matrix(NA_real_, n, length(models_tau_g))
    for (j in seq_along(models_tau_g)) {
      if (!is.null(models_tau_g[[j]]))
        preds_tau[, j] <- dr_pred_outcome_model(models_tau_g[[j]], newdata, object$learner_tau)
    }
    te <- matrix(rowMeans(preds_tau, na.rm = TRUE), ncol = 1L)
    preds_yd_c <- matrix(NA_real_, n, length(object$models_yd))
    preds_yd_t <- matrix(NA_real_, n, length(object$models_yd))
    for (j in seq_along(object$models_yd)) {
      if (!is.null(object$models_yd[[j]])) {
        preds_yd_c[, j] <- dr_pred_mu_yd(object$models_yd[[j]], newdata, rep(d0, n), object$learner_mu_t)
        preds_yd_t[, j] <- dr_pred_mu_yd(object$models_yd[[j]], newdata, rep(d1, n), object$learner_mu_t)
      }
    }
    yhat_cs <- list(`1` = rowMeans(preds_yd_c, na.rm = TRUE))
    yhat_ts <- list(`1` = rowMeans(preds_yd_t, na.rm = TRUE))
    if (verbose && !is.null(treatment) && !is.null(y)) {
      treatment <- as.numeric(treatment)
      y <- as.numeric(y)
      yhat <- yhat_cs[[1L]] * as.integer(treatment <= object$dose_split) +
        yhat_ts[[1L]] * as.integer(treatment > object$dose_split)
      w <- as.integer(treatment > object$dose_split)
      msg <- regression_metrics(y, yhat, w)
      message("Continuous DR-Learner (high vs low D split at median fit): MSE control = ",
              round(msg$mse[1], 4), ", treated = ", round(msg$mse[2], 4))
    }
    if (return_components) return(list(te = te, yhat_cs = yhat_cs, yhat_ts = yhat_ts))
    return(drop(te))
  }
  te <- matrix(NA_real_, n, n_t)
  yhat_cs <- list()
  yhat_ts <- list()
  for (i in seq_along(t_groups)) {
    g <- t_groups[i]
    models_tau_g <- object$models_tau[[as.character(g)]]
    preds_tau <- matrix(NA_real_, n, length(models_tau_g))
    for (j in seq_along(models_tau_g)) {
      if (!is.null(models_tau_g[[j]]))
        preds_tau[, j] <- dr_pred_outcome_model(models_tau_g[[j]], newdata, object$learner_tau)
    }
    te[, i] <- rowMeans(preds_tau, na.rm = TRUE)
    preds_mu_c <- matrix(NA_real_, n, length(object$models_mu_c))
    for (j in seq_along(object$models_mu_c)) {
      if (!is.null(object$models_mu_c[[j]]))
        preds_mu_c[, j] <- dr_pred_outcome_model(object$models_mu_c[[j]], newdata, object$learner_mu_c)
    }
    yhat_cs[[as.character(g)]] <- rowMeans(preds_mu_c, na.rm = TRUE)
    preds_mu_t <- matrix(NA_real_, n, length(object$models_mu_t[[as.character(g)]]))
    for (j in seq_along(object$models_mu_t[[as.character(g)]])) {
      m <- object$models_mu_t[[as.character(g)]][[j]]
      if (!is.null(m)) preds_mu_t[, j] <- dr_pred_outcome_model(m, newdata, object$learner_mu_t)
    }
    yhat_ts[[as.character(g)]] <- rowMeans(preds_mu_t, na.rm = TRUE)

    if (verbose && !is.null(treatment) && !is.null(y)) {
      treatment <- as.numeric(treatment)
      y <- as.numeric(y)
      mask <- (treatment == g) | (treatment == object$control_name)
      treatment_filt <- treatment[mask]
      y_filt <- y[mask]
      w <- as.integer(treatment_filt == g)
      yhat <- numeric(length(y_filt))
      yhat[w == 0] <- yhat_cs[[as.character(g)]][mask][w == 0]
      yhat[w == 1] <- yhat_ts[[as.character(g)]][mask][w == 1]
      msg <- regression_metrics(y_filt, yhat, w)
      message("Error metrics for group ", g, ": MSE control = ", round(msg$mse[1], 4),
              ", treated = ", round(msg$mse[2], 4))
    }
  }
  if (return_components) list(te = te, yhat_cs = yhat_cs, yhat_ts = yhat_ts)
  else if (n_t == 1L) drop(te) else te
}

#' Fit and predict CATE (DR-learner)
#' @param obj DRLearner object
#' @param X covariate matrix
#' @param treatment treatment vector
#' @param y outcome vector
#' @param p optional propensity (vector or list by group)
#' @param return_ci if TRUE return list with \code{te}, \code{te_lower}, \code{te_upper} (bootstrap)
#' @param n_bootstraps number of bootstrap iterations (default 1000)
#' @param bootstrap_size number of samples per bootstrap (default 10000; uses min(bootstrap_size, n))
#' @param n_cores number of cores for bootstrap when return_ci=TRUE (1 = sequential)
#' @param return_components if TRUE return \code{te}, \code{yhat_cs}, \code{yhat_ts}
#' @param seed random seed for fit and optional bootstrap
#' @param ... passed to \code{fit}
#' @return
#' Object returned by \code{fit_predict.DRLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit_predict.DRLearner(...)
#' }
#' @export
fit_predict.DRLearner <- function(obj, X, treatment, y, p = NULL,
                                  return_ci = FALSE,
                                  n_bootstraps = 1000L,
                                  bootstrap_size = 10000L,
                                  n_cores = 1L,
                                  return_components = FALSE,
                                  seed = NULL,
                                  ...) {
  if (!is.null(seed)) obj$seed <- seed
  obj <- fit(obj, X, treatment, y, p = p, ...)
  out <- predict(obj, X, return_components = return_components, verbose = FALSE)
  if (return_components) {
    if (!return_ci) return(list(fit = obj, te = out$te, yhat_cs = out$yhat_cs, yhat_ts = out$yhat_ts))
    te <- out$te
  } else {
    if (!return_ci) return(out)
    te <- out
  }
  if (is.vector(te)) te <- matrix(te, ncol = 1)
  n <- nrow(te)
  size <- min(as.integer(bootstrap_size), n)
  n_boot <- as.integer(n_bootstraps)
  t_groups_global <- obj$t_groups
  models_mu_c_global <- obj$models_mu_c
  models_mu_t_global <- obj$models_mu_t
  models_tau_global <- obj$models_tau
  models_e_global <- obj$models_e
  models_yd_global <- obj$models_yd
  continuous_dr_global <- obj$continuous_dr
  dose_values_global <- obj$dose_values
  dose_split_global <- obj$dose_split
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  boot_one <- function(b) {
    idx <- sample(n, size = size, replace = TRUE)
    obj_b <- DRLearner(learner = obj$learner,
                       control_outcome_learner = obj$learner_mu_c,
                       treatment_outcome_learner = obj$learner_mu_t,
                       treatment_effect_learner = obj$learner_tau,
                       control_name = obj$control_name,
                       n_fold = obj$n_fold,
                       ate_alpha = obj$ate_alpha,
                       seed = obj$seed,
                       treatment_type = obj$treatment_type,
                       continuous_delta_d = obj$continuous_delta_d)
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx],
                 p = if (is.null(p)) NULL else if (is.list(p)) lapply(p, function(v) v[idx]) else p[idx],
                 ...)
    predict(obj_b, X)
  }
  res_boot <- parallel_lapply(seq_len(n_boot), boot_one, n_cores = n_cores)
  te_boot <- array(NA_real_, c(n, ncol(te), n_boot))
  for (b in seq_len(n_boot)) te_boot[, , b] <- res_boot[[b]]
  obj$t_groups <- t_groups_global
  obj$models_mu_c <- models_mu_c_global
  obj$models_mu_t <- models_mu_t_global
  obj$models_tau <- models_tau_global
  obj$models_e <- models_e_global
  obj$models_yd <- models_yd_global
  obj$continuous_dr <- continuous_dr_global
  obj$dose_values <- dose_values_global
  obj$dose_split <- dose_split_global
  alpha <- obj$ate_alpha
  te_lower <- apply(te_boot, c(1, 2), quantile, probs = alpha / 2)
  te_upper <- apply(te_boot, c(1, 2), quantile, probs = 1 - alpha / 2)
  if (return_components)
    list(fit = obj, te = te, te_lower = te_lower, te_upper = te_upper,
         yhat_cs = out$yhat_cs, yhat_ts = out$yhat_ts)
  else
    list(te = te, te_lower = te_lower, te_upper = te_upper)
}

#' Estimate Average Treatment Effect (ATE) for DR-Learner
#'
#' Uses analytical standard error (Imbens & Wooldridge 2009) by default; optional bootstrap CI.
#' Reference: formula (7) in "Recent Developments in the Econometrics of Program Evaluation."
#'
#' @param obj DRLearner object (fitted or not)
#' @param X covariate matrix
#' @param treatment treatment vector
#' @param y outcome vector
#' @param p optional propensity (vector or list by group)
#' @param bootstrap_ci if TRUE use bootstrap for CI instead of analytical SE
#' @param n_bootstraps number of bootstrap iterations (default 1000)
#' @param bootstrap_size number of samples per bootstrap (default 10000)
#' @param n_cores number of cores for bootstrap CI (1 = sequential)
#' @param seed random seed for fit and bootstrap
#' @param pretrain if TRUE, assume model already fitted (use predict only)
#' @param ... passed to \code{fit} or \code{fit_predict}
#' @param return_ci if FALSE return only \code{ate} vector (default TRUE)
#' @return list with \code{ate}, \code{ate_lb}, \code{ate_ub} (vectors, one per treatment group); if \code{return_ci=FALSE} only \code{ate}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # estimate_ate.DRLearner(...)
#' }
#' @export
estimate_ate.DRLearner <- function(obj, X, treatment, y, p = NULL,
                                    return_ci = TRUE,
                                    bootstrap_ci = FALSE,
                                    n_bootstraps = 1000L,
                                    bootstrap_size = 10000L,
                                    n_cores = 1L,
                                    seed = NULL,
                                    pretrain = FALSE,
                                    ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- length(y)
  t_groups <- obj$t_groups
  if (is.null(t_groups)) {
    check_treatment_vector(treatment, obj$control_name)
    t_groups <- sort(unique(treatment[treatment != obj$control_name]))
  }
  if (pretrain) {
    out <- predict(obj, X, return_components = TRUE, verbose = FALSE)
    te <- out$te
    yhat_cs <- out$yhat_cs
    yhat_ts <- out$yhat_ts
  } else {
    if (!is.null(seed)) obj$seed <- seed
    fp <- fit_predict(obj, X, treatment, y, p = p, return_components = TRUE, seed = seed, ...)
    obj <- fp$fit
    te <- fp$te
    yhat_cs <- fp$yhat_cs
    yhat_ts <- fp$yhat_ts
  }
  if (is.vector(te)) te <- matrix(te, ncol = 1)
  p_use <- if (is.null(p)) obj$propensity else p
  if (!is.null(p_use) && !isTRUE(obj$continuous_dr)) check_p_conditions(p_use, t_groups, n)
  if (is.list(p_use)) {
    p_dict <- lapply(p_use, as.numeric)
  } else if (!is.null(p_use)) {
    p_dict <- setNames(list(as.numeric(p_use)), as.character(t_groups[1L]))
  } else {
    p_dict <- obj$propensity
  }
  n_t <- length(t_groups)
  ate <- numeric(n_t)
  ate_lb <- numeric(n_t)
  ate_ub <- numeric(n_t)
  for (i in seq_len(n_t)) {
    g <- t_groups[i]
    ate[i] <- mean(te[, i], na.rm = TRUE)
    if (isTRUE(obj$continuous_dr)) {
      w_full <- as.integer(treatment > obj$dose_split)
      prob_treatment <- mean(w_full)
      yhat_c <- yhat_cs[[as.character(g)]]
      yhat_t <- yhat_ts[[as.character(g)]]
      var_c <- if (sum(w_full == 0) > 1) stats::var(y[w_full == 0] - yhat_c[w_full == 0]) else 0
      var_t <- if (sum(w_full == 1) > 1) stats::var(y[w_full == 1] - yhat_t[w_full == 1]) else 0
      var_tau <- stats::var(yhat_t - yhat_c)
      se <- sqrt((var_c / (1 - prob_treatment) + var_t / prob_treatment + var_tau) / n)
    } else {
      mask <- (treatment == g) | (treatment == obj$control_name)
      treatment_filt <- treatment[mask]
      w <- as.integer(treatment_filt == g)
      prob_treatment <- mean(w)
      yhat_c <- yhat_cs[[as.character(g)]][mask]
      yhat_t <- yhat_ts[[as.character(g)]][mask]
      y_filt <- y[mask]
      n_filt <- length(y_filt)
      var_c <- if (sum(w == 0) > 1) var(y_filt[w == 0] - yhat_c[w == 0]) else 0
      var_t <- if (sum(w == 1) > 1) var(y_filt[w == 1] - yhat_t[w == 1]) else 0
      var_tau <- var(yhat_t - yhat_c)
      se <- sqrt((var_c / (1 - prob_treatment) + var_t / prob_treatment + var_tau) / n_filt)
    }
    if (!is.finite(se) || se <= 0) se <- sqrt(var(te[, i], na.rm = TRUE) / n)
    z <- stats::qnorm(1 - obj$ate_alpha / 2)
    ate_lb[i] <- ate[i] - se * z
    ate_ub[i] <- ate[i] + se * z
  }
  if (!bootstrap_ci)
    return(if (return_ci) list(ate = ate, ate_lb = ate_lb, ate_ub = ate_ub) else list(ate = ate))
  t_groups_global <- obj$t_groups
  models_mu_c_global <- obj$models_mu_c
  models_mu_t_global <- obj$models_mu_t
  models_tau_global <- obj$models_tau
  models_e_global <- obj$models_e
  models_yd_global <- obj$models_yd
  continuous_dr_global <- obj$continuous_dr
  dose_values_global <- obj$dose_values
  dose_split_global <- obj$dose_split
  size <- min(as.integer(bootstrap_size), n)
  n_boot <- as.integer(n_bootstraps)
  boot_one <- function(b) {
    idx <- sample(n, size = size, replace = TRUE)
    obj_b <- DRLearner(learner = obj$learner,
                       control_outcome_learner = obj$learner_mu_c,
                       treatment_outcome_learner = obj$learner_mu_t,
                       treatment_effect_learner = obj$learner_tau,
                       control_name = obj$control_name,
                       n_fold = obj$n_fold,
                       ate_alpha = obj$ate_alpha,
                       seed = obj$seed,
                       treatment_type = obj$treatment_type,
                       continuous_delta_d = obj$continuous_delta_d)
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx],
                 p = if (is.null(p)) NULL else if (is.list(p)) lapply(p, function(v) v[idx]) else p[idx],
                 ...)
    te_b <- predict(obj_b, X)
    if (is.vector(te_b)) te_b <- matrix(te_b, ncol = 1)
    colMeans(te_b, na.rm = TRUE)
  }
  res_boot <- parallel_lapply(seq_len(n_boot), boot_one, n_cores = n_cores)
  ate_boot <- matrix(NA_real_, n_t, n_boot)
  for (b in seq_len(n_boot)) ate_boot[, b] <- res_boot[[b]]
  obj$t_groups <- t_groups_global
  obj$models_mu_c <- models_mu_c_global
  obj$models_mu_t <- models_mu_t_global
  obj$models_tau <- models_tau_global
  obj$models_e <- models_e_global
  obj$models_yd <- models_yd_global
  obj$continuous_dr <- continuous_dr_global
  obj$dose_values <- dose_values_global
  obj$dose_split <- dose_split_global
  alpha <- obj$ate_alpha
  ate_lower <- apply(ate_boot, 1, quantile, probs = alpha / 2)
  ate_upper <- apply(ate_boot, 1, quantile, probs = 1 - alpha / 2)
  if (return_ci) list(ate = ate, ate_lb = ate_lower, ate_ub = ate_upper) else list(ate = ate)
}

# ---------- coef and intercept for LinearDRLearner / SparseLinearDRLearner ----------

#' Coefficient(s) of the linear CATE model (LinearDRLearner or SparseLinearDRLearner)
#'
#' Returns the slope coefficients (excluding intercept) for the final-stage linear CATE model,
#' averaged across cross-fitting folds. For multiple treatment groups, use \code{treatment} to select one.
#'
#' @param object fitted \code{LinearDRLearner} or \code{SparseLinearDRLearner}
#' @param treatment which treatment group (default first non-control). Ignored if only one group.
#' @param ... unused
#' @return vector of coefficients (slopes) for the selected treatment, or list of vectors if \code{treatment} is NULL and multiple groups.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # coef.LinearDRLearner(...)
#' }
#' @export
coef.LinearDRLearner <- function(object, treatment = NULL, ...) {
  dr_coef_inner(object, treatment = treatment, type = "lm")
}

#' @rdname coef.LinearDRLearner
#' @return
#' Object returned by \code{coef.SparseLinearDRLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # coef.SparseLinearDRLearner(...)
#' }
#' @export
coef.SparseLinearDRLearner <- function(object, treatment = NULL, ...) {
  dr_coef_inner(object, treatment = treatment, type = "glmnet")
}

#' Intercept of the linear CATE model (generic)
#' @param object fitted model (e.g. LinearDRLearner, SparseLinearDRLearner)
#' @param ... passed to methods
#' @return
#' Object returned by \code{intercept}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # intercept(...)
#' }
#' @export
intercept <- function(object, ...) UseMethod("intercept")

#' Intercept(s) of the linear CATE model (LinearDRLearner or SparseLinearDRLearner)
#'
#' @param object fitted \code{LinearDRLearner} or \code{SparseLinearDRLearner}
#' @param treatment which treatment group (default first non-control)
#' @param ... unused
#' @return scalar intercept for the selected treatment, or vector of intercepts for all groups if \code{treatment} is NULL.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # intercept.LinearDRLearner(...)
#' }
#' @export
intercept.LinearDRLearner <- function(object, treatment = NULL, ...) {
  dr_intercept_inner(object, treatment = treatment, type = "lm")
}

#' @rdname intercept.LinearDRLearner
#' @return
#' Object returned by \code{intercept.SparseLinearDRLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # intercept.SparseLinearDRLearner(...)
#' }
#' @export
intercept.SparseLinearDRLearner <- function(object, treatment = NULL, ...) {
  dr_intercept_inner(object, treatment = treatment, type = "glmnet")
}

#' @noRd
dr_coef_inner <- function(object, treatment, type) {
  t_groups <- object$t_groups
  if (is.null(t_groups)) return(NULL)
  n_t <- length(t_groups)
  if (n_t == 1L) treatment <- t_groups[1]
  if (is.null(treatment)) {
    return(setNames(lapply(t_groups, function(g) dr_coef_inner(object, treatment = g, type = type)), as.character(t_groups)))
  }
  models <- object$models_tau[[as.character(treatment)]]
  if (is.null(models)) return(NULL)
  valid <- vapply(models, function(m) !is.null(m) && (if (type == "lm") inherits(m$model, "lm") else inherits(m$model, "cv.glmnet")), logical(1))
  if (!any(valid)) return(NULL)
  models <- models[valid]
  if (type == "lm") {
    # coef(model) = (intercept, slope1, ...) or (slope1, ...); return slopes only
    coefs <- lapply(models, function(m) {
      cf <- stats::coef(m$model)
      if ("(Intercept)" %in% names(cf)) cf[-1] else cf
    })
  } else {
    coefs <- lapply(models, function(m) {
      cf <- as.numeric(coef(m$model, s = "lambda.1se"))
      if (length(cf) <= 1) numeric(0) else cf[-1]
    })
  }
  if (length(coefs) == 0) return(NULL)
  nms <- unique(unlist(lapply(coefs, names)))
  if (length(nms) == 0) nms <- NULL
  # When names are missing (e.g. glmnet as.numeric), align by position
  if (is.null(nms)) {
    len <- max(vapply(coefs, length, integer(1)))
    out <- rowMeans(do.call(cbind, lapply(coefs, function(c) c(c, rep(NA, len - length(c)))[seq_len(len)])))
    if (length(object$X_names) >= len) names(out) <- object$X_names[seq_len(len)]
  } else {
    out <- rowMeans(do.call(cbind, lapply(coefs, function(c) c(c)[match(nms, names(c))])))
    names(out) <- nms
  }
  out
}

#' @noRd
dr_intercept_inner <- function(object, treatment, type) {
  t_groups <- object$t_groups
  if (is.null(t_groups)) return(NULL)
  n_t <- length(t_groups)
  if (n_t == 1L && is.null(treatment)) treatment <- t_groups[1]
  if (is.null(treatment)) {
    return(setNames(vapply(t_groups, function(g) dr_intercept_inner(object, treatment = g, type = type), numeric(1)), as.character(t_groups)))
  }
  models <- object$models_tau[[as.character(treatment)]]
  if (is.null(models)) return(NA_real_)
  valid <- vapply(models, function(m) !is.null(m) && (if (type == "lm") inherits(m$model, "lm") else inherits(m$model, "cv.glmnet")), logical(1))
  models <- models[valid]
  if (length(models) == 0) return(NA_real_)
  if (type == "lm") {
    intercepts <- vapply(models, function(m) {
      cf <- stats::coef(m$model)
      if ("(Intercept)" %in% names(cf)) cf["(Intercept)"] else 0
    }, numeric(1))
  } else {
    intercepts <- vapply(models, function(m) as.numeric(coef(m$model, s = "lambda.1se"))[1], numeric(1))
  }
  mean(intercepts)
}

#' XGBDRRegressor: DR-Learner with XGBoost (causalml-style; see \code{\link{DRLearner}})
#' @param control_name value for control group (0 or "control")
#' @param n_fold number of folds for cross-fitting (default 3)
#' @param ate_alpha confidence level for ATE interval
#' @param seed random seed for fold splits
#' @param control_outcome_learner,treatment_outcome_learner,treatment_effect_learner optional; default "xgb"
#' @return
#' Object returned by \code{XGBDRRegressor}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # XGBDRRegressor(...)
#' }
#' @export
XGBDRRegressor <- function(control_name = 0, n_fold = 3L, ate_alpha = 0.05, seed = NULL,
                           control_outcome_learner = NULL, treatment_outcome_learner = NULL,
                           treatment_effect_learner = NULL) {
  DRLearner(learner = "xgb", control_outcome_learner = control_outcome_learner,
            treatment_outcome_learner = treatment_outcome_learner,
            treatment_effect_learner = treatment_effect_learner,
            control_name = control_name, n_fold = n_fold,
            ate_alpha = ate_alpha, seed = seed)
}

#' Estimate Average Treatment Effect from a fitted meta-learner or fit-predict
#' @param obj meta-learner object (SLearner, TLearner, etc.) or character "S","T","X","R","DR"
#' @param X covariates
#' @param treatment treatment vector
#' @param y outcome
#' @param p propensity (optional, for X/R/DR)
#' @param return_ci return confidence interval
#' @param bootstrap_ci use bootstrap for CI instead of analytical SE
#' @param n_cores number of cores for bootstrap CI (1 = sequential)
#' @param pretrain if TRUE assume model already fitted
#' @param learner base learner for meta-learner if obj is character
#' @return ate (and ate_lb, ate_ub if return_ci). For DRLearner see \code{estimate_ate.DRLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # estimate_ate(...)
#' }
#' @export
estimate_ate <- function(obj, X, treatment, y, p = NULL, return_ci = TRUE,
                        bootstrap_ci = FALSE, n_bootstraps = 1000L, bootstrap_size = 10000L,
                        n_cores = 1L,
                        pretrain = FALSE, learner = "ranger", ...) {
  UseMethod("estimate_ate")
}

#' Default ATE estimation dispatcher
#'
#' @return
#' Object returned by \code{estimate_ate.default}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # estimate_ate.default(...)
#' }
#' @export
estimate_ate.default <- function(obj, X, treatment, y, p = NULL, return_ci = TRUE,
                                 bootstrap_ci = FALSE, n_bootstraps = 1000L, bootstrap_size = 10000L,
                                 n_cores = 1L,
                                 pretrain = FALSE, learner = "ranger", ...) {
  if (is.character(obj)) {
    obj <- switch(toupper(obj),
      S = SLearner(learner = learner),
      T = TLearner(learner = learner),
      X = XLearner(learner = learner),
      R = RLearner(learner = learner),
      DR = DRLearner(learner = learner),
      stop("obj must be S, T, X, R, DR or a fitted learner object"))
  }
  if (inherits(obj, "DRLearner"))
    return(estimate_ate.DRLearner(obj, X, treatment, y, p = p, return_ci = return_ci, bootstrap_ci = bootstrap_ci, n_bootstraps = n_bootstraps, bootstrap_size = bootstrap_size, n_cores = n_cores, pretrain = pretrain, ...))
  if (inherits(obj, "DMLearner"))
    return(estimate_ate.DMLearner(obj, X, treatment, y, p = p, return_ci = return_ci, bootstrap_ci = bootstrap_ci, n_bootstraps = n_bootstraps, bootstrap_size = bootstrap_size, n_cores = n_cores, pretrain = pretrain, ...))
  if (inherits(obj, "TMLELearner")) return(estimate_ate.TMLELearner(obj, X, treatment, y, p = p, return_ci = return_ci, ...))
  if (inherits(obj, "SLearner")) return(estimate_ate.SLearner(obj, X, treatment, y, p = p, return_ci = return_ci, bootstrap_ci = bootstrap_ci, n_bootstraps = n_bootstraps, bootstrap_size = bootstrap_size, n_cores = n_cores, pretrain = pretrain, ...))
  if (inherits(obj, "TLearner")) return(estimate_ate.TLearner(obj, X, treatment, y, p = p, return_ci = return_ci, bootstrap_ci = bootstrap_ci, n_bootstraps = n_bootstraps, bootstrap_size = bootstrap_size, n_cores = n_cores, pretrain = pretrain, ...))
  if (inherits(obj, "XLearner")) return(estimate_ate.XLearner(obj, X, treatment, y, p = p, return_ci = return_ci, bootstrap_ci = bootstrap_ci, n_bootstraps = n_bootstraps, bootstrap_size = bootstrap_size, n_cores = n_cores, pretrain = pretrain, ...))
  if (inherits(obj, "RLearner")) return(estimate_ate.RLearner(obj, X, treatment, y, p = p, return_ci = return_ci, bootstrap_ci = bootstrap_ci, n_bootstraps = n_bootstraps, bootstrap_size = bootstrap_size, n_cores = n_cores, pretrain = pretrain, ...))
  if (inherits(obj, "DomainAdaptationLearner")) return(estimate_ate.DomainAdaptationLearner(obj, X, treatment, y, p = p, return_ci = return_ci, bootstrap_ci = bootstrap_ci, n_bootstraps = n_bootstraps, bootstrap_size = bootstrap_size, n_cores = n_cores, pretrain = pretrain, ...))
  if (is.null(obj$models) && is.null(obj$model_0) && is.null(obj$model_tau) && is.null(obj$models_tau))
    obj <- fit(obj, X, treatment, y, p = p, ...)
  te <- predict(obj, X)
  if (is.matrix(te)) te <- te[, 1]
  ate <- mean(te, na.rm = TRUE)
  if (!return_ci) return(ate)
  z <- stats::qnorm(1 - 0.05 / 2)
  se <- sqrt(var(te, na.rm = TRUE) / length(te))
  list(ate = ate, ate_lb = ate - z * se, ate_ub = ate + z * se)
}

#' Estimate ATE for S-Learner (analytical SE or bootstrap). Like Python BaseSLearner.estimate_ate.
#' For LRSRegressor (learner = "lm") returns OLS coefficient and confint like Python LRSRegressor.estimate_ate.
#' @param n_cores number of cores for bootstrap CI (1 = sequential)
#' @return
#' Object returned by \code{estimate_ate.SLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # estimate_ate.SLearner(...)
#' }
#' @export
estimate_ate.SLearner <- function(obj, X, treatment, y, p = NULL, return_ci = TRUE,
                                 bootstrap_ci = FALSE, n_bootstraps = 1000L, bootstrap_size = 10000L,
                                 n_cores = 1L,
                                 pretrain = FALSE, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- length(y)
  if (!pretrain && is.null(obj$models)) obj <- fit(obj, X, treatment, y, p = p, ...)
  if (identical(obj$treatment_type, "continuous")) {
    te_v <- predict(obj, X, verbose = FALSE)
    if (is.matrix(te_v)) te_v <- te_v[, 1]
    ate <- mean(te_v, na.rm = TRUE)
    z <- stats::qnorm(1 - obj$ate_alpha / 2)
    se <- sqrt(stats::var(te_v, na.rm = TRUE) / n)
    if (!return_ci) return(list(ate = ate))
    if (!bootstrap_ci) return(list(ate = ate, ate_lb = ate - z * se, ate_ub = ate + z * se))
    boot_one <- function(b) {
      idx <- sample(n, min(bootstrap_size, n), replace = TRUE)
      obj_b <- SLearner(learner = obj$learner, control_name = obj$control_name, ate_alpha = obj$ate_alpha,
                        treatment_type = obj$treatment_type, dose_values = obj$dose_values)
      obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx], ...)
      mean(predict(obj_b, X), na.rm = TRUE)
    }
    res_boot <- parallel_lapply(seq_len(n_bootstraps), boot_one, n_cores = n_cores)
    ate_boot <- unlist(res_boot)
    return(list(ate = ate, ate_lb = quantile(ate_boot, probs = obj$ate_alpha / 2),
                 ate_ub = quantile(ate_boot, probs = 1 - obj$ate_alpha / 2)))
  }
  if (identical(obj$learner, "lm") && !is.null(obj$models)) {
    m <- obj$models[[1]]$model
    if (inherits(m, "lm")) {
      ci <- stats::confint(m, "w", level = 1 - obj$ate_alpha)
      return(list(ate = stats::coef(m)["w"], ate_lb = ci[1], ate_ub = ci[2]))
    }
  }
  out <- predict(obj, X, treatment = treatment, y = y, return_components = TRUE, verbose = FALSE)
  te <- out$te
  yhat_cs <- out$yhat_cs
  yhat_ts <- out$yhat_ts
  if (is.vector(te)) te <- matrix(te, ncol = 1L)
  t_groups <- obj$t_groups
  n_t <- length(t_groups)
  ate <- numeric(n_t)
  ate_lb <- numeric(n_t)
  ate_ub <- numeric(n_t)
  z <- stats::qnorm(1 - obj$ate_alpha / 2)
  for (i in seq_len(n_t)) {
    g <- t_groups[i]
    ate[i] <- mean(te[, i], na.rm = TRUE)
    mask <- (treatment == g) | (treatment == obj$control_name)
    treatment_filt <- treatment[mask]
    y_filt <- y[mask]
    w <- as.integer(treatment_filt == g)
    prob_treatment <- mean(w)
    yhat_c <- yhat_cs[[as.character(g)]][mask]
    yhat_t <- yhat_ts[[as.character(g)]][mask]
    var_c <- var(y_filt[w == 0] - yhat_c[w == 0])
    var_t <- var(y_filt[w == 1] - yhat_t[w == 1])
    var_tau <- var(yhat_t - yhat_c)
    se <- sqrt((var_c / (1 - prob_treatment) + var_t / prob_treatment + var_tau) / sum(mask))
    if (!is.finite(se) || se <= 0) se <- sqrt(var(te[, i], na.rm = TRUE) / n)
    ate_lb[i] <- ate[i] - z * se
    ate_ub[i] <- ate[i] + z * se
  }
  if (!return_ci) return(list(ate = ate))
  if (!bootstrap_ci) return(list(ate = ate, ate_lb = ate_lb, ate_ub = ate_ub))
  size <- min(as.integer(bootstrap_size), n)
  boot_one <- function(b) {
    idx <- sample(n, size = size, replace = TRUE)
    obj_b <- SLearner(learner = obj$learner, control_name = obj$control_name, ate_alpha = obj$ate_alpha,
                      treatment_type = obj$treatment_type, dose_values = obj$dose_values)
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx], ...)
    te_b <- predict(obj_b, X)
    if (is.matrix(te_b)) colMeans(te_b) else mean(te_b)
  }
  res_boot <- parallel_lapply(seq_len(n_bootstraps), boot_one, n_cores = n_cores)
  ate_boot <- matrix(NA_real_, n_t, n_bootstraps)
  for (b in seq_len(n_bootstraps)) ate_boot[, b] <- res_boot[[b]]
  ate_lower <- apply(ate_boot, 1, quantile, probs = obj$ate_alpha / 2)
  ate_upper <- apply(ate_boot, 1, quantile, probs = 1 - obj$ate_alpha / 2)
  list(ate = ate, ate_lb = ate_lower, ate_ub = ate_upper)
}

#' Estimate ATE for T-Learner. Like Python BaseTLearner.estimate_ate.
#' @param n_cores number of cores for bootstrap CI (1 = sequential)
#' @return
#' Object returned by \code{estimate_ate.TLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # estimate_ate.TLearner(...)
#' }
#' @export
estimate_ate.TLearner <- function(obj, X, treatment, y, p = NULL, return_ci = TRUE,
                                 bootstrap_ci = FALSE, n_bootstraps = 1000L, bootstrap_size = 10000L,
                                 n_cores = 1L,
                                 pretrain = FALSE, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- length(y)
  if (!pretrain && is.null(obj$model_0)) obj <- fit(obj, X, treatment, y, p = p, ...)
  out <- predict(obj, X, treatment = treatment, y = y, return_components = TRUE, verbose = FALSE)
  te <- out$te
  yhat_cs <- out$yhat_cs
  yhat_ts <- out$yhat_ts
  if (is.vector(te)) te <- matrix(te, ncol = 1L)
  t_groups <- obj$t_groups
  if (is.null(t_groups)) t_groups <- 1
  n_t <- length(t_groups)
  ate <- numeric(n_t)
  ate_lb <- numeric(n_t)
  ate_ub <- numeric(n_t)
  z <- stats::qnorm(1 - obj$ate_alpha / 2)
  for (i in seq_len(n_t)) {
    g <- t_groups[i]
    ate[i] <- mean(te[, i], na.rm = TRUE)
    if (identical(obj$treatment_type, "continuous")) {
      w_full <- as.integer(treatment > obj$dose_split)
      prob_treatment <- mean(w_full)
      yhat_c <- yhat_cs[[as.character(g)]]
      yhat_t <- yhat_ts[[as.character(g)]]
      var_c <- stats::var(y[w_full == 0] - yhat_c[w_full == 0])
      var_t <- stats::var(y[w_full == 1] - yhat_t[w_full == 1])
      var_tau <- stats::var(yhat_t - yhat_c)
      se <- sqrt((var_c / (1 - prob_treatment) + var_t / prob_treatment + var_tau) / n)
    } else {
      mask <- (treatment == g) | (treatment == obj$control_name)
      treatment_filt <- treatment[mask]
      y_filt <- y[mask]
      w <- as.integer(treatment_filt == g)
      prob_treatment <- mean(w)
      yhat_c <- yhat_cs[[as.character(g)]][mask]
      yhat_t <- yhat_ts[[as.character(g)]][mask]
      var_c <- var(y_filt[w == 0] - yhat_c[w == 0])
      var_t <- var(y_filt[w == 1] - yhat_t[w == 1])
      var_tau <- var(yhat_t - yhat_c)
      se <- sqrt((var_c / (1 - prob_treatment) + var_t / prob_treatment + var_tau) / sum(mask))
    }
    if (!is.finite(se) || se <= 0) se <- sqrt(var(te[, i], na.rm = TRUE) / n)
    ate_lb[i] <- ate[i] - z * se
    ate_ub[i] <- ate[i] + z * se
  }
  if (!return_ci) return(list(ate = ate))
  if (!bootstrap_ci) return(list(ate = ate, ate_lb = ate_lb, ate_ub = ate_ub))
  boot_one <- function(b) {
    idx <- sample(n, min(bootstrap_size, n), replace = TRUE)
    obj_b <- TLearner(learner = obj$learner, control_name = obj$control_name, ate_alpha = obj$ate_alpha,
                      treatment_type = obj$treatment_type, dose_split = obj$dose_split)
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx], ...)
    te_b <- predict(obj_b, X)
    if (is.matrix(te_b)) colMeans(te_b) else mean(te_b)
  }
  res_boot <- parallel_lapply(seq_len(n_bootstraps), boot_one, n_cores = n_cores)
  ate_boot <- matrix(NA_real_, n_t, n_bootstraps)
  for (b in seq_len(n_bootstraps)) ate_boot[, b] <- res_boot[[b]]
  list(ate = ate, ate_lb = apply(ate_boot, 1, quantile, probs = obj$ate_alpha / 2),
       ate_ub = apply(ate_boot, 1, quantile, probs = 1 - obj$ate_alpha / 2))
}

#' Estimate ATE for X-Learner. Like Python BaseXLearner.estimate_ate.
#' @param n_cores number of cores for bootstrap CI (1 = sequential)
#' @return
#' Object returned by \code{estimate_ate.XLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # estimate_ate.XLearner(...)
#' }
#' @export
estimate_ate.XLearner <- function(obj, X, treatment, y, p = NULL, return_ci = TRUE,
                                 bootstrap_ci = FALSE, n_bootstraps = 1000L, bootstrap_size = 10000L,
                                 n_cores = 1L,
                                 pretrain = FALSE, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- length(y)
  if (!pretrain && is.null(obj$model_tau0)) obj <- fit(obj, X, treatment, y, p = p, ...)
  out <- predict(obj, X, p = obj$p, return_components = TRUE)
  te <- out$te
  dhat_cs <- out$dhat_cs
  dhat_ts <- out$dhat_ts
  t_groups <- obj$t_groups
  g <- t_groups[1L]
  ate <- mean(te, na.rm = TRUE)
  if (identical(obj$treatment_type, "continuous")) {
    w_full <- as.integer(treatment > obj$dose_split)
    prob_treatment <- mean(w_full)
    dhat_c <- dhat_cs[[as.character(g)]]
    dhat_t <- dhat_ts[[as.character(g)]]
    p_fu <- if (is.list(obj$p)) obj$p[[as.character(g)]] else obj$p
    if (is.null(p_fu) || length(p_fu) != n) p_fu <- rep(0.5, n)
    se <- sqrt((obj$vars_t / prob_treatment + obj$vars_c / (1 - prob_treatment) +
      stats::var(p_fu * dhat_c + (1 - p_fu) * dhat_t)) / n)
  } else {
    mask <- (treatment == g) | (treatment == obj$control_name)
    w <- as.integer(treatment[mask] == g)
    prob_treatment <- mean(w)
    dhat_c <- dhat_cs[[as.character(g)]][mask]
    dhat_t <- dhat_ts[[as.character(g)]][mask]
    p_filt <- if (is.list(obj$p)) obj$p[[as.character(g)]][mask] else obj$p[mask]
    if (is.null(p_filt) || length(p_filt) != sum(mask)) p_filt <- rep(0.5, sum(mask))
    se <- sqrt((obj$vars_t / prob_treatment + obj$vars_c / (1 - prob_treatment) + var(p_filt * dhat_c + (1 - p_filt) * dhat_t)) / sum(mask))
  }
  z <- stats::qnorm(1 - obj$ate_alpha / 2)
  ate_lb <- ate - z * se
  ate_ub <- ate + z * se
  if (!return_ci) return(list(ate = ate))
  if (!bootstrap_ci) return(list(ate = ate, ate_lb = ate_lb, ate_ub = ate_ub))
  boot_one <- function(b) {
    idx <- sample(n, min(bootstrap_size, n), replace = TRUE)
    obj_b <- XLearner(learner = obj$learner, control_name = obj$control_name, ate_alpha = obj$ate_alpha,
                      treatment_type = obj$treatment_type, dose_split = obj$dose_split)
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx], p = if (is.null(p)) NULL else p[idx], ...)
    mean(predict(obj_b, X), na.rm = TRUE)
  }
  res_boot <- parallel_lapply(seq_len(n_bootstraps), boot_one, n_cores = n_cores)
  ate_boot <- unlist(res_boot)
  list(ate = ate, ate_lb = quantile(ate_boot, probs = obj$ate_alpha / 2), ate_ub = quantile(ate_boot, probs = 1 - obj$ate_alpha / 2))
}

#' Estimate ATE for R-Learner. Like Python BaseRLearner.estimate_ate (analytical SE).
#' @param n_cores number of cores for bootstrap CI (1 = sequential)
#' @return
#' Object returned by \code{estimate_ate.RLearner}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # estimate_ate.RLearner(...)
#' }
#' @export
estimate_ate.RLearner <- function(obj, X, treatment, y, p = NULL, return_ci = TRUE,
                                 bootstrap_ci = FALSE, n_bootstraps = 1000L, bootstrap_size = 10000L,
                                 n_cores = 1L,
                                 pretrain = FALSE, sample_weight = NULL, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- length(y)
  if (!pretrain && is.null(obj$model_tau)) obj <- fit(obj, X, treatment, y, p = p, sample_weight = sample_weight, ...)
  te <- predict(obj, X)
  if (is.vector(te)) te <- matrix(te, ncol = 1L)
  g <- obj$t_groups[1L]
  ate <- mean(te[, 1], na.rm = TRUE)
  if (identical(obj$treatment_type, "continuous")) {
    se <- sqrt(stats::var(te[, 1], na.rm = TRUE) / n)
  } else {
    w <- as.integer(treatment == g)
    prob_treatment <- mean(w)
    se <- sqrt((obj$vars_t / prob_treatment + obj$vars_c / (1 - prob_treatment) + var(te[, 1], na.rm = TRUE)) / n)
  }
  z <- stats::qnorm(1 - obj$ate_alpha / 2)
  ate_lb <- ate - z * se
  ate_ub <- ate + z * se
  if (!return_ci) return(list(ate = ate))
  if (!bootstrap_ci) return(list(ate = ate, ate_lb = ate_lb, ate_ub = ate_ub))
  boot_one <- function(b) {
    idx <- sample(n, min(bootstrap_size, n), replace = TRUE)
    obj_b <- RLearner(learner = obj$learner, control_name = obj$control_name, n_fold = obj$n_fold, ate_alpha = obj$ate_alpha,
                      treatment_type = obj$treatment_type)
    obj_b <- fit(obj_b, X[idx, , drop = FALSE], treatment[idx], y[idx], p = if (is.null(p)) NULL else p[idx], verbose = FALSE, ...)
    mean(predict(obj_b, X), na.rm = TRUE)
  }
  res_boot <- parallel_lapply(seq_len(n_bootstraps), boot_one, n_cores = n_cores)
  ate_boot <- unlist(res_boot)
  list(ate = ate, ate_lb = quantile(ate_boot, probs = obj$ate_alpha / 2), ate_ub = quantile(ate_boot, probs = 1 - obj$ate_alpha / 2))
}
