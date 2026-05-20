# CausalML-R: Propensity score estimation (mirrors Python causalml.inference.tree.utils)

# Default when NULL (used in closures below)
`%||%` <- function(x, y) if (is.null(x)) y else x

# Internal: predict from isoreg fit with clipping (like sklearn IsotonicRegression)
# @param iso result of stats::isoreg(x, y)
# @param new_ps new propensity values to transform
# @param y_min,y_max clip output to [y_min, y_max]
# @noRd
.predict_isotonic <- function(iso, new_ps, y_min, y_max) {
  ox <- iso$x[iso$ord]
  yf <- iso$yf
  lo <- max(y_min, min(yf, na.rm = TRUE), na.rm = TRUE)
  hi <- min(y_max, max(yf, na.rm = TRUE), na.rm = TRUE)
  out <- approx(ox, yf, xout = new_ps, method = "constant", f = 0,
                yleft = lo, yright = hi, rule = 2)$y
  pmax(y_min, pmin(y_max, out))
}

#' Propensity model object (abstract interface)
#'
#' Objects have \code{fit(X, y)}, \code{predict(X)}, and \code{fit_predict(X, y)}.
#' \code{clip_bounds} and \code{calibrate} control clipping and isotonic calibration.
#' @param clip_bounds length-2 numeric: (lower, upper) for clipping propensity scores; use \code{0 < lower < upper < 1}.
#' @param calibrate logical: whether to calibrate propensity scores with isotonic regression.
#' @param ... passed to the underlying model (e.g. \code{n_fold}, \code{random_state}).
#' @return Object with \code{fit}, \code{predict}, \code{fit_predict} methods.
#' @name propensity_model
NULL

#' Logistic regression propensity model (elastic net)
#'
#' Propensity model using \code{glmnet::cv.glmnet} with binomial family and elastic net
#' (alpha in (0,1)). Mirrors Python \code{LogisticRegressionPropensityModel}.
#' @param clip_bounds length-2 numeric: bounds for clipping (default \code{c(1e-3, 1 - 1e-3)}).
#' @param calibrate logical: whether to calibrate with isotonic regression (default \code{TRUE}).
#' @param n_fold integer: number of CV folds for lambda selection (default 4).
#' @param alpha numeric: elastic net mixing (0=ridge, 1=lasso; default 0.5).
#' @param ... passed to \code{glmnet::cv.glmnet}.
#' @return Propensity model object with \code{fit}, \code{predict}, \code{fit_predict}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # propensity_model_logistic_regression(...)
#' }
#' @export
propensity_model_logistic_regression <- function(clip_bounds = c(1e-3, 1 - 1e-3),
                                                calibrate = TRUE,
                                                n_fold = 4L,
                                                alpha = 0.5,
                                                ...) {
  model_kwargs <- list(n_fold = n_fold, alpha = alpha, ...)
  internal <- new.env()
  internal$model <- NULL
  internal$calibrator <- NULL

  obj <- list(
    clip_bounds = clip_bounds,
    calibrate = calibrate,
    model_kwargs = model_kwargs,
    internal = internal
  )

  obj$fit <- function(X, y) {
    if (inherits(X, "data.frame")) X <- as.matrix(X)
    y <- as.numeric(y)
    n_fold <- model_kwargs$n_fold %||% 4L
    alpha <- model_kwargs$alpha %||% 0.5
    other <- model_kwargs; other$n_fold <- NULL; other$alpha <- NULL
    internal$model <- do.call(glmnet::cv.glmnet, c(
      list(x = X, y = y, family = "binomial", nfolds = n_fold, alpha = alpha),
      other
    ))
    if (calibrate) {
      p_train <- as.vector(predict(internal$model, newx = X, s = "lambda.1se", type = "response"))
      internal$calibrator <- isoreg(p_train, y)
    }
    invisible(obj)
  }

  obj$predict <- function(X) {
    if (is.null(internal$model)) stop("Model not fitted; call fit() first.")
    if (inherits(X, "data.frame")) X <- as.matrix(X)
    p <- as.vector(predict(internal$model, newx = X, s = "lambda.1se", type = "response"))
    if (calibrate && !is.null(internal$calibrator)) {
      p <- .predict_isotonic(internal$calibrator, p, clip_bounds[1L], clip_bounds[2L])
    }
    pmax(clip_bounds[1L], pmin(clip_bounds[2L], p))
  }

  obj$fit_predict <- function(X, y) {
    obj$fit(X, y)
    obj$predict(X)
  }

  class(obj) <- c("propensity_model_logistic", "propensity_model")
  obj
}

#' Elastic net propensity model
#'
#' Alias for \code{propensity_model_logistic_regression}. Mirrors Python \code{ElasticNetPropensityModel}.
#' @param clip_bounds length-2 numeric: bounds for clipping.
#' @param calibrate logical: whether to calibrate.
#' @param ... passed to \code{propensity_model_logistic_regression}.
#' @return Propensity model object.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # propensity_model_elastic_net(...)
#' }
#' @export
propensity_model_elastic_net <- function(clip_bounds = c(1e-3, 1 - 1e-3),
                                         calibrate = TRUE,
                                         ...) {
  propensity_model_logistic_regression(clip_bounds = clip_bounds, calibrate = calibrate, ...)
}

#' Gradient boosted propensity model (XGBoost)
#'
#' Propensity model using \code{xgboost::xgb.train} / \code{xgboost::xgb.DMatrix} or
#' \code{xgboost::xgb.Booster} for binary classification. Optional early stopping.
#' Requires the \code{xgboost} package. Mirrors Python \code{GradientBoostedPropensityModel}.
#' @param clip_bounds length-2 numeric: bounds for clipping.
#' @param calibrate logical: whether to calibrate (default \code{TRUE}).
#' @param early_stop logical: whether to use a validation set for early stopping (default \code{FALSE}).
#' @param stop_val_size numeric: fraction of data for validation when \code{early_stop=TRUE} (default 0.2).
#' @param max_depth,learning_rate,n_estimators,colsample_bytree,random_state passed to xgboost; defaults mirror Python.
#' @param ... other arguments passed to \code{xgboost::xgb.train} or \code{xgboost::xgb.Booster}.
#' @return Propensity model object with \code{fit}, \code{predict}, \code{fit_predict}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # propensity_model_gradient_boosted(...)
#' }
#' @export
propensity_model_gradient_boosted <- function(clip_bounds = c(1e-3, 1 - 1e-3),
                                              calibrate = TRUE,
                                              early_stop = FALSE,
                                              stop_val_size = 0.2,
                                              max_depth = 8L,
                                              learning_rate = 0.1,
                                              n_estimators = 100L,
                                              colsample_bytree = 0.8,
                                              random_state = 42L,
                                              ...) {
  if (!requireNamespace("xgboost", quietly = TRUE))
    stop("Package 'xgboost' is required for propensity_model_gradient_boosted.")
  model_kwargs <- list(
    max_depth = max_depth,
    learning_rate = learning_rate,
    n_estimators = n_estimators,
    colsample_bytree = colsample_bytree,
    random_state = random_state,
    ...
  )
  internal <- new.env()
  internal$model <- NULL
  internal$calibrator <- NULL

  obj <- list(
    clip_bounds = clip_bounds,
    calibrate = calibrate,
    early_stop = early_stop,
    stop_val_size = stop_val_size,
    model_kwargs = model_kwargs,
    internal = internal
  )

  obj$fit <- function(X, y) {
    if (inherits(X, "data.frame")) X <- as.matrix(X)
    y <- as.numeric(y)
    set.seed(model_kwargs$random_state %||% 42L)
    params <- list(
      max_depth = model_kwargs$max_depth %||% 8L,
      eta = model_kwargs$learning_rate %||% 0.1,
      objective = "binary:logistic",
      colsample_bytree = model_kwargs$colsample_bytree %||% 0.8
    )
    nrounds <- model_kwargs$n_estimators %||% 100L
    other <- setdiff(names(model_kwargs), c("max_depth", "learning_rate", "n_estimators", "colsample_bytree", "random_state"))
    for (k in other) if (!is.null(model_kwargs[[k]])) params[[k]] <- model_kwargs[[k]]

    if (early_stop) {
      n <- nrow(X)
      val_idx <- sample.int(n, ceiling(n * stop_val_size))
      train_idx <- setdiff(seq_len(n), val_idx)
      dtrain <- xgboost::xgb.DMatrix(X[train_idx, , drop = FALSE], label = y[train_idx])
      dval <- xgboost::xgb.DMatrix(X[val_idx, , drop = FALSE], label = y[val_idx])
      watchlist <- list(train = dtrain, validation = dval)
      internal$model <- xgboost::xgb.train(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        watchlist = watchlist,
        early_stopping_rounds = 10L,
        verbose = 0L
      )
      if (calibrate) {
        p_train <- predict(internal$model, xgboost::xgb.DMatrix(X, label = y))
        internal$calibrator <- isoreg(p_train, y)
      }
    } else {
      dtrain <- xgboost::xgb.DMatrix(X, label = y)
      internal$model <- xgboost::xgb.train(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        verbose = 0L
      )
      if (calibrate) {
        p_train <- predict(internal$model, dtrain)
        internal$calibrator <- isoreg(p_train, y)
      }
    }
    invisible(obj)
  }

  obj$predict <- function(X) {
    if (is.null(internal$model)) stop("Model not fitted; call fit() first.")
    if (inherits(X, "data.frame")) X <- as.matrix(X)
    d <- xgboost::xgb.DMatrix(X)
    p <- predict(internal$model, d)
    if (calibrate && !is.null(internal$calibrator)) {
      p <- .predict_isotonic(internal$calibrator, p, clip_bounds[1L], clip_bounds[2L])
    }
    pmax(clip_bounds[1L], pmin(clip_bounds[2L], p))
  }

  obj$fit_predict <- function(X, y) {
    obj$fit(X, y)
    obj$predict(X)
  }

  class(obj) <- c("propensity_model_gradient_boosted", "propensity_model")
  obj
}

#' Compute propensity scores (train and optionally predict)
#'
#' Fits a propensity model on \code{(X, treatment)} and returns propensity scores
#' for \code{X_pred} (or \code{X}). If \code{p_model} is not provided, uses
#' \code{propensity_model_elastic_net}. Mirrors Python \code{compute_propensity_score}.
#' @param X matrix or data.frame: features for training.
#' @param treatment numeric or factor: binary treatment vector (0/1) for training.
#' @param p_model optional: a propensity model object with \code{fit()} and \code{predict()};
#'   or any model with \code{predict(..., type="response")} or \code{predict_proba}.
#'   If \code{NULL}, an elastic net propensity model is created and fitted.
#' @param X_pred matrix or data.frame: features for prediction; if \code{NULL}, uses \code{X}.
#' @param treatment_pred optional: treatment vector for calibration target; if \code{NULL}, uses \code{treatment}.
#' @param calibrate_p logical: whether to calibrate propensity scores (default \code{TRUE});
#'   only used when \code{p_model} is \code{NULL} (the default model is created with \code{calibrate=calibrate_p}).
#' @param clip_bounds length-2 numeric: bounds for clipping (default \code{c(1e-3, 1 - 1e-3)}).
#' @return List with \code{p}: numeric vector of propensity scores, and \code{p_model}: the fitted
#'   propensity model (either the provided one or the created elastic net model).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # compute_propensity_score(...)
#' }
#' @export
compute_propensity_score <- function(X,
                                    treatment,
                                    p_model = NULL,
                                    X_pred = NULL,
                                    treatment_pred = NULL,
                                    calibrate_p = TRUE,
                                    clip_bounds = c(1e-3, 1 - 1e-3)) {
  if (is.null(treatment_pred)) treatment_pred <- treatment
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  X_pred <- X_pred %||% X
  if (inherits(X_pred, "data.frame")) X_pred <- as.matrix(X_pred)
  treatment <- as.numeric(treatment)

  if (is.null(p_model)) {
    p_model <- propensity_model_elastic_net(clip_bounds = clip_bounds, calibrate = calibrate_p)
  }

  # Propensity model objects (our S3-style objects with fit/predict)
  if (inherits(p_model, "propensity_model")) {
    p_model$fit(X, treatment)
    p <- p_model$predict(X_pred)
    return(list(p = p, p_model = p_model))
  }

  # Pre-fitted cv.glmnet: predict only (no refit)
  if (inherits(p_model, "cv.glmnet")) {
    p <- as.vector(predict(p_model, newx = X_pred, s = "lambda.1se", type = "response"))
    p <- pmax(clip_bounds[1L], pmin(clip_bounds[2L], p))
    return(list(p = p, p_model = p_model))
  }

  # Custom model with fit(): fit then predict
  if (is.function(p_model$fit)) {
    p_model$fit(X, treatment)
  }
  p <- NULL
  tryCatch({
    pp <- p_model$predict(X_pred)
    if (is.matrix(pp) && ncol(pp) >= 2L) p <- pp[, 2L]
    else if (is.matrix(pp) && ncol(pp) == 1L) p <- as.vector(pp)
    else p <- as.vector(pp)
  }, error = function(e) {
    tryCatch({
      pp <- predict(p_model, X_pred, type = "response")
      if (is.matrix(pp) && ncol(pp) >= 2L) p <<- as.vector(pp[, 2L])
      else p <<- as.vector(pp)
    }, error = function(e2) {
      p <<- as.vector(predict(p_model, X_pred))
    })
  })
  if (!is.null(p)) {
    p <- pmax(clip_bounds[1L], pmin(clip_bounds[2L], as.numeric(p)))
    return(list(p = p, p_model = p_model))
  }
  stop("Could not obtain propensity predictions from p_model.")
}

# Backward compatibility: simple functions that return a vector of propensity scores

#' Elastic net propensity score model (glmnet)
#'
#' Fits \code{glmnet::cv.glmnet} with binomial family and returns propensity scores.
#' For clip_bounds and calibration use \code{propensity_model_elastic_net()} and
#' \code{compute_propensity_score()} instead.
#' @param X covariate matrix or data.frame
#' @param w treatment indicator (0/1)
#' @param n_fold number of folds for cv.glmnet
#' @param lambda optional lambda; if NULL uses lambda.1se
#' @param ... passed to cv.glmnet or glmnet
#' @return vector of propensity scores (probability of w=1)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # propensity_glmnet(...)
#' }
#' @export
propensity_glmnet <- function(X, w, n_fold = 5, lambda = NULL, ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  w <- as.numeric(w)
  fit <- glmnet::cv.glmnet(X, w, family = "binomial", nfolds = n_fold, ...)
  if (is.null(lambda)) lambda <- fit$lambda.1se
  p <- predict(fit, newx = X, s = lambda, type = "response")
  as.vector(p)
}

#' Logistic regression propensity score
#'
#' Simple GLM propensity model. For elastic net + calibration use
#' \code{propensity_model_logistic_regression()} or \code{compute_propensity_score()}.
#' @param X covariate matrix or data.frame (will add intercept)
#' @param w treatment indicator
#' @param ... passed to \code{glm}
#' @return vector of propensity scores
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # propensity_glm(...)
#' }
#' @export
propensity_glm <- function(X, w, ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  df <- data.frame(w = as.numeric(w), X = X)
  m <- stats::glm(w ~ ., data = df, family = stats::binomial, ...)
  as.vector(stats::predict(m, type = "response"))
}

