# CausalML-R: Automated Machine Learning support for causal estimators
# Port of EconML econml.automated_ml: automate model selection for Y, T, and nuisance
# models in causal inference estimators (local AutoML; Python version uses Azure ML).
# Reference: EconML automated_ml module.

# ------------------------------------------------------------------------------
# Model whitelists (aligned with EconML constraints)
# ------------------------------------------------------------------------------

#' Linear models allowed when linear_model_required = TRUE
#' @noRd
LINEAR_MODELS_SET <- c("lm", "glmnet")

#' Models that support sample/case weights (for sample_weights_required = TRUE)
#' @noRd
SAMPLE_WEIGHTS_MODELS_SET <- c("lm", "glmnet", "ranger", "rpart")

# ------------------------------------------------------------------------------
# Workspace configuration (stub; Python version uses Azure ML)
# ------------------------------------------------------------------------------

#' Set configuration for AutomatedML (local experiments)
#'
#' In the Python EconML version, this configures the Azure ML workspace. In this R
#' port we use local runs only; this function optionally sets a directory for
#' saving experiment metadata and logs.
#'
#' @param experiment_dir character or NULL. If provided, directory path for
#'   saving AutoML run metadata (e.g. best model name, CV scores). If NULL,
#'   no persistent config is written.
#' @param create_dir logical. If TRUE and \code{experiment_dir} is set, create
#'   the directory if it does not exist.
#' @return Invisible NULL. Optionally creates \code{experiment_dir} and saves
#'   a small config list there for use by \code{AutomatedMLModel}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # set_automated_ml_workspace(...)
#' }
#' @export
set_automated_ml_workspace <- function(experiment_dir = NULL, create_dir = FALSE) {
  if (is.null(experiment_dir)) {
    message("No experiment_dir set; AutoML runs will not persist config.")
    return(invisible(NULL))
  }
  if (create_dir && !dir.exists(experiment_dir)) {
    dir.create(experiment_dir, recursive = TRUE)
    message("Created experiment directory: ", experiment_dir)
  }
  if (!dir.exists(experiment_dir)) {
    stop("experiment_dir does not exist. Set create_dir = TRUE to create it.")
  }
  config <- list(experiment_dir = experiment_dir, timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  saveRDS(config, file.path(experiment_dir, "automated_ml_config.rds"))
  message("Workspace configuration saved to ", experiment_dir)
  invisible(NULL)
}

# ------------------------------------------------------------------------------
# EconAutoMLConfig: config object for AutoML with EconML-style guards
# ------------------------------------------------------------------------------

#' Create an EconML-style AutoML config for causal estimators
#'
#' Configuration for automated model selection when fitting nuisance models (Y, T,
#' etc.). Constraints ensure only models suitable for causal inference are
#' considered (e.g. linear-only or sample-weight support).
#'
#' @param sample_weights_required logical. If TRUE, only models that support
#'   \code{sample_weight} / \code{case.weights} are tried (e.g. for IPW/DR).
#' @param linear_model_required logical. If TRUE, only linear models are tried.
#' @param show_output logical. If TRUE, print progress and best model during fit.
#' @param task character. \code{"regression"} or \code{"classification"}.
#' @param n_folds integer. Number of CV folds for model selection (default 5).
#' @param time_budget_sec numeric or NULL. Max seconds for tuning (optional;
#'   not all backends respect this).
#' @param experiment_name_prefix character. Prefix for experiment names (max 18 chars).
#' @param ... additional options (stored in config for future use).
#' @return List of class \code{EconAutoMLConfig} with the above fields and
#'   \code{whitelist_models} set from the constraints.
#' @references EconML \code{econml.automated_ml.EconAutoMLConfig}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # EconAutoMLConfig(...)
#' }
#' @export
EconAutoMLConfig <- function(sample_weights_required = FALSE,
                             linear_model_required = FALSE,
                             show_output = FALSE,
                             task = c("regression", "classification"),
                             n_folds = 5L,
                             time_budget_sec = NULL,
                             experiment_name_prefix = "aml_experiment",
                             ...) {
  task <- match.arg(task)
  whitelist_models <- NULL
  if (linear_model_required && sample_weights_required) {
    whitelist_models <- intersect(LINEAR_MODELS_SET, SAMPLE_WEIGHTS_MODELS_SET)
  } else if (linear_model_required) {
    whitelist_models <- LINEAR_MODELS_SET
  } else if (sample_weights_required) {
    whitelist_models <- SAMPLE_WEIGHTS_MODELS_SET
  } else {
    whitelist_models <- unique(c(LINEAR_MODELS_SET, SAMPLE_WEIGHTS_MODELS_SET))
  }
  out <- list(
    sample_weights_required = sample_weights_required,
    linear_model_required = linear_model_required,
    show_output = show_output,
    task = task,
    n_folds = as.integer(n_folds),
    time_budget_sec = time_budget_sec,
    experiment_name_prefix = substr(experiment_name_prefix, 1L, 18L),
    whitelist_models = whitelist_models,
    extra = list(...)
  )
  class(out) <- "EconAutoMLConfig"
  out
}

# ------------------------------------------------------------------------------
# Inner single-output model: fit one learner with optional sample weights
# ------------------------------------------------------------------------------

#' Fit a single learner (regression or classification) with optional weights
#' @param X matrix or data.frame of features
#' @param y numeric vector (regression) or factor (classification)
#' @param learner character, one of whitelist_models
#' @param sample_weight numeric vector or NULL
#' @param task "regression" or "classification"
#' @return fitted model list with \code{predict} and optionally \code{predict_proba}
#' @noRd
fit_single_automl_learner <- function(X, y, learner, sample_weight = NULL, task = "regression") {
  n <- length(y)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  df <- as.data.frame(X)
  df$y <- y
  if (task == "classification" && !is.factor(y)) df$y <- as.factor(y)

  if (learner == "lm") {
    if (task == "classification") {
      if (!is.null(sample_weight)) {
        m <- stats::glm(y ~ ., data = df, family = stats::binomial, weights = sample_weight)
      } else {
        m <- stats::glm(y ~ ., data = df, family = stats::binomial)
      }
    } else {
      if (!is.null(sample_weight)) {
        m <- stats::lm(y ~ ., data = df, weights = sample_weight)
      } else {
        m <- stats::lm(y ~ ., data = df)
      }
    }
    return(list(type = "lm", model = m, task = task))
  }

  if (learner == "glmnet") {
    x_mat <- as.matrix(df[, -which(names(df) == "y"), drop = FALSE])
    if (task == "classification") {
      fam <- "binomial"
      if (nlevels(df$y) > 2) fam <- "multinomial"
      fit <- glmnet::cv.glmnet(x_mat, df$y, family = fam, weights = sample_weight, nfolds = 5)
    } else {
      fit <- glmnet::cv.glmnet(x_mat, df$y, weights = sample_weight, nfolds = 5)
    }
    return(list(type = "glmnet", model = fit, task = task))
  }

  if (learner == "ranger") {
    if (task == "classification") {
      if (!is.null(sample_weight)) {
        m <- ranger::ranger(y ~ ., data = df, probability = TRUE, case.weights = sample_weight)
      } else {
        m <- ranger::ranger(y ~ ., data = df, probability = TRUE)
      }
    } else {
      if (!is.null(sample_weight)) {
        m <- ranger::ranger(y ~ ., data = df, case.weights = sample_weight)
      } else {
        m <- ranger::ranger(y ~ ., data = df)
      }
    }
    return(list(type = "ranger", model = m, task = task))
  }

  if (learner == "rpart") {
    if (task == "classification") {
      method <- "class"
      if (!is.null(sample_weight)) {
        m <- rpart::rpart(y ~ ., data = df, weights = sample_weight, method = method)
      } else {
        m <- rpart::rpart(y ~ ., data = df, method = method)
      }
    } else {
      if (!is.null(sample_weight)) {
        m <- rpart::rpart(y ~ ., data = df, weights = sample_weight)
      } else {
        m <- rpart::rpart(y ~ ., data = df)
      }
    }
    return(list(type = "rpart", model = m, task = task))
  }

  stop("Unknown learner: ", learner)
}

#' Predict from a single fitted learner
#' @noRd
predict_single_automl_learner <- function(obj, X) {
  df <- as.data.frame(X)
  if (obj$type == "lm") {
    if (obj$task == "classification") {
      return(as.numeric(predict(obj$model, newdata = df, type = "response")))
    }
    return(as.numeric(predict(obj$model, newdata = df)))
  }
  if (obj$type == "glmnet") {
    x_mat <- as.matrix(df)
    if (obj$task == "classification") {
      return(as.numeric(predict(obj$model, newx = x_mat, s = "lambda.1se", type = "response")))
    }
    return(as.numeric(predict(obj$model, newx = x_mat, s = "lambda.1se")))
  }
  if (obj$type == "ranger") {
    pred <- predict(obj$model, data = df)
    if (obj$task == "classification" && !is.null(pred$predictions)) {
      if (is.matrix(pred$predictions)) return(pred$predictions[, 2]) else return(pred$predictions)
    }
    return(as.numeric(pred$predictions))
  }
  if (obj$type == "rpart") {
    if (obj$task == "classification") {
      return(as.numeric(predict(obj$model, newdata = df)[, 2]))
    }
    return(as.numeric(predict(obj$model, newdata = df)))
  }
  stop("Unknown type: ", obj$type)
}

#' Predict probabilities from a single fitted learner (classification only)
#' @noRd
predict_proba_single_automl_learner <- function(obj, X) {
  if (obj$task != "classification") {
    stop("predict_proba only for classification task")
  }
  df <- as.data.frame(X)
  if (obj$type == "lm") {
    p <- predict(obj$model, newdata = df, type = "response")
    return(cbind(1 - p, p))
  }
  if (obj$type == "glmnet") {
    x_mat <- as.matrix(df)
    p <- predict(obj$model, newx = x_mat, s = "lambda.1se", type = "response")
    if (is.array(p) && length(dim(p)) == 3) p <- p[, 1, ]
    if (is.matrix(p)) return(p)
    return(cbind(1 - p, p))
  }
  if (obj$type == "ranger") {
    pred <- predict(obj$model, data = df)$predictions
    return(pred)
  }
  if (obj$type == "rpart") {
    return(predict(obj$model, newdata = df))
  }
  stop("Unknown type: ", obj$type)
}

# ------------------------------------------------------------------------------
# Cross-validation to select best model from whitelist
# ------------------------------------------------------------------------------

#' Run CV for one learner and return mean metric (RMSE or accuracy)
#' @noRd
cv_one_learner <- function(X, y, learner, sample_weight, task, n_folds, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(X)
  folds <- sample(rep(seq_len(n_folds), length.out = n))
  preds <- numeric(n)
  for (k in seq_len(n_folds)) {
    tr <- which(folds != k)
    te <- which(folds == k)
    fit <- fit_single_automl_learner(X[tr, , drop = FALSE], y[tr], learner,
                                    sample_weight = if (!is.null(sample_weight)) sample_weight[tr] else NULL,
                                    task = task)
    preds[te] <- predict_single_automl_learner(fit, X[te, , drop = FALSE])
  }
  if (task == "regression") {
    sqrt(mean((y - preds)^2))
  } else {
    # preds are probability of positive class; round to 0/1; factor levels are 1,2
    pred_class <- round(preds) + 1L
    pred_class <- pmin(pmax(pred_class, 1L), 2L)
    acc <- mean(as.integer(as.factor(y)) == pred_class)
    1 - acc
  }
}

# ------------------------------------------------------------------------------
# AutomatedMLModel: sklearn-style fit/predict interface
# ------------------------------------------------------------------------------

#' AutomatedML model: fit and predict using auto-selected learner
#'
#' Selects a model from the config whitelist via cross-validation, then fits
#' it on the full data. Provides \code{fit}, \code{predict}, and
#' \code{predict_proba} (for classification) for use as a nuisance model in
#' causal estimators.
#'
#' @param automl_config \code{EconAutoMLConfig} object.
#' @param workspace optional; in R port ignored (local runs only).
#' @param experiment_name_prefix character. Prefix for experiment naming.
#' @return Object of class \code{AutomatedMLModel} (empty until \code{fit} is called).
#' @references EconML \code{AutomatedMLModel}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # AutomatedMLModel(...)
#' }
#' @export
AutomatedMLModel <- function(automl_config,
                             workspace = NULL,
                             experiment_name_prefix = NULL) {
  if (!inherits(automl_config, "EconAutoMLConfig")) {
    stop("automl_config must be an EconAutoMLConfig object")
  }
  prefix <- if (!is.null(experiment_name_prefix)) {
    substr(experiment_name_prefix, 1L, 18L)
  } else {
    automl_config$experiment_name_prefix
  }
  structure(
    list(
      automl_config = automl_config,
      workspace = workspace,
      experiment_name_prefix = prefix,
      inner_models = NULL,
      best_learner = NULL,
      task = automl_config$task,
      is_multioutput = FALSE,
      n_outputs = 1L
    ),
    class = "AutomatedMLModel"
  )
}

#' Fit AutomatedMLModel: select best learner via CV and refit on full data
#'
#' @param object \code{AutomatedMLModel} from \code{AutomatedMLModel()}.
#' @param X matrix or data.frame of features.
#' @param y vector (regression) or factor/vector (classification); or matrix for
#'   multi-output regression (one model per column).
#' @param sample_weight numeric vector or NULL.
#' @param ... passed to internal fitters (e.g. ranger, glmnet).
#' @return Fitted \code{AutomatedMLModel} (invisibly). Assign the result, e.g.
#'   \code{object <- fit(object, X, y)}, so predictions use the fitted model.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.AutomatedMLModel(...)
#' }
#' @export
fit.AutomatedMLModel <- function(object, X, y, sample_weight = NULL, ...) {
  cfg <- object$automl_config
  task <- cfg$task
  n_folds <- cfg$n_folds
  show <- cfg$show_output
  X <- as.data.frame(X)
  X <- as.matrix(X)

  # Multi-output: one model per column (like Python MultiOutputRegressor)
  if (is.matrix(y) && ncol(y) > 1) {
    object$is_multioutput <- TRUE
    object$n_outputs <- ncol(y)
    object$inner_models <- vector("list", ncol(y))
    for (j in seq_len(ncol(y))) {
      if (show) message("AutomatedMLModel fitting output ", j, " of ", ncol(y))
      obj_j <- AutomatedMLModel(cfg, object$workspace,
                                paste0(object$experiment_name_prefix, "_", j))
      obj_j <- fit.AutomatedMLModel(obj_j, X, y[, j], sample_weight = sample_weight, ...)
      object$inner_models[[j]] <- obj_j
    }
    return(invisible(object))
  }

  if (is.matrix(y)) y <- as.numeric(y)
  learners <- cfg$whitelist_models
  if (is.null(learners) || length(learners) == 0) {
    learners <- unique(c(LINEAR_MODELS_SET, SAMPLE_WEIGHTS_MODELS_SET))
  }

  # Select best learner by CV
  cv_scores <- numeric(length(learners))
  names(cv_scores) <- learners
  for (i in seq_along(learners)) {
    cv_scores[i] <- tryCatch(
      cv_one_learner(X, y, learners[i], sample_weight, task, n_folds),
      error = function(e) Inf
    )
    if (show) message("  CV score (", learners[i], "): ", round(cv_scores[i], 4))
  }
  best_learner <- learners[which.min(cv_scores)[1]]
  object$best_learner <- best_learner
  if (show) message("Best learner: ", best_learner)

  # Refit on full data
  object$inner_models <- fit_single_automl_learner(X, y, best_learner, sample_weight, task)
  invisible(object)
}

#' Predict using fitted AutomatedMLModel
#'
#' @param object fitted \code{AutomatedMLModel}.
#' @param newdata matrix or data.frame of features.
#' @param ... unused.
#' @return Numeric vector of predictions, or matrix for multi-output.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.AutomatedMLModel(...)
#' }
#' @export
predict.AutomatedMLModel <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  if (object$is_multioutput) {
    out <- matrix(NA_real_, nrow(newdata), object$n_outputs)
    for (j in seq_len(object$n_outputs)) {
      out[, j] <- predict(object$inner_models[[j]], newdata)
    }
    return(out)
  }
  predict_single_automl_learner(object$inner_models, newdata)
}

#' Predict class probabilities using fitted AutomatedMLModel (classification only)
#'
#' @param object fitted \code{AutomatedMLModel} with \code{task = "classification"}.
#' @param newdata matrix or data.frame of features.
#' @param ... unused.
#' @return Matrix of probabilities (columns = classes).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict_proba.AutomatedMLModel(...)
#' }
#' @export
predict_proba.AutomatedMLModel <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  if (object$is_multioutput) {
    stop("predict_proba for multi-output not implemented")
  }
  predict_proba_single_automl_learner(object$inner_models, newdata)
}

# ------------------------------------------------------------------------------
# addAutomatedML: convert configs to models in estimator args (R helper)
# ------------------------------------------------------------------------------

#' Replace EconAutoMLConfig args with AutomatedMLModel instances
#'
#' In Python, \code{addAutomatedML} is a mixin that wraps the base estimator so
#' any \code{EconAutoMLConfig} in constructor args is turned into an
#' \code{AutomatedMLModel}. In R we provide a helper that takes a list of
#' arguments (e.g. for a meta-learner) and replaces any \code{EconAutoMLConfig}
#' with an \code{AutomatedMLModel} so you can pass the result into estimators
#' that accept "learner" or model objects.
#'
#' @param args named list of arguments (e.g. \code{list(learner = my_config)}).
#'   Any element that is an \code{EconAutoMLConfig} is replaced by an
#'   \code{AutomatedMLModel} with that config.
#' @param workspace optional; passed to \code{AutomatedMLModel} (unused in local port).
#' @return List with same names; configs replaced by \code{AutomatedMLModel} objects.
#' @references EconML \code{addAutomatedML}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # add_automated_ml(...)
#' }
#' @export
add_automated_ml <- function(args, workspace = NULL) {
  if (!is.list(args)) stop("args must be a list")
  out <- args
  for (nm in names(out)) {
    if (inherits(out[[nm]], "EconAutoMLConfig")) {
      prefix <- substr(nm, 1L, 18L)
      out[[nm]] <- AutomatedMLModel(out[[nm]], workspace = workspace,
                                    experiment_name_prefix = prefix)
    }
  }
  out
}

# ------------------------------------------------------------------------------
# Backward-compatible aliases (snake_case preferred in R)
# ------------------------------------------------------------------------------

#' @rdname set_automated_ml_workspace
#' @return
#' Object returned by \code{function_name}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # function_name(...)
#' }
#' @export
setAutomatedMLWorkspace <- set_automated_ml_workspace

#' @rdname add_automated_ml
#' @return
#' Object returned by \code{function_name}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # function_name(...)
#' }
#' @export
addAutomatedML <- add_automated_ml
