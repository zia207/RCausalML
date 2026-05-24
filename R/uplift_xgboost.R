# CausalML-R: Uplift XGBoost for heterogeneous treatment effects
# T-learner XGBoost port aligned with uplift_randomForest() API.
# Supports binary / multi-arm treatment, classification and regression outcomes.

# --- Internal helpers --------------------------------------------------------

#' Require xgboost for uplift XGBoost
#' @noRd
.uplift_xgb_check <- function() {
  if (!requireNamespace("xgboost", quietly = TRUE))
    stop("Package 'xgboost' is required. Install with install.packages('xgboost').")
}

#' Map evaluation function names; uplift XGBoost supports T-learner only
#' @noRd
.uplift_xgb_criterion <- function(evaluation_function) {
  ef <- toupper(trimws(as.character(evaluation_function)[1L]))
  switch(ef,
    "TLEARNER" = "tlearner",
    "T-LEARNER" = "tlearner",
    stop("Unknown evaluation_function: ", evaluation_function,
         ". uplift_xgboost supports TLearner only.")
  )
}

#' Detect whether outcome is binary (classification uplift)
#' @noRd
.uplift_xgb_is_binary_outcome <- function(y) {
  y <- as.numeric(y)
  uniq <- unique(na.omit(y))
  length(uniq) <= 2L && all(uniq %in% c(0, 1))
}

#' Normalize treatment to character labels; infer control when NULL
#' @noRd
.uplift_xgb_prepare_treatment <- function(treatment, control_name = NULL) {
  treatment <- as.vector(treatment)
  if (is.factor(treatment)) treatment <- as.character(treatment)
  if (is.numeric(treatment)) {
    uniq <- sort(unique(na.omit(treatment)))
    if (is.null(control_name)) control_name <- as.character(min(uniq))
    treatment <- ifelse(treatment == as.numeric(control_name), control_name,
                        as.character(treatment))
  } else {
    treatment <- as.character(treatment)
    if (is.null(control_name)) {
      if ("control" %in% treatment) {
        control_name <- "control"
      } else {
        control_name <- names(sort(table(treatment), decreasing = TRUE))[1L]
      }
    }
  }
  list(treatment = treatment, control_name = control_name)
}

#' Build xgboost() args for regression or classification
#' @noRd
.uplift_xgb_args <- function(x, y, task, nrounds, random_state = NULL, ...) {
  dots <- list(...)
  nrounds <- if ("nrounds" %in% names(dots)) dots$nrounds else nrounds
  dots <- dots[setdiff(names(dots), c("nrounds", "seed"))]
  # xgboost >= 3 simplified API: numeric 0/1 outcomes use regression (see DMLearner.R).
  objective <- "reg:squarederror"
  out <- c(
    list(
      x = x,
      y = y,
      nrounds = nrounds,
      verbosity = 0L,
      objective = objective
    ),
    dots
  )
  if (!is.null(random_state)) out$seed <- random_state
  out
}

#' Predict from a fitted xgboost model
#' @noRd
.uplift_xgb_predict <- function(model, newdata) {
  as.vector(predict(model, newdata = as.matrix(newdata)))
}

#' Fit T-learner XGBoost models (control vs treated)
#' @noRd
.uplift_xgb_build_tlearner <- function(X, y, w, task, nrounds, random_state = NULL, ...) {
  .uplift_xgb_check()
  df0 <- as.matrix(X[w == 0, , drop = FALSE])
  df1 <- as.matrix(X[w == 1, , drop = FALSE])
  y0 <- y[w == 0]
  y1 <- y[w == 1]
  if (nrow(df0) == 0L || nrow(df1) == 0L)
    stop("Both control and treatment arms need at least one observation.")
  fit0 <- do.call(
    xgboost::xgboost,
    .uplift_xgb_args(df0, y0, task, nrounds, random_state = random_state, ...)
  )
  fit1 <- do.call(
    xgboost::xgboost,
    .uplift_xgb_args(df1, y1, task, nrounds, random_state = random_state, ...)
  )
  list(fit_0 = fit0, fit_1 = fit1)
}

#' Core fit for binary treatment (0/1) contrast
#' @noRd
.uplift_xgb_fit_binary <- function(X, y, w, task, nrounds, random_state = NULL, ...) {
  w <- as.integer(w)
  tl <- .uplift_xgb_build_tlearner(X, y, w, task, nrounds, random_state = random_state, ...)
  structure(
    c(
      tl,
      list(
        X_names = colnames(X),
        type = "tlearner",
        task = task,
        evaluation_function = "TLearner",
        n_estimators = nrounds,
        classes_ = c("control", "treatment")
      )
    ),
    class = c("uplift_xgboost", "uplift_xgb")
  )
}

#' Predict CATE from uplift_xgb object
#' @noRd
.uplift_xgb_predict_single <- function(object, newdata, full_output = FALSE) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  p0 <- .uplift_xgb_predict(object$fit_0, newdata)
  p1 <- .uplift_xgb_predict(object$fit_1, newdata)
  if (object$task == "classification") {
    p0 <- pmin(1, pmax(0, p0))
    p1 <- pmin(1, pmax(0, p1))
  }
  if (full_output)
    return(data.frame(control = p0, treatment1 = p1, delta_treatment1 = p1 - p0))
  p1 - p0
}

#' Predict from multi-arm uplift_xgb object
#' @noRd
.uplift_xgb_predict_multi <- function(object, newdata, full_output = FALSE) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  n <- nrow(newdata)
  tn <- object$treatment_names
  deltas <- matrix(NA_real_, n, length(tn))
  colnames(deltas) <- tn
  p0_mat <- p1_mat <- matrix(NA_real_, n, length(tn))
  for (k in seq_along(tn)) {
    nm <- tn[k]
    pred <- .uplift_xgb_predict_single(object$models[[nm]], newdata, full_output = TRUE)
    deltas[, k] <- pred$delta_treatment1
    p0_mat[, k] <- pred$control
    p1_mat[, k] <- pred$treatment1
  }
  control_prob <- rowMeans(p0_mat)
  treatment_probs <- as.data.frame(p1_mat)
  names(treatment_probs) <- tn
  if (full_output) {
    best_idx <- apply(deltas, 1, function(r) {
      if (all(r < 0)) 0L else which.max(r)
    })
    recommended_treatment <- c(object$control_name, tn)[best_idx + 1L]
    max_delta <- apply(deltas, 1, max)
    out <- data.frame(control = control_prob, treatment_probs,
                      recommended_treatment = recommended_treatment)
    for (k in seq_along(tn))
      out[[paste0("delta_", tn[k])]] <- deltas[, k]
    out$max_delta <- max_delta
    return(out)
  }
  deltas
}

# --- User-facing API ---------------------------------------------------------

#' Uplift XGBoost for heterogeneous treatment effects
#'
#' T-learner XGBoost uplift model aligned with \code{\link{uplift_randomForest}}.
#' Fits separate XGBoost outcome models for control and treated units and
#' estimates CATE as the difference in predicted outcomes.
#'
#' For multi-arm experiments, one model pair is fit per non-control arm
#' (control vs treatment_k), matching the multi-treatment uplift API.
#'
#' @param X covariate matrix or data.frame
#' @param treatment treatment indicator: numeric 0/1, or character/factor group
#'   labels (e.g. \code{"control"}, \code{"treatment1"})
#' @param y outcome: binary 0/1 for classification; continuous for regression
#' @param control_name control level when \code{treatment} is character/factor;
#'   default \code{"control"} if present, else the most frequent level
#' @param n_estimators number of boosting rounds per arm (XGBoost \code{nrounds})
#' @param max_depth maximum tree depth passed to XGBoost
#' @param min_samples_leaf ignored (kept for API parity with uplift forests)
#' @param min_samples_treatment ignored (kept for API parity with uplift forests)
#' @param n_reg ignored (kept for API parity with uplift forests)
#' @param max_features ignored (kept for API parity with uplift forests)
#' @param normalization ignored (kept for API parity with uplift forests)
#' @param evaluation_function splitting / learner type; only \code{"TLearner"} is
#'   supported for XGBoost uplift
#' @param task \code{"auto"} (default), \code{"classification"}, or
#'   \code{"regression"}
#' @param random_state random seed for reproducibility
#' @param n_cores ignored (kept for API parity with uplift forests)
#' @param ... passed to \code{xgboost::xgboost} (e.g. \code{eta}, \code{subsample})
#' @return An object of class \code{uplift_xgboost}. Single-arm models also
#'   inherit \code{uplift_xgb}; multi-arm models inherit \code{uplift_xgb_multi}.
#' @seealso \code{\link{uplift_randomForest}}, \code{\link{make_uplift_classification}},
#'   \code{\link{make_uplift_regression}}
#' @examples
#' \dontrun{
#' out <- make_uplift_classification(n_samples = 500, random_seed = 1)
#' df <- out$data
#' X <- as.matrix(df[, out$X_names])
#' w <- as.integer(df$treatment_group_key != "control")
#' y <- df$conversion
#' fit <- uplift_xgboost(X, w, y, n_estimators = 50, max_depth = 4)
#' predict(fit, X)
#' }
#' @export
uplift_xgboost <- function(X, treatment, y,
                           control_name = NULL,
                           n_estimators = 100L,
                           max_depth = 5L,
                           min_samples_leaf = 10L,
                           min_samples_treatment = 10L,
                           n_reg = 100,
                           max_features = NULL,
                           normalization = TRUE,
                           evaluation_function = NULL,
                           task = c("auto", "classification", "regression"),
                           random_state = NULL,
                           n_cores = 1L,
                           ...) {
  .uplift_xgb_check()
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  y <- as.numeric(as.vector(y))
  task <- match.arg(task)
  if (task == "auto") {
    task <- if (.uplift_xgb_is_binary_outcome(y)) "classification" else "regression"
  }
  if (is.null(evaluation_function)) evaluation_function <- "TLearner"
  criterion <- .uplift_xgb_criterion(evaluation_function)

  trt_prep <- .uplift_xgb_prepare_treatment(treatment, control_name)
  treatment_chr <- trt_prep$treatment
  control_name <- trt_prep$control_name
  trt_levels <- unique(treatment_chr)
  treatment_names <- setdiff(trt_levels, control_name)

  if (length(treatment_names) == 0L)
    stop("Need at least one non-control treatment level.")

  xgb_dots <- list(...)
  if (!"max_depth" %in% names(xgb_dots)) xgb_dots$max_depth <- max_depth

  fit_binary <- function(X_sub, y_sub, w_sub) {
    do.call(
      .uplift_xgb_fit_binary,
      c(
        list(
          X = X_sub,
          y = y_sub,
          w = w_sub,
          task = task,
          nrounds = n_estimators,
          random_state = random_state
        ),
        xgb_dots
      )
    )
  }

  if (length(treatment_names) > 1L ||
      !all(treatment_chr %in% c(control_name, treatment_names[1L]))) {
    models <- list()
    for (nm in treatment_names) {
      idx <- treatment_chr %in% c(control_name, nm)
      w_bin <- as.integer(treatment_chr[idx] != control_name)
      models[[nm]] <- fit_binary(
        X[idx, , drop = FALSE], y[idx], w_bin
      )
    }
    return(structure(
      list(
        models = models,
        control_name = control_name,
        treatment_names = treatment_names,
        classes_ = c(control_name, treatment_names),
        X_names = colnames(X),
        task = task,
        evaluation_function = "TLearner",
        n_estimators = n_estimators
      ),
      class = c("uplift_xgboost", "uplift_xgb_multi")
    ))
  }

  if (is.numeric(treatment) || all(trt_levels %in% c("0", "1", control_name, treatment_names[1L]))) {
    if (!is.numeric(treatment)) {
      w_bin <- as.integer(treatment_chr != control_name)
    } else {
      w_bin <- as.integer(treatment != as.numeric(control_name))
    }
  } else {
    w_bin <- as.integer(treatment_chr != control_name)
  }

  fit_binary(X, y, w_bin)
}

#' Uplift XGBoost classifier (binary outcome)
#'
#' Wrapper around \code{\link{uplift_xgboost}} with
#' \code{task = "classification"}.
#' @inheritParams uplift_xgboost
#' @export
uplift_xgboostClassifier <- function(X, treatment, y, ...) {
  uplift_xgboost(X, treatment, y, task = "classification", ...)
}

#' Uplift XGBoost regressor (continuous outcome)
#'
#' Wrapper around \code{\link{uplift_xgboost}} with
#' \code{task = "regression"}.
#' @inheritParams uplift_xgboost
#' @export
uplift_xgboostRegressor <- function(X, treatment, y, ...) {
  uplift_xgboost(X, treatment, y, task = "regression", ...)
}

#' Predict CATE from an uplift XGBoost model
#'
#' @param object fitted \code{uplift_xgboost} object
#' @param newdata feature matrix or data.frame
#' @param full_output if \code{TRUE}, return control/treatment probabilities or
#'   means, deltas, and (multi-arm) recommended treatment
#' @param ... unused
#' @return CATE vector, matrix (multi-arm), or data.frame when
#'   \code{full_output = TRUE}
#' @export
predict.uplift_xgboost <- function(object, newdata, full_output = FALSE, ...) {
  if (inherits(object, "uplift_xgb_multi")) {
    return(.uplift_xgb_predict_multi(object, newdata, full_output = full_output))
  }
  if (inherits(object, "uplift_xgb")) {
    return(.uplift_xgb_predict_single(object, newdata, full_output = full_output))
  }
  stop("Invalid uplift_xgboost object.")
}

#' @export
print.uplift_xgboost <- function(x, ...) {
  cls <- paste(class(x), collapse = ", ")
  cat("Uplift XGBoost (", cls, ")\n", sep = "")
  cat("  task:", x$task, "\n")
  cat("  criterion:", x$evaluation_function, "\n")
  cat("  n_estimators:", x$n_estimators, "\n")
  if (!is.null(x$treatment_names)) {
    cat("  arms:", paste(c(x$control_name, x$treatment_names), collapse = ", "), "\n")
  }
  invisible(x)
}
