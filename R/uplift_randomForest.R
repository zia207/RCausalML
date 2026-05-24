# CausalML-R: Uplift random forest for heterogeneous treatment effects
# Python reference: causalml.inference.tree.UpliftRandomForestClassifier
# Supports binary / multi-arm treatment, classification and regression outcomes.

# --- Internal helpers --------------------------------------------------------

#' Map Python-style evaluation function names to internal criterion codes
#' @noRd
.uplift_rf_criterion <- function(evaluation_function) {
  ef <- toupper(trimws(as.character(evaluation_function)[1L]))
  switch(ef,
    "KL" = "kl",
    "ED" = "ed",
    "EUCLIDEAN" = "ed",
    "EUCLIDEAN DISTANCE" = "ed",
    "CHI" = "chi",
    "CHI-SQUARE" = "chi",
    "CHI-SQUARED" = "chi",
    "CTS" = "cts",
    "DDP" = "ddp",
    "IDDP" = "iddp",
    "IT" = "it",
    "CIT" = "cit",
    "TLEARNER" = "tlearner",
    "T-LEARNER" = "tlearner",
    stop("Unknown evaluation_function: ", evaluation_function,
         ". Use KL, ED, Chi, CTS, DDP, IDDP, IT, CIT, or TLearner.")
  )
}

#' Detect whether outcome is binary (classification uplift)
#' @noRd
.uplift_rf_is_binary_outcome <- function(y) {
  y <- as.numeric(y)
  uniq <- unique(na.omit(y))
  length(uniq) <= 2L && all(uniq %in% c(0, 1))
}

#' Normalize treatment to character labels; infer control when NULL
#' @noRd
.uplift_rf_prepare_treatment <- function(treatment, control_name = NULL) {
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

#' Bag uplift trees for one control-vs-treatment contrast
#' @noRd
.uplift_rf_build_trees <- function(X, y, w, criterion, n_trees, min_node_size,
                                   min_samples_treatment, max_depth, max_features,
                                   n_reg, normalization, random_state, n_cores) {
  n <- nrow(X)
  build_one <- function(b) {
    idx <- sample.int(n, replace = TRUE)
    build_uplift_tree(
      X[idx, , drop = FALSE], y[idx], w[idx],
      criterion = criterion,
      min_node_size = min_node_size,
      min_samples_treatment = min_samples_treatment,
      max_depth = max_depth,
      n_reg = n_reg,
      normalization = normalization,
      max_features = max_features,
      random_state = if (is.null(random_state)) NULL else random_state + b
    )
  }
  parallel_lapply(seq_len(n_trees), build_one, n_cores = n_cores)
}

#' Fit T-learner ranger forests (regression uplift fallback)
#' @noRd
.uplift_rf_build_tlearner <- function(X, y, w, n_trees, min_node_size, ...) {
  df0 <- as.data.frame(X[w == 0, , drop = FALSE]); df0$y <- y[w == 0]
  df1 <- as.data.frame(X[w == 1, , drop = FALSE]); df1$y <- y[w == 1]
  fit0 <- ranger::ranger(
    y ~ ., data = df0, num.trees = n_trees,
    min.node.size = min_node_size, importance = "impurity", ...
  )
  fit1 <- ranger::ranger(
    y ~ ., data = df1, num.trees = n_trees,
    min.node.size = min_node_size, importance = "impurity", ...
  )
  list(fit_0 = fit0, fit_1 = fit1)
}

#' Core fit for binary treatment (0/1) contrast
#' @noRd
.uplift_rf_fit_binary <- function(X, y, w, task, criterion, n_trees, min_node_size,
                                  min_samples_treatment, max_depth, max_features,
                                  n_reg, normalization, random_state, n_cores, ...) {
  w <- as.integer(w)
  if (task == "classification") {
    if (criterion %in% c("ddp", "iddp") && !.uplift_rf_is_binary_outcome(y))
      stop("DDP and IDDP require binary outcome (0/1).")
    trees <- .uplift_rf_build_trees(
      X, y, w, criterion, n_trees, min_node_size, min_samples_treatment,
      max_depth, max_features, n_reg, normalization, random_state, n_cores
    )
    structure(
      list(
        trees = trees,
        X_names = colnames(X),
        type = criterion,
        task = task,
        evaluation_function = toupper(criterion),
        n_estimators = n_trees,
        classes_ = c("control", "treatment")
      ),
      class = c("uplift_randomForest", "uplift_rf")
    )
  } else {
    if (criterion == "tlearner") {
      tl <- .uplift_rf_build_tlearner(X, y, w, n_trees, min_node_size, ...)
      structure(
        c(
          tl,
          list(
            X_names = colnames(X),
            type = "tlearner",
            task = task,
            evaluation_function = "TLearner",
            n_estimators = n_trees,
            classes_ = c("control", "treatment")
          )
        ),
        class = c("uplift_randomForest", "uplift_rf")
      )
    } else {
      trees <- .uplift_rf_build_trees(
        X, y, w, criterion, n_trees, min_node_size, min_samples_treatment,
        max_depth, max_features, n_reg, normalization, random_state, n_cores
      )
      structure(
        list(
          trees = trees,
          X_names = colnames(X),
          type = criterion,
          task = task,
          evaluation_function = toupper(criterion),
          n_estimators = n_trees,
          classes_ = c("control", "treatment")
        ),
        class = c("uplift_randomForest", "uplift_rf")
      )
    }
  }
}

# --- User-facing API ---------------------------------------------------------

#' Uplift random forest for heterogeneous treatment effects
#'
#' Unified R port of Python \code{UpliftRandomForestClassifier} /
#' regression uplift forests. Estimates conditional average treatment effects
#' (CATE) with bagged uplift trees for **binary** outcomes, or T-learner /
#' interaction-tree forests for **continuous** outcomes.
#'
#' For multi-arm experiments, one forest is fit per non-control arm (control
#' vs treatment_k), matching Python's multi-treatment API.
#'
#' @param X covariate matrix or data.frame
#' @param treatment treatment indicator: numeric 0/1, or character/factor group
#'   labels (e.g. \code{"control"}, \code{"treatment1"})
#' @param y outcome: binary 0/1 for classification; continuous for regression
#' @param control_name control level when \code{treatment} is character/factor;
#'   default \code{"control"} if present, else the most frequent level
#' @param n_estimators number of trees in each forest (Python \code{n_estimators})
#' @param max_depth maximum tree depth
#' @param min_samples_leaf minimum samples per leaf (Python \code{min_samples_leaf})
#' @param min_samples_treatment minimum samples per treatment arm in a node
#' @param n_reg regularization weight for parent node (classification splits)
#' @param max_features number of features considered at each split (\code{NULL} = all)
#' @param normalization normalize split gain by treatment entropy (classification)
#' @param evaluation_function splitting criterion: \code{"KL"}, \code{"ED"},
#'   \code{"Chi"}, \code{"CTS"}, \code{"DDP"}, \code{"IDDP"} (classification);
#'   \code{"TLearner"}, \code{"IT"}, or \code{"CIT"} (regression). When
#'   \code{NULL}, defaults to \code{"KL"} for classification and
#'   \code{"TLearner"} for regression.
#' @param task \code{"auto"} (default), \code{"classification"}, or
#'   \code{"regression"}
#' @param random_state random seed for reproducibility
#' @param n_cores number of cores for parallel tree building (\code{1} = sequential)
#' @param ... passed to \pkg{ranger} when \code{evaluation_function = "TLearner"}
#' @return An object of class \code{uplift_randomForest}. Single-arm models also
#'   inherit \code{uplift_rf}; multi-arm models inherit \code{uplift_rf_multi}.
#' @seealso \code{\link{uplift_rf_kl}}, \code{\link{uplift_rf_multi}},
#'   \code{\link{interaction_tree}}, \code{\link{make_uplift_classification}},
#'   \code{\link{make_uplift_regression}}
#' @examples
#' \dontrun{
#' out <- make_uplift_classification(n_samples = 500, random_seed = 1)
#' df <- out$data
#' X <- as.matrix(df[, out$X_names])
#' w <- as.integer(df$treatment_group_key != "control")
#' y <- df$conversion
#' fit <- uplift_randomForest(X, w, y, n_estimators = 20, max_depth = 4)
#' predict(fit, X)
#' }
#' @export
uplift_randomForest <- function(X, treatment, y,
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
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  y <- as.numeric(as.vector(y))
  task <- match.arg(task)
  if (task == "auto") {
    task <- if (.uplift_rf_is_binary_outcome(y)) "classification" else "regression"
  }
  if (is.null(evaluation_function)) {
    evaluation_function <- if (task == "classification") "KL" else "TLearner"
  }
  criterion <- .uplift_rf_criterion(evaluation_function)

  trt_prep <- .uplift_rf_prepare_treatment(treatment, control_name)
  treatment_chr <- trt_prep$treatment
  control_name <- trt_prep$control_name
  trt_levels <- unique(treatment_chr)
  treatment_names <- setdiff(trt_levels, control_name)

  if (!is.null(random_state)) set.seed(random_state)

  if (length(treatment_names) == 0L)
    stop("Need at least one non-control treatment level.")

  # Multi-arm: one forest per treatment vs control
  if (length(treatment_names) > 1L || !all(treatment_chr %in% c(control_name, treatment_names[1L]))) {
    models <- list()
    for (nm in treatment_names) {
      idx <- treatment_chr %in% c(control_name, nm)
      w_bin <- as.integer(treatment_chr[idx] != control_name)
      models[[nm]] <- .uplift_rf_fit_binary(
        X[idx, , drop = FALSE], y[idx], w_bin, task, criterion,
        n_estimators, min_samples_leaf, min_samples_treatment, max_depth,
        max_features, n_reg, normalization, random_state, n_cores, ...
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
        evaluation_function = toupper(criterion),
        n_estimators = n_estimators
      ),
      class = c("uplift_randomForest", "uplift_rf_multi")
    ))
  }

  # Binary contrast (numeric 0/1 or two-level character)
  if (is.numeric(treatment) || all(trt_levels %in% c("0", "1", control_name, treatment_names[1L]))) {
    if (!is.numeric(treatment)) {
      w_bin <- as.integer(treatment_chr != control_name)
    } else {
      w_bin <- as.integer(treatment != as.numeric(control_name))
    }
  } else {
    w_bin <- as.integer(treatment_chr != control_name)
  }

  .uplift_rf_fit_binary(
    X, y, w_bin, task, criterion, n_estimators, min_samples_leaf,
    min_samples_treatment, max_depth, max_features, n_reg, normalization,
    random_state, n_cores, ...
  )
}

#' Uplift random forest classifier (binary outcome)
#'
#' Wrapper around \code{\link{uplift_randomForest}} with
#' \code{task = "classification"}.
#' @inheritParams uplift_randomForest
#' @export
uplift_randomForestClassifier <- function(X, treatment, y, ...) {
  uplift_randomForest(X, treatment, y, task = "classification", ...)
}

#' Uplift random forest regressor (continuous outcome)
#'
#' Wrapper around \code{\link{uplift_randomForest}} with
#' \code{task = "regression"}.
#' @inheritParams uplift_randomForest
#' @export
uplift_randomForestRegressor <- function(X, treatment, y, ...) {
  uplift_randomForest(X, treatment, y, task = "regression", ...)
}

#' Predict CATE from an uplift random forest
#'
#' @param object fitted \code{uplift_randomForest} object
#' @param newdata feature matrix or data.frame
#' @param full_output if \code{TRUE}, return control/treatment probabilities or
#'   means, deltas, and (multi-arm) recommended treatment
#' @param ... unused
#' @return CATE vector, matrix (multi-arm), or data.frame when
#'   \code{full_output = TRUE}
#' @export
predict.uplift_randomForest <- function(object, newdata, full_output = FALSE, ...) {
  if (inherits(object, "uplift_rf_multi")) {
    return(predict.uplift_rf_multi(object, newdata, full_output = full_output, ...))
  }
  if (inherits(object, "uplift_rf")) {
    return(predict.uplift_rf(object, newdata, full_output = full_output, ...))
  }
  stop("Invalid uplift_randomForest object.")
}

#' @export
print.uplift_randomForest <- function(x, ...) {
  cls <- paste(class(x), collapse = ", ")
  cat("Uplift random forest (", cls, ")\n", sep = "")
  cat("  task:", x$task, "\n")
  cat("  criterion:", x$evaluation_function, "\n")
  cat("  n_estimators:", x$n_estimators, "\n")
  if (!is.null(x$treatment_names)) {
    cat("  arms:", paste(c(x$control_name, x$treatment_names), collapse = ", "), "\n")
  }
  invisible(x)
}
