# CausalML-R: CATE interpreters (EconML-style)
# Single-tree interpretation of CATE and policy trees.
# Reference: EconML cate_interpreter (SingleTreeCateInterpreter, SingleTreePolicyInterpreter)

# --------------- Helpers ---------------

#' Get CATE predictions from a fitted estimator
#'
#' Dispatches to \code{predict} or \code{effect} depending on the object.
#' Returns a numeric vector (single treatment) or matrix (multi-treatment).
#' @param estimator fitted CATE estimator (e.g. SLearner, TLearner, DMLearner, DRLearner)
#' @param X feature matrix or data.frame
#' @return numeric vector of length nrow(X) or matrix (n x n_treatments)
#' @noRd
get_cate_from_estimator <- function(estimator, X) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  pred <- tryCatch(
    predict(estimator, newdata = X),
    error = function(e) NULL
  )
  if (is.null(pred)) {
    if (!is.null(estimator$model_cate) && inherits(estimator, "DynamicDMLearner"))
      pred <- predict(estimator, newdata = X)
    else
      stop("CATE estimator must support predict(estimator, newdata = X) returning CATE.")
  }
  if (is.matrix(pred)) pred <- as.matrix(pred) else pred <- as.numeric(pred)
  pred
}

#' Leaf node id (rpart frame row name) for each observation
#' @noRd
rpart_leaf_ids <- function(fit, X) {
  if (is.null(fit$where)) {
    # predict node from new data
    wh <- attr(predict(fit, newdata = as.data.frame(X), type = "matrix"), "where")
    if (is.null(wh)) stop("rpart tree has no 'where'; use the same X as in fit.")
    return(wh)
  }
  fit$where
}

#' Node ids (frame row names) that are leaves
#' @noRd
rpart_leaf_node_ids <- function(fit) {
  f <- fit$frame
  as.integer(row.names(f)[f$var == "<leaf>"])
}

#' For each node id in the tree, get descendant leaf node ids (rpart convention: 2*k left, 2*k+1 right)
#' @noRd
rpart_descendant_leaves <- function(fit) {
  f <- fit$frame
  node_ids <- as.integer(row.names(f))
  leaves <- as.integer(row.names(f)[f$var == "<leaf>"])
  desc <- list()
  for (nid in node_ids) {
    desc[[as.character(nid)]] <- rpart_descendant_leaves_rec(nid, node_ids, leaves)
  }
  desc
}

rpart_descendant_leaves_rec <- function(nid, node_ids, leaves) {
  if (nid %in% leaves) return(nid)
  left <- 2L * nid
  right <- 2L * nid + 1L
  out <- integer(0)
  if (left %in% node_ids) out <- c(out, rpart_descendant_leaves_rec(left, node_ids, leaves))
  if (right %in% node_ids) out <- c(out, rpart_descendant_leaves_rec(right, node_ids, leaves))
  out
}

#' For each node (by frame row index 1..nrow(frame)), get observation indices that pass through that node
#' @param fit fitted rpart object
#' @param X feature matrix used in fitting (or same structure)
#' @noRd
rpart_node_to_obs <- function(fit, X) {
  leaf_of_obs <- rpart_leaf_ids(fit, X)
  node_ids <- as.integer(row.names(fit$frame))
  desc_leaves <- rpart_descendant_leaves(fit)
  # Leaf node id for each observation (row name of frame at row leaf_of_obs)
  leaf_node_ids <- as.integer(row.names(fit$frame)[leaf_of_obs])
  node_dict <- list()
  for (i in seq_along(node_ids)) {
    nid <- node_ids[i]
    desc <- desc_leaves[[as.character(nid)]]
    mask <- leaf_node_ids %in% desc
    node_dict[[i]] <- which(mask)
  }
  node_dict
}


# =============================================================================
# SingleTreeCateInterpreter
# =============================================================================

#' Interpret a CATE or policy model with a tree (generic)
#' @param obj interpreter object (e.g. SingleTreeCateInterpreter, SingleTreePolicyInterpreter)
#' @param cate_estimator fitted CATE estimator
#' @param X feature matrix
#' @param ... passed to methods
#' @return
#' Object returned by \code{interpret}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # interpret(...)
#' }
#' @export
interpret <- function(obj, cate_estimator, X, ...) UseMethod("interpret")

#' Assign treatment from a policy interpreter (generic)
#' @param obj fitted SingleTreePolicyInterpreter
#' @param X feature matrix
#' @param ... passed to methods
#' @return
#' Object returned by \code{treat}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # treat(...)
#' }
#' @export
treat <- function(obj, X, ...) UseMethod("treat")

#' Single-tree CATE interpreter
#'
#' Interprets the heterogeneity of a CATE estimator by fitting a single
#' decision tree to the predicted CATE surface. The tree approximates the
#' CATE model with an interpretable partition. Aligns with EconML
#' \code{econml.cate_interpreter.SingleTreeCateInterpreter}.
#'
#' @param max_depth integer or NULL. Maximum depth of the tree. If NULL, nodes
#'   are expanded until purity or min_samples_split.
#' @param min_samples_split integer. Minimum number of samples to split an
#'   internal node (default 2).
#' @param min_samples_leaf integer. Minimum number of samples in a leaf
#'   (default 1).
#' @param min_weight_fraction_leaf numeric (default 0). Minimum weighted
#'   fraction of total weight required at a leaf.
#' @param max_features integer, character, or NULL. Number of features to
#'   consider at each split: "auto", "sqrt", "log2", or NULL for all.
#' @param random_state integer or NULL. Random seed.
#' @param min_impurity_decrease numeric (default 0). Minimum impurity decrease
#'   to split.
#' @return Object of class \code{SingleTreeCateInterpreter}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # SingleTreeCateInterpreter(...)
#' }
#' @export
SingleTreeCateInterpreter <- function(max_depth = NULL,
                                     min_samples_split = 2L,
                                     min_samples_leaf = 1L,
                                     min_weight_fraction_leaf = 0,
                                     max_features = NULL,
                                     random_state = NULL,
                                     min_impurity_decrease = 0) {
  structure(
    list(
      max_depth = max_depth,
      min_samples_split = as.integer(min_samples_split),
      min_samples_leaf = as.integer(min_samples_leaf),
      min_weight_fraction_leaf = min_weight_fraction_leaf,
      max_features = max_features,
      random_state = random_state,
      min_impurity_decrease = min_impurity_decrease,
      tree_model_ = NULL,
      node_dict_ = NULL,
      X_names_ = NULL,
      cate_col_ = NULL
    ),
    class = "SingleTreeCateInterpreter"
  )
}

#' Interpret a CATE estimator with a single tree
#'
#' Fits a regression tree to the CATE predictions so that the tree structure
#' summarizes effect heterogeneity by covariates.
#'
#' @param obj \code{SingleTreeCateInterpreter} object from
#'   \code{\link{SingleTreeCateInterpreter}}.
#' @param cate_estimator Fitted CATE estimator (e.g. \code{SLearner},
#'   \code{TLearner}, \code{DMLearner}, \code{DRLearner}) that supports
#'   \code{predict(estimator, newdata = X)} returning CATE.
#' @param X Feature matrix or data.frame used to interpret the estimator;
#'   must be compatible with the estimator.
#' @param ... Not used.
#' @return The \code{SingleTreeCateInterpreter} object (invisibly) with
#'   \code{tree_model_} and \code{node_dict_} filled.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # interpret.SingleTreeCateInterpreter(...)
#' }
#' @export
interpret.SingleTreeCateInterpreter <- function(obj, cate_estimator, X, ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  obj$X_names_ <- colnames(X)
  if (is.null(obj$X_names_)) obj$X_names_ <- paste0("X", seq_len(ncol(X)))

  y_pred <- get_cate_from_estimator(cate_estimator, X)
  # Single response for tree: if matrix, take first column or mean across treatments
  if (is.matrix(y_pred)) {
    if (ncol(y_pred) > 1L) y_pred <- rowMeans(y_pred) else y_pred <- y_pred[, 1L]
  }
  obj$cate_col_ <- as.numeric(y_pred)

  df <- as.data.frame(X)
  colnames(df) <- obj$X_names_
  df$cate <- obj$cate_col_

  control <- rpart::rpart.control(
    minsplit = obj$min_samples_split,
    minbucket = obj$min_samples_leaf,
    cp = 0,
    maxdepth = if (is.null(obj$max_depth)) 30L else obj$max_depth,
    minbranch = 1L
  )
  if (!is.null(obj$max_depth)) control$maxdepth <- obj$max_depth
  if (obj$min_impurity_decrease > 0) control$cp <- obj$min_impurity_decrease

  set_seed(obj$random_state)
  obj$tree_model_ <- rpart::rpart(cate ~ ., data = df, method = "anova", control = control, ...)

  # Node statistics: for each node, mean and std of CATE for observations in that node
  node_to_obs <- rpart_node_to_obs(obj$tree_model_, X)
  node_dict <- list()
  for (i in seq_along(node_to_obs)) {
    idx <- node_to_obs[[i]]
    if (length(idx) == 0) {
      node_dict[[i]] <- list(mean = NA_real_, std = NA_real_)
    } else {
      cate_node <- obj$cate_col_[idx]
      node_dict[[i]] <- list(
        mean = mean(cate_node, na.rm = TRUE),
        std = sd(cate_node, na.rm = TRUE)
      )
    }
  }
  obj$node_dict_ <- node_dict
  invisible(obj)
}

#' Predict CATE from the interpreted tree (simplified model)
#'
#' @param object fitted \code{SingleTreeCateInterpreter}.
#' @param newdata feature matrix or data.frame.
#' @param ... passed to \code{predict.rpart}.
#' @return Numeric vector of tree-predicted CATE.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.SingleTreeCateInterpreter(...)
#' }
#' @export
predict.SingleTreeCateInterpreter <- function(object, newdata, ...) {
  if (is.null(object$tree_model_))
    stop("interpret() must be called first.")
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  df <- as.data.frame(newdata)
  if (!is.null(object$X_names_) && ncol(df) == length(object$X_names_))
    colnames(df) <- object$X_names_
  as.numeric(predict(object$tree_model_, newdata = df, ...))
}


# =============================================================================
# SingleTreePolicyInterpreter
# =============================================================================

#' Single-tree policy interpreter
#'
#' Interprets a treatment assignment policy based on a CATE estimator by
#' fitting a single decision tree that predicts the best treatment (control
#' vs treat) from covariates. Aligns with EconML
#' \code{econml.cate_interpreter.SingleTreePolicyInterpreter}.
#'
#' @param risk_level numeric or NULL. If NULL, point CATE is used; if set (e.g.
#'   0.05), the lower (risk_seeking=FALSE) or upper (risk_seeking=TRUE) end of
#'   a confidence interval can be used (requires estimator to support intervals).
#' @param risk_seeking logical (default FALSE). If \code{risk_level} is set,
#'   use upper (TRUE) or lower (FALSE) end of the interval as effect.
#' @param max_depth integer or NULL. Maximum depth of the policy tree.
#' @param min_samples_split integer (default 2).
#' @param min_samples_leaf integer (default 1).
#' @param min_weight_fraction_leaf numeric (default 0).
#' @param max_features integer, character, or NULL.
#' @param min_balancedness_tol numeric in [0, 0.5] (default 0.45). Minimum
#'   fraction of samples that must fall on each side of a split.
#' @param min_impurity_decrease numeric (default 0).
#' @param random_state integer or NULL.
#' @return Object of class \code{SingleTreePolicyInterpreter}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # SingleTreePolicyInterpreter(...)
#' }
#' @export
SingleTreePolicyInterpreter <- function(risk_level = NULL,
                                        risk_seeking = FALSE,
                                        max_depth = NULL,
                                        min_samples_split = 2L,
                                        min_samples_leaf = 1L,
                                        min_weight_fraction_leaf = 0,
                                        max_features = NULL,
                                        min_balancedness_tol = 0.45,
                                        min_impurity_decrease = 0,
                                        random_state = NULL) {
  structure(
    list(
      risk_level = risk_level,
      risk_seeking = risk_seeking,
      max_depth = max_depth,
      min_samples_split = as.integer(min_samples_split),
      min_samples_leaf = as.integer(min_samples_leaf),
      min_weight_fraction_leaf = min_weight_fraction_leaf,
      max_features = max_features,
      min_balancedness_tol = min_balancedness_tol,
      min_impurity_decrease = min_impurity_decrease,
      random_state = random_state,
      tree_model_ = NULL,
      node_dict_ = NULL,
      policy_value_ = NULL,
      always_treat_value_ = NULL,
      treatment_levels_ = NULL,
      X_names_ = NULL,
      n_treatments_ = NULL
    ),
    class = "SingleTreePolicyInterpreter"
  )
}

#' Interpret a policy from a CATE estimator with a single tree
#'
#' Fits a classification tree that assigns treatment (or control) to maximize
#' expected outcome; tree summarizes the policy by covariates.
#'
#' @param obj \code{SingleTreePolicyInterpreter} object.
#' @param cate_estimator Fitted CATE estimator with \code{predict(..., newdata)}.
#' @param X Feature matrix or data.frame.
#' @param sample_treatment_costs optional numeric vector or matrix of treatment
#'   costs (same length as nrow(X) or scalar). Subtracted from CATE to get
#'   net value.
#' @param ... Not used.
#' @return The interpreter object (invisibly) with \code{tree_model_},
#'   \code{policy_value_}, \code{always_treat_value_}, \code{node_dict_}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # interpret.SingleTreePolicyInterpreter(...)
#' }
#' @export
interpret.SingleTreePolicyInterpreter <- function(obj, cate_estimator, X,
                                                  sample_treatment_costs = NULL, ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  n <- nrow(X)
  obj$X_names_ <- colnames(X)
  if (is.null(obj$X_names_)) obj$X_names_ <- paste0("X", seq_len(ncol(X)))

  y_pred <- get_cate_from_estimator(cate_estimator, X)
  # Ensure matrix: (n x n_treatments), with control=0 as first column implied
  if (is.vector(y_pred)) y_pred <- matrix(y_pred, ncol = 1L)
  if (nrow(y_pred) != n) stop("CATE predictions must have nrow equal to nrow(X).")
  n_t <- ncol(y_pred)
  obj$n_treatments_ <- n_t
  # Value matrix: [0, cate_1, cate_2, ...] for control and each treatment
  value_matrix <- cbind(0, y_pred)
  if (!is.null(sample_treatment_costs)) {
    if (length(sample_treatment_costs) == 1L)
      value_matrix[, -1] <- value_matrix[, -1] - sample_treatment_costs
    else {
      if (is.vector(sample_treatment_costs)) sample_treatment_costs <- matrix(sample_treatment_costs, ncol = 1L)
      if (nrow(sample_treatment_costs) == n && ncol(sample_treatment_costs) == n_t)
        value_matrix[, -1] <- value_matrix[, -1] - sample_treatment_costs
      else if (nrow(sample_treatment_costs) == n && ncol(sample_treatment_costs) == 1L)
        value_matrix[, -1] <- value_matrix[, -1] - as.numeric(sample_treatment_costs)
      else
        stop("sample_treatment_costs must be scalar or (n x n_treatments) or (n x 1).")
    }
  }

  # Best treatment index per row (1 = control, 2 = first treatment, ...)
  best_idx <- max.col(value_matrix, ties.method = "first")
  value_at_best <- value_matrix[cbind(seq_len(n), best_idx)]

  # Weights for tree: use value at best (or absolute value) so splits favor high-value regions
  wts <- pmax(value_at_best, 1e-6)
  if (any(value_at_best <= 0)) wts <- pmax(abs(value_at_best), 1e-6)

  df <- as.data.frame(X)
  colnames(df) <- obj$X_names_
  df$best <- factor(best_idx, levels = seq_len(ncol(value_matrix)))

  control <- rpart::rpart.control(
    minsplit = obj$min_samples_split,
    minbucket = obj$min_samples_leaf,
    cp = 0,
    maxdepth = if (is.null(obj$max_depth)) 30L else obj$max_depth
  )
  if (!is.null(obj$max_depth)) control$maxdepth <- obj$max_depth

  set_seed(obj$random_state)
  obj$tree_model_ <- rpart::rpart(best ~ ., data = df, weights = wts, method = "class", control = control, ...)

  # Policy value: average value when following the learned policy
  pred_value <- value_matrix[cbind(seq_len(n), best_idx)]
  obj$policy_value_ <- mean(pred_value, na.rm = TRUE)
  # Always-treat value: average CATE (average over treatments for multi)
  obj$always_treat_value_ <- if (n_t == 1L) mean(y_pred[, 1L], na.rm = TRUE) else mean(rowMeans(y_pred, na.rm = TRUE), na.rm = TRUE)

  # Node dict: mean and std of value (or CATE) in each node
  node_to_obs <- rpart_node_to_obs(obj$tree_model_, X)
  node_dict <- list()
  for (i in seq_along(node_to_obs)) {
    idx <- node_to_obs[[i]]
    if (length(idx) == 0) {
      node_dict[[i]] <- list(mean = NA_real_, std = NA_real_)
    } else {
      v_node <- value_at_best[idx]
      node_dict[[i]] <- list(mean = mean(v_node, na.rm = TRUE), std = sd(v_node, na.rm = TRUE))
    }
  }
  obj$node_dict_ <- node_dict
  obj$treatment_levels_ <- c(0, seq_len(n_t))  # 0 = control, 1..n_t = treatments
  invisible(obj)
}

#' Assign treatment using the interpreted policy tree
#'
#' @param obj fitted \code{SingleTreePolicyInterpreter}.
#' @param X feature matrix or data.frame.
#' @param ... Not used.
#' @return Integer vector of recommended treatment (0 = control, 1 = first
#'   treatment, ...). If the tree was fit with a single CATE column, returns
#'   0 or 1.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # treat.SingleTreePolicyInterpreter(...)
#' }
#' @export
treat.SingleTreePolicyInterpreter <- function(obj, X, ...) {
  if (is.null(obj$tree_model_))
    stop("interpret() must be called before treat().")
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  df <- as.data.frame(X)
  if (!is.null(obj$X_names_) && ncol(df) == length(obj$X_names_))
    colnames(df) <- obj$X_names_
  pred_class <- predict(obj$tree_model_, newdata = df, type = "class")
  idx <- as.integer(as.character(pred_class))
  # Map index to treatment level (0, 1, 2, ...)
  obj$treatment_levels_[idx]
}

#' Predict recommended treatment from SingleTreePolicyInterpreter (alias for treat)
#'
#' @param object fitted \code{SingleTreePolicyInterpreter}.
#' @param newdata feature matrix or data.frame.
#' @param ... Not used.
#' @return Integer vector of recommended treatment (0 = control, 1 = first treatment, ...).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.SingleTreePolicyInterpreter(...)
#' }
#' @export
predict.SingleTreePolicyInterpreter <- function(object, newdata, ...) {
  treat.SingleTreePolicyInterpreter(object, newdata, ...)
}


# --------------- Shared ---------------

set_seed <- function(seed) {
  if (!is.null(seed)) set.seed(seed)
}
