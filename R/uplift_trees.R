# CausalML-R: Uplift trees and forests (KL, ED, Chi, CTS, DDP, IDDP, IT, CIT)
# Aligned with Python causalml UpliftTreeClassifier / UpliftRandomForestClassifier.
# References: Rzepakowski & Jaroszewicz (2012), CausalML methodology docs

# Uplift Random Forest: bagged uplift trees with divergence criterion
#' Uplift random forest using KL divergence splitting criterion
#' @param X covariate matrix
#' @param treatment treatment 0/1 (or 0,1,...,k for multi-arm; control = 0)
#' @param y outcome (binary for KL/ED/Chi; continuous uses T-learner fallback)
#' @param n_trees number of trees
#' @param min_node_size minimum node size (Python min_samples_leaf)
#' @param min_samples_treatment minimum samples per treatment in node (Python min_samples_treatment)
#' @param max_depth maximum tree depth
#' @param max_features number of features to consider per split (NULL = all; Python max_features)
#' @param n_reg regularization weight for parent node (Python n_reg)
#' @param normalization use normalization factor for gain (Python normalization)
#' @param random_state seed for reproducibility
#' @param n_cores number of cores for building trees (1 = sequential; >1 uses parallel::mclapply on Unix/mac)
#' @param ... passed to ranger when y is continuous
#' @return object with predict method for CATE
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # uplift_rf_kl(...)
#' }
#' @export
uplift_rf_kl <- function(X, treatment, y, n_trees = 100, min_node_size = 10, min_samples_treatment = 10,
                         max_depth = 5, max_features = NULL, n_reg = 100, normalization = TRUE,
                         random_state = NULL, n_cores = 1L, ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  w <- as.integer(treatment)
  # For binary y use divergence-based trees; for continuous use T-learner forest (no binary split criterion)
  binary_y <- all(y %in% c(0, 1))
  if (binary_y) {
    n <- nrow(X)
    build_one <- function(b) {
      idx <- sample(n, replace = TRUE)
      build_uplift_tree(X[idx, , drop = FALSE], y[idx], w[idx],
        criterion = "kl", min_node_size = min_node_size, min_samples_treatment = min_samples_treatment,
        max_depth = max_depth, n_reg = n_reg, normalization = normalization,
        max_features = max_features, random_state = if (is.null(random_state)) NULL else random_state + b)
    }
    trees <- parallel_lapply(seq_len(n_trees), build_one, n_cores = n_cores)
    structure(list(trees = trees, X_names = colnames(X), type = "uplift_kl", classes_ = c("control", "treatment")),
              class = "uplift_rf")
  } else {
    df0 <- as.data.frame(X[w == 0, , drop = FALSE]); df0$y <- y[w == 0]
    df1 <- as.data.frame(X[w == 1, , drop = FALSE]); df1$y <- y[w == 1]
    fit0 <- ranger::ranger(y ~ ., data = df0, num.trees = n_trees, min.node.size = min_node_size, importance = "impurity", ...)
    fit1 <- ranger::ranger(y ~ ., data = df1, num.trees = n_trees, min.node.size = min_node_size, importance = "impurity", ...)
    structure(list(fit_0 = fit0, fit_1 = fit1, X_names = colnames(X), type = "tlearner"),
              class = "uplift_rf")
  }
}

#' Uplift RF with Euclidean distance criterion
#' @param n_cores number of cores for building trees (1 = sequential)
#' @return
#' Object returned by \code{uplift_rf_ed}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # uplift_rf_ed(...)
#' }
#' @export
uplift_rf_ed <- function(X, treatment, y, n_trees = 100, min_node_size = 10, min_samples_treatment = 10,
                         max_depth = 5, max_features = NULL, n_reg = 100, normalization = TRUE, random_state = NULL, n_cores = 1L, ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  w <- as.integer(treatment)
  if (!all(y %in% c(0, 1))) {
    return(uplift_rf_kl(X, treatment, y, n_trees = n_trees, min_node_size = min_node_size, max_depth = max_depth, n_cores = n_cores))
  }
  n <- nrow(X)
  build_one <- function(b) {
    idx <- sample(n, replace = TRUE)
    build_uplift_tree(X[idx, , drop = FALSE], y[idx], w[idx],
      criterion = "ed", min_node_size = min_node_size, min_samples_treatment = min_samples_treatment,
      max_depth = max_depth, n_reg = n_reg, normalization = normalization,
      max_features = max_features, random_state = if (is.null(random_state)) NULL else random_state + b)
  }
  trees <- parallel_lapply(seq_len(n_trees), build_one, n_cores = n_cores)
  structure(list(trees = trees, X_names = colnames(X), type = "uplift_ed"), class = "uplift_rf")
}

#' Uplift RF with Chi-squared criterion
#' @param n_cores number of cores for building trees (1 = sequential)
#' @return
#' Object returned by \code{uplift_rf_chi}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # uplift_rf_chi(...)
#' }
#' @export
uplift_rf_chi <- function(X, treatment, y, n_trees = 100, min_node_size = 10, min_samples_treatment = 10,
                          max_depth = 5, max_features = NULL, n_reg = 100, normalization = TRUE, random_state = NULL, n_cores = 1L, ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  w <- as.integer(treatment)
  if (!all(y %in% c(0, 1))) {
    return(uplift_rf_kl(X, treatment, y, n_trees = n_trees, min_node_size = min_node_size, max_depth = max_depth, n_cores = n_cores))
  }
  n <- nrow(X)
  build_one <- function(b) {
    idx <- sample(n, replace = TRUE)
    build_uplift_tree(X[idx, , drop = FALSE], y[idx], w[idx],
      criterion = "chi", min_node_size = min_node_size, min_samples_treatment = min_samples_treatment,
      max_depth = max_depth, n_reg = n_reg, normalization = normalization,
      max_features = max_features, random_state = if (is.null(random_state)) NULL else random_state + b)
  }
  trees <- parallel_lapply(seq_len(n_trees), build_one, n_cores = n_cores)
  structure(list(trees = trees, X_names = colnames(X), type = "uplift_chi"), class = "uplift_rf")
}

#' Uplift Random Forest on Contextual Treatment Selection (CTS)
#' Split criterion: expected value of best treatment in children vs parent.
#' @param n_cores number of cores for building trees (1 = sequential)
#' @return
#' Object returned by \code{uplift_rf_cts}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # uplift_rf_cts(...)
#' }
#' @export
uplift_rf_cts <- function(X, treatment, y, n_trees = 100, min_node_size = 10, min_samples_treatment = 10,
                          max_depth = 5, max_features = NULL, n_reg = 100, normalization = TRUE, random_state = NULL, n_cores = 1L, ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  w <- as.integer(treatment)
  n <- nrow(X)
  build_one <- function(b) {
    idx <- sample(n, replace = TRUE)
    build_uplift_tree(X[idx, , drop = FALSE], y[idx], w[idx],
      criterion = "cts", min_node_size = min_node_size, min_samples_treatment = min_samples_treatment,
      max_depth = max_depth, n_reg = n_reg, normalization = normalization,
      max_features = max_features, random_state = if (is.null(random_state)) NULL else random_state + b)
  }
  trees <- parallel_lapply(seq_len(n_trees), build_one, n_cores = n_cores)
  structure(list(trees = trees, X_names = colnames(X), type = "cts"), class = "uplift_rf")
}

#' Uplift tree on delta-delta-p (DDP) criterion; binary outcome only
#' @return
#' Object returned by \code{uplift_tree_ddp}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # uplift_tree_ddp(...)
#' }
#' @export
uplift_tree_ddp <- function(X, treatment, y, min_node_size = 10, max_depth = 5) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  w <- as.integer(treatment)
  if (!all(y %in% c(0, 1))) stop("uplift_tree_ddp requires binary outcome")
  tree <- build_uplift_tree(X, y, w, criterion = "ddp", min_node_size = min_node_size, max_depth = max_depth)
  structure(list(tree = tree, X_names = colnames(X)), class = "uplift_tree_ddp")
}

#' Uplift tree on IDDP criterion; binary outcome only
#' @return
#' Object returned by \code{uplift_tree_iddp}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # uplift_tree_iddp(...)
#' }
#' @export
uplift_tree_iddp <- function(X, treatment, y, min_node_size = 10, max_depth = 5) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  w <- as.integer(treatment)
  if (!all(y %in% c(0, 1))) stop("uplift_tree_iddp requires binary outcome")
  tree <- build_uplift_tree(X, y, w, criterion = "iddp", min_node_size = min_node_size, max_depth = max_depth)
  structure(list(tree = tree, X_names = colnames(X)), class = "uplift_tree_iddp")
}

#' Interaction Tree (IT); binary treatment, continuous or binary outcome.
#' For continuous outcome, leaf tau = E[Y|T=1] - E[Y|T=0]; split uses t-statistic on mean difference (Su et al.).
#' @return
#' Object returned by \code{interaction_tree}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # interaction_tree(...)
#' }
#' @export
interaction_tree <- function(X, treatment, y, min_node_size = 10, max_depth = 5) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  w <- as.integer(treatment)
  tree <- build_uplift_tree(X, y, w, criterion = "it", min_node_size = min_node_size, max_depth = max_depth)
  structure(list(tree = tree, X_names = colnames(X)), class = "interaction_tree")
}

#' Causal Inference Tree (CIT); binary treatment, continuous or binary outcome.
#' For continuous outcome, leaf tau = E[Y|T=1] - E[Y|T=0]; split uses same t-statistic as IT.
#' @return
#' Object returned by \code{causal_inference_tree}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # causal_inference_tree(...)
#' }
#' @export
causal_inference_tree <- function(X, treatment, y, min_node_size = 10, max_depth = 5) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  w <- as.integer(treatment)
  tree <- build_uplift_tree(X, y, w, criterion = "cit", min_node_size = min_node_size, max_depth = max_depth)
  structure(list(tree = tree, X_names = colnames(X)), class = "causal_inference_tree")
}

#' Predict CATE (and optionally probabilities) from uplift_rf object.
#' When full_output=TRUE and model has tree leaves with p0/p1, returns data.frame with control, treatment, delta (Python-style).
#' @param object uplift_rf object
#' @param newdata matrix or data.frame of features
#' @param full_output if TRUE return data.frame with control (p0), treatment (p1), delta (uplift); otherwise return CATE vector
#' @param ... unused
#' @return
#' Object returned by \code{predict.uplift_rf}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.uplift_rf(...)
#' }
#' @export
predict.uplift_rf <- function(object, newdata, full_output = FALSE, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$fit_0)) {
    p1 <- predict(object$fit_1, data = as.data.frame(newdata))$predictions
    p0 <- predict(object$fit_0, data = as.data.frame(newdata))$predictions
    if (full_output)
      return(data.frame(control = p0, treatment1 = p1, delta_treatment1 = p1 - p0))
    return(p1 - p0)
  }
  preds <- matrix(NA_real_, nrow(newdata), length(object$trees))
  p0_all <- p1_all <- matrix(NA_real_, nrow(newdata), length(object$trees))
  have_full <- FALSE
  for (i in seq_along(object$trees)) {
    out <- predict_uplift_tree(object$trees[[i]], newdata, full_output = full_output)
    if (full_output && is.list(out)) {
      have_full <- TRUE
      preds[, i] <- out$tau
      p0_all[, i] <- out$p0
      p1_all[, i] <- out$p1
    } else {
      preds[, i] <- out
    }
  }
  if (full_output && have_full) {
    return(data.frame(control = rowMeans(p0_all), treatment1 = rowMeans(p1_all),
                      delta_treatment1 = rowMeans(preds)))
  }
  rowMeans(preds)
}

#' Predict from single uplift tree (DDP, IDDP, IT, CIT)
#' @return
#' Object returned by \code{predict.uplift_tree_ddp}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.uplift_tree_ddp(...)
#' }
#' @export
predict.uplift_tree_ddp <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  predict_uplift_tree(object$tree, newdata)
}

#' Predict method for uplift_tree_iddp
#'
#' @return
#' Object returned by \code{predict.uplift_tree_iddp}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.uplift_tree_iddp(...)
#' }
#' @export
predict.uplift_tree_iddp <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  predict_uplift_tree(object$tree, newdata)
}

#' Predict method for interaction_tree
#'
#' @return
#' Object returned by \code{predict.interaction_tree}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.interaction_tree(...)
#' }
#' @export
predict.interaction_tree <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  predict_uplift_tree(object$tree, newdata)
}

#' Predict method for causal_inference_tree
#'
#' @return
#' Object returned by \code{predict.causal_inference_tree}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.causal_inference_tree(...)
#' }
#' @export
predict.causal_inference_tree <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  predict_uplift_tree(object$tree, newdata)
}

# --- Multi-treatment uplift forest (Python UpliftRandomForestClassifier-style) ---
#' Multi-treatment uplift random forest (Python UpliftRandomForestClassifier-style).
#' Fits one uplift forest per treatment arm (control vs treatment_k). predict(..., full_output=TRUE)
#' returns control/treatment probabilities, deltas, recommended_treatment, max_delta.
#' @param X covariate matrix
#' @param treatment character or factor: treatment group (e.g. "control", "treatment1", "treatment2", "treatment3")
#' @param y outcome (binary for KL-based splits; if continuous, each arm uses a T-learner \pkg{ranger} forest)
#' @param control_name name of control level
#' @param n_trees number of trees per forest
#' @param ... passed to uplift_rf_kl (min_node_size, max_depth, etc.)
#' @return object of class uplift_rf_multi with predict(..., full_output=TRUE)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # uplift_rf_multi(...)
#' }
#' @export
uplift_rf_multi <- function(X, treatment, y, control_name = "control", n_trees = 50, ...) {
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  treatment <- as.character(treatment)
  trt_levels <- unique(treatment)
  treatment_names <- setdiff(trt_levels, control_name)
  if (length(treatment_names) == 0) stop("Need at least one non-control treatment")
  models <- list()
  for (nm in treatment_names) {
    idx <- treatment %in% c(control_name, nm)
    models[[nm]] <- uplift_rf_kl(
      X[idx, , drop = FALSE],
      as.integer(treatment[idx] != control_name),
      y[idx],
      n_trees = n_trees, ...
    )
  }
  structure(list(
    models = models,
    control_name = control_name,
    treatment_names = treatment_names,
    classes_ = c(control_name, treatment_names),
    X_names = colnames(X)
  ), class = "uplift_rf_multi")
}

#' Predict from uplift_rf_multi. full_output=TRUE returns Python-style DataFrame.
#' @return
#' Object returned by \code{predict.uplift_rf_multi}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.uplift_rf_multi(...)
#' }
#' @export
predict.uplift_rf_multi <- function(object, newdata, full_output = FALSE, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  n <- nrow(newdata)
  tn <- object$treatment_names
  deltas <- matrix(NA_real_, n, length(tn))
  colnames(deltas) <- tn
  p0_mat <- p1_mat <- matrix(NA_real_, n, length(tn))
  for (k in seq_along(tn)) {
    nm <- tn[k]
    pred <- predict(object$models[[nm]], newdata, full_output = TRUE)
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

# --- Uplift tree/forest visualization (R port of Python causalml/inference/tree/plot.py) ---
# Python: https://github.com/uber/causalml/blob/master/causalml/inference/tree/plot.py
# "Visualization functions for forest of trees-based ensemble methods for Uplift modeling"
# Exports: uplift_tree_string, uplift_tree_plot, uplift_forest_plot, uplift_tree_to_rpart

#' Recursively build text representation of an uplift tree
#' @param tree tree list from build_uplift_tree (leaf = TRUE with tau, n; or split_var, split_val, left, right)
#' @param x_names character vector of feature names
#' @param indent character, current indentation
#' @noRd
uplift_tree_string_impl <- function(tree, x_names, indent = "") {
  var_label <- function(j) {
    if (is.null(x_names) || length(x_names) < j || is.na(x_names[j])) paste0("X", j) else x_names[j]
  }
  if (tree$leaf) {
    return(sprintf("[tau=%.4f, n=%d]", tree$tau, tree$n))
  }
  var_name <- var_label(tree$split_var)
  left_str <- uplift_tree_string_impl(tree$left, x_names, paste0(indent, "    "))
  right_str <- uplift_tree_string_impl(tree$right, x_names, paste0(indent, "    "))
  paste0(
    var_name, " >= ", sprintf("%.4f", tree$split_val), "?\n",
    indent, "  yes -> ", left_str, "\n",
    indent, "  no  -> ", right_str
  )
}

#' Print uplift tree as a string (for interpretation and diagnosis)
#' @param tree tree list (from build_uplift_tree, or object$tree from uplift_tree_ddp, etc.)
#' @param x_names character vector of feature names (optional)
#' @return character string representation of the tree
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # uplift_tree_string(...)
#' }
#' @export
uplift_tree_string <- function(tree, x_names = NULL) {
  if (inherits(tree, "uplift_tree_ddp") || inherits(tree, "uplift_tree_iddp") ||
      inherits(tree, "interaction_tree") || inherits(tree, "causal_inference_tree")) {
    x_names <- tree$X_names
    tree <- tree$tree
  }
  if (is.null(x_names)) x_names <- paste0("X", seq_len(20))
  uplift_tree_string_impl(tree, x_names)
}

#' Compute layout (depth, x position) for each node and edges for plotting
#' @noRd
tree_layout <- function(tree, x_names, depth = 0, x_min = 0, x_max = 1) {
  var_label <- function(j) {
    if (is.null(x_names) || length(x_names) < j) paste0("X", j) else x_names[j]
  }
  empty_edges <- data.frame(from_x = numeric(0), from_depth = integer(0), to_x = numeric(0), to_depth = integer(0))
  if (tree$leaf) {
    return(list(
      nodes = data.frame(depth = depth, x = (x_min + x_max) / 2, leaf = TRUE, tau = tree$tau, n = tree$n,
                        label = sprintf("%.3f\nn=%d", tree$tau, tree$n), stringsAsFactors = FALSE),
      edges = empty_edges,
      x_next = (x_min + x_max) / 2
    ))
  }
  mid <- (x_min + x_max) / 2
  L <- tree_layout(tree$left, x_names, depth + 1, x_min, mid)
  R <- tree_layout(tree$right, x_names, depth + 1, mid, x_max)
  nodes <- rbind(
    data.frame(depth = depth, x = mid, leaf = FALSE, tau = NA_real_, n = NA_integer_,
               label = sprintf("%s >= %.2f", var_label(tree$split_var), tree$split_val),
               stringsAsFactors = FALSE),
    L$nodes, R$nodes
  )
  # Edges: parent (mid, depth) -> left child root, parent -> right child root
  edges <- rbind(
    data.frame(from_x = mid, from_depth = depth, to_x = L$nodes$x[1], to_depth = L$nodes$depth[1], stringsAsFactors = FALSE),
    data.frame(from_x = mid, from_depth = depth, to_x = R$nodes$x[1], to_depth = R$nodes$depth[1], stringsAsFactors = FALSE),
    L$edges, R$edges
  )
  list(nodes = nodes, edges = edges, x_next = R$x_next)
}

#' Plot an uplift tree (base R), with arrows connecting each split to its children
#'
#' Edges are drawn with arrowheads pointing to child nodes (similar to Python CausalML's
#' graphviz-style uplift_tree_plot). Optionally use \code{rpart.plot} when available
#' (see \code{use_rpart_plot}) for a polished rpart-style diagram.
#'
#' @param tree tree list or fitted object (uplift_tree_ddp, etc.)
#' @param x_names character vector of feature names (optional)
#' @param main optional title
#' @param use_rpart_plot if \code{TRUE}, attempt to plot via \pkg{rpart.plot} (requires
#'   converting the tree to \code{rpart} format; needs \pkg{rpart} and \pkg{rpart.plot}).
#'   If \code{FALSE} (default), use the built-in plot with arrows.
#' @param arrow_length size of arrowhead in user coordinates (default 0.08); larger values give bigger arrowheads.
#' @param color_leaves_by_tau if \code{TRUE}, color leaf nodes by uplift (tau): blue for higher,
#'   green for lower (similar to Python causalml \code{plot.py} upliftScoreToColor).
#' @param ... passed to \code{plot()} or \code{rpart.plot()}
#' @return Invisible node layout data (when \code{use_rpart_plot = FALSE}) or the return
#'   value of \code{rpart.plot} when \code{use_rpart_plot = TRUE}.
#' @seealso \code{\link{uplift_tree_string}}, \code{\link{uplift_rf_kl}}, \code{\link{uplift_forest_plot}}.
#'   Python reference: \url{https://github.com/uber/causalml/blob/master/causalml/inference/tree/plot.py}
#' @examples
#' \dontrun{
#' # Basic usage
#' # uplift_tree_plot(...)
#' }
#' @export
uplift_tree_plot <- function(tree, x_names = NULL, main = "Uplift tree", use_rpart_plot = FALSE, arrow_length = 0.08, color_leaves_by_tau = FALSE, ...) {
  if (inherits(tree, "uplift_tree_ddp") || inherits(tree, "uplift_tree_iddp") ||
      inherits(tree, "interaction_tree") || inherits(tree, "causal_inference_tree")) {
    x_names <- tree$X_names
    tree <- tree$tree
  }
  if (is.null(x_names)) x_names <- paste0("X", seq_len(20))

  if (use_rpart_plot) {
    rp <- uplift_tree_to_rpart(tree, x_names)
    if (!is.null(rp)) {
      if (requireNamespace("rpart.plot", quietly = TRUE)) {
        return(rpart.plot::rpart.plot(rp, main = main, roundint = FALSE, ...))
      }
      message("Package 'rpart.plot' not installed; falling back to built-in plot. Install with: install.packages('rpart.plot')")
    } else {
      message("rpart conversion not available; using built-in plot.")
    }
  }

  lay <- tree_layout(tree, x_names)
  nodes <- lay$nodes
  depth_max <- max(nodes$depth)
  x_range <- range(nodes$x)
  d <- diff(x_range)
  if (d < 1e-6) d <- 1
  x_range <- x_range + c(-0.1, 0.1) * d
  if (depth_max == 0) {
    plot(1, 1, type = "n", xaxt = "n", yaxt = "n", xlab = "", ylab = "", main = main, ...)
    text(1, 1, nodes$label[1], cex = 0.9)
    return(invisible(nodes))
  }
  y_pos <- depth_max - nodes$depth
  plot(nodes$x, y_pos, type = "n", xlim = x_range, ylim = c(-0.5, depth_max + 0.5),
       xaxt = "n", yaxt = "n", xlab = "", ylab = "depth", main = main, ...)
  # Draw edges with arrows linking each split to its children (arrowheads in user coords so they scale)
  if (nrow(lay$edges) > 0) {
    from_x <- lay$edges$from_x
    from_y <- depth_max - lay$edges$from_depth
    to_x   <- lay$edges$to_x
    to_y   <- depth_max - lay$edges$to_depth
    # Shorten line so arrow tip ends at node box edge
    box_inset <- 0.18
    dx <- to_x - from_x
    dy <- to_y - from_y
    L <- sqrt(dx * dx + dy * dy)
    L[L < 1e-8] <- 1
    end_x <- from_x + dx * (1 - box_inset / L)
    end_y <- from_y + dy * (1 - box_inset / L)
    # Line from parent to child
    segments(from_x, from_y, end_x, end_y, col = "gray30", lwd = 1.5)
    # Arrowhead in user coordinates (visible at any fig size / par(mfrow), e.g. ctrees in 17 notebook)
    h <- max(0.03, arrow_length)
    angle_deg <- 22
    angle_rad <- angle_deg * pi / 180
    for (k in seq_len(nrow(lay$edges))) {
      ux <- dx[k] / L[k]
      uy <- dy[k] / L[k]
      theta <- atan2(uy, ux)
      x1 <- end_x[k] - h * cos(theta + angle_rad)
      y1 <- end_y[k] - h * sin(theta + angle_rad)
      x2 <- end_x[k] - h * cos(theta - angle_rad)
      y2 <- end_y[k] - h * sin(theta - angle_rad)
      segments(end_x[k], end_y[k], x1, y1, col = "gray30", lwd = 1.5)
      segments(end_x[k], end_y[k], x2, y2, col = "gray30", lwd = 1.5)
    }
  }
  # Leaf colors: optional gradient by tau (Python plot.py upliftScoreToColor: blue = higher, green = lower)
  leaf_col <- rep(NA_character_, nrow(nodes))
  if (color_leaves_by_tau && any(nodes$leaf)) {
    taus <- nodes$tau[nodes$leaf]
    rng <- range(taus, na.rm = TRUE)
    if (diff(rng) < 1e-10) rng[2] <- rng[1] + 1
    lvl <- (nodes$tau - rng[1]) / (rng[2] - rng[1])
    lvl[is.na(lvl)] <- 0.5
    base_lvl <- 0.5
    for (ii in which(nodes$leaf)) {
      ll <- lvl[ii]
      if (ll >= base_lvl) {
        leaf_col[ii] <- rgb(0.12 + (ll - base_lvl) * 0.8, 0.47, 0.71, maxColorValue = 1)
      } else {
        leaf_col[ii] <- rgb(0, 0.5 + ll * 0.5, 0.2, maxColorValue = 1)
      }
    }
  }

  for (i in seq_len(nrow(nodes))) {
    if (nodes$leaf[i]) {
      col_use <- if (color_leaves_by_tau && !is.na(leaf_col[i])) leaf_col[i] else "lightyellow"
      rect(nodes$x[i] - 0.08, y_pos[i] - 0.15, nodes$x[i] + 0.08, y_pos[i] + 0.15,
           col = col_use, border = "gray")
    } else {
      rect(nodes$x[i] - 0.12, y_pos[i] - 0.18, nodes$x[i] + 0.12, y_pos[i] + 0.18,
           col = "lightblue", border = "gray")
    }
    text(nodes$x[i], y_pos[i], nodes$label[i], cex = 0.65)
  }
  invisible(nodes)
}

#' Plot multiple trees from an uplift forest (ensemble visualization)
#'
#' Visualization for forest of trees-based ensemble: plots a grid of individual trees
#' from an \code{uplift_rf} object. Mirrors the Python causalml inference/tree module
#' usage where you visualize trees from \code{UpliftRandomForestClassifier} (e.g.
#' \code{uplift_model.uplift_forest[0]}, \code{uplift_forest[1]}, ...).
#'
#' @param forest an \code{uplift_rf} object (from \code{uplift_rf_kl}, \code{uplift_rf_ed},
#'   \code{uplift_rf_chi}, or \code{uplift_rf_cts}).
#' @param x_names character vector of feature names (default: \code{forest$X_names}).
#' @param which_trees integer indices of trees to plot (default: first 4, or fewer if forest has fewer).
#' @param mfrow \code{c(nr, nc)} for \code{par(mfrow)}. If \code{NULL}, set from length of \code{which_trees}.
#' @param main_prefix prefix for each panel title (default \code{"Tree "}); title becomes \code{main_prefix}{index}.
#' @param ... passed to \code{uplift_tree_plot} for each tree (e.g. \code{arrow_length}, \code{color_leaves_by_tau}).
#' @return Invisible list of layout data from each \code{uplift_tree_plot} call.
#' @seealso \code{\link{uplift_tree_plot}}, \code{\link{uplift_tree_string}}, \code{\link{uplift_rf_kl}}.
#'   Python: \url{https://github.com/uber/causalml/tree/master/causalml/inference/tree}
#' @examples
#' \dontrun{
#' # Basic usage
#' # uplift_forest_plot(...)
#' }
#' @export
uplift_forest_plot <- function(forest, x_names = NULL, which_trees = NULL, mfrow = NULL, main_prefix = "Tree ", ...) {
  if (!inherits(forest, "uplift_rf")) stop("forest must be an uplift_rf object")
  trees <- forest$trees
  n_trees <- length(trees)
  if (n_trees == 0L) stop("forest has no trees")
  if (is.null(x_names)) x_names <- forest$X_names
  if (is.null(which_trees)) which_trees <- seq_len(min(4L, n_trees))
  which_trees <- as.integer(which_trees)
  which_trees <- which_trees[which_trees >= 1L & which_trees <= n_trees]
  if (length(which_trees) == 0L) stop("no valid tree indices in which_trees")
  if (is.null(mfrow)) {
    n <- length(which_trees)
    nr <- if (n <= 2L) 1L else if (n <= 4L) 2L else ceiling(sqrt(n))
    nc <- ceiling(n / nr)
    mfrow <- c(nr, nc)
  }
  opar <- par(mfrow = mfrow)
  on.exit(par(opar))
  out <- vector("list", length(which_trees))
  for (j in seq_along(which_trees)) {
    idx <- which_trees[j]
    main <- paste0(main_prefix, idx)
    out[[j]] <- uplift_tree_plot(trees[[idx]], x_names = x_names, main = main, ...)
  }
  invisible(out)
}

#' Convert an uplift tree (list) to an rpart object for use with rpart.plot
#'
#' Builds a minimal \code{rpart} object so that \code{rpart.plot::rpart.plot()} can
#' be used. Useful when you want the polished layout and options of the
#' \pkg{rpart.plot} package (see \url{http://www.milbo.org/rpart-plot/index.html}).
#'
#' @param tree tree list (from \code{build_uplift_tree} or \code{uplift_rf_kl(...)$trees[[1]]})
#' @param x_names character vector of feature names
#' @return An object of class \code{rpart} that can be passed to \code{rpart.plot()},
#'   or \code{NULL} if \pkg{rpart} is not available.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # uplift_tree_to_rpart(...)
#' }
#' @export
uplift_tree_to_rpart <- function(tree, x_names = NULL) {
  if (!requireNamespace("rpart", quietly = TRUE)) return(NULL)
  if (is.null(x_names)) x_names <- paste0("X", seq_len(20))
  var_label <- function(j) {
    if (is.null(x_names) || length(x_names) < j || is.na(x_names[j])) paste0("X", j) else x_names[j]
  }
  leaf_tau <- function(tr) {
    if (tr$leaf) tr$tau else mean(c(leaf_tau(tr$left), leaf_tau(tr$right)), na.rm = TRUE)
  }
  node_id <- 0L
  frame_list <- list()
  splits_list <- list()
  .build <- function(tr) {
    node_id <<- node_id + 1L
    this_id <- node_id
    if (tr$leaf) {
      frame_list[[this_id]] <<- data.frame(
        var = "<leaf>", n = tr$n, wt = tr$n, dev = 0, yval = tr$tau,
        complexity = 0.01, ncompete = 0L, nsurrogate = 0L,
        stringsAsFactors = FALSE
      )
      return(list(id = this_id, leaf = TRUE))
    }
    frame_list[[this_id]] <<- data.frame(
      var = var_label(tr$split_var), n = tr$n, wt = tr$n, dev = 1,
      yval = leaf_tau(tr),
      complexity = 0.01, ncompete = 0L, nsurrogate = 0L,
      stringsAsFactors = FALSE
    )
    splits_list[[length(splits_list) + 1L]] <<- c(count = tr$n, ncat = -1L, improve = 1, index = tr$split_val, adj = 0)
    .build(tr$left)
    .build(tr$right)
    list(id = this_id, leaf = FALSE)
  }
  .build(tree)
  frame <- do.call(rbind, frame_list)
  rownames(frame) <- seq_len(nrow(frame))
  splits <- do.call(rbind, splits_list)
  split_vars <- vapply(seq_along(frame_list), function(i) frame_list[[i]]$var, "")
  rownames(splits) <- split_vars[split_vars != "<leaf>"]
  obj <- list(
    frame = frame,
    splits = splits,
    call = call("uplift_tree_to_rpart"),
    terms = NULL,
    cptable = matrix(c(0.01, 1, 0, 0, 0), nrow = 1,
                     dimnames = list("1", c("CP", "nsplit", "rel error", "xerror", "xstd"))),
    method = "anova",
    parms = list(),
    control = rpart::rpart.control(),
    functions = list(summary = function(y, wt, ...) list(label = "tau", dev = 0))
  )
  class(obj) <- "rpart"
  obj
}
