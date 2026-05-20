# grf integration: Re-exports from the grf package with proper citation
#
# RCausalML re-exports selected functions from the grf (Generalized Random Forests)
# package so users can access them via RCausalML when grf is installed. All
# implementations and algorithms are from grf; see references in each function.
#
# Citation for grf:
#   Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.
#   Annals of Statistics, 47(2), 1148-1178.
#   R package: https://github.com/grf-labs/grf

NULL

# =============================================================================
# Conflicting names: RCausalML has its own causal_forest and average_treatment_effect.
# We re-export grf's versions with a grf_ prefix so both are available.
# =============================================================================

.check_grf <- function() {
  if (!requireNamespace("grf", quietly = TRUE))
    stop("Package 'grf' is required for this function. Install with: install.packages(\"grf\")")
}

#' Causal forest (grf implementation)
#'
#' Re-export of \code{\link[grf]{causal_forest}} from the \pkg{grf} package.
#' Trains a causal forest for estimating conditional average treatment effects
#' tau(X). Requires the \pkg{grf} package.
#'
#' Use \code{\link{causal_forest}} for RCausalML's built-in causal forest.
#' Use \code{grf_causal_forest} for the grf implementation (honest splitting,
#' tuning, confidence intervals).
#'
#' @param X The covariates.
#' @param Y The outcome.
#' @param W The treatment assignment.
#' @param ... All other arguments passed to \code{\link[grf]{causal_forest}}.
#' @return A trained causal forest object (class \code{causal_forest}, \code{grf}).
#'
#' @references
#' Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.
#' \emph{Annals of Statistics}, 47(2), 1148-1178.
#'
#' Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous
#' treatment effects using random forests. \emph{Journal of the American
#' Statistical Association}, 113(523), 1228-1242.
#'
#' Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous
#' treatment effects. \emph{Biometrika}, 108(2), 299-319.
#'
#' @seealso \code{\link[grf]{causal_forest}}, \code{\link{causal_forest}},
#' \code{\link{grf_average_treatment_effect}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # grf_causal_forest(...)
#' }
#' @export
grf_causal_forest <- function(X, Y, W, ...) {
  .check_grf()
  grf::causal_forest(X, Y, W, ...)
}

#' Average treatment effect (grf implementation)
#'
#' Re-export of \code{\link[grf]{average_treatment_effect}} from the \pkg{grf}
#' package. Computes doubly robust estimates of the average treatment effect
#' from a trained grf forest. Requires the \pkg{grf} package.
#'
#' Use \code{\link{average_treatment_effect}} for RCausalML's built-in
#' implementation. Use \code{grf_average_treatment_effect} for grf forests
#' (AIPW/TMLE, target.sample options).
#'
#' @param forest A trained forest from \pkg{grf} (e.g. \code{grf_causal_forest},
#'   \code{instrumental_forest}, \code{causal_survival_forest}).
#' @param ... All other arguments passed to \code{\link[grf]{average_treatment_effect}}.
#' @return Estimate and standard error (or data frame for multi-arm forests).
#'
#' @references
#' Athey, S., & Wager, S. (2021). Policy learning with observational data.
#' \emph{Econometrica}, 89(1), 133-161.
#'
#' Robins, J. M., & Rotnitzky, A. (1995). Semiparametric efficiency in
#' multivariate regression models with missing data. \emph{JASA}, 90(429), 122-129.
#'
#' @seealso \code{\link[grf]{average_treatment_effect}}, \code{\link{average_treatment_effect}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # grf_average_treatment_effect(...)
#' }
#' @export
grf_average_treatment_effect <- function(forest, ...) {
  .check_grf()
  grf::average_treatment_effect(forest, ...)
}

# =============================================================================
# Non-conflicting grf functions: re-export with original names and citation
# =============================================================================

#' Causal survival forest (grf)
#'
#' Re-export of \code{\link[grf]{causal_survival_forest}} from the \pkg{grf}
#' package. Trains a causal survival forest for right-censored outcomes.
#' Requires the \pkg{grf} package.
#'
#' @param X The covariates.
#' @param Y The event time.
#' @param W The treatment assignment.
#' @param D The event type (0: censored, 1: failure).
#' @param ... All other arguments passed to \code{\link[grf]{causal_survival_forest}}.
#' @return A trained \code{causal_survival_forest} object.
#'
#' @references
#' Cui, Y., Kosorok, M. R., Sverdrup, E., Wager, S., & Zhu, R. (2023).
#' Estimating heterogeneous treatment effects with right-censored data via
#' causal survival forests. \emph{Journal of the Royal Statistical Society:
#' Series B}, 85(2), 497-523.
#'
#' @seealso \code{\link[grf]{causal_survival_forest}}, \code{\link{grf_average_treatment_effect}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # causal_survival_forest(...)
#' }
#' @export
causal_survival_forest <- function(X, Y, W, D, ...) {
  .check_grf()
  grf::causal_survival_forest(X, Y, W, D, ...)
}

#' Instrumental forest (grf)
#'
#' Re-export of \code{\link[grf]{instrumental_forest}} from the \pkg{grf}
#' package. Trains an instrumental forest for conditional local average
#' treatment effects using instruments. Formally, the forest estimates
#' tau(X) = Cov[Y, Z | X = x] / Cov[W, Z | X = x]. When Z and W coincide,
#' an instrumental forest is equivalent to a causal forest. Requires the
#' \pkg{grf} package.
#'
#' @param X The covariates used in the instrumental regression.
#' @param Y The outcome.
#' @param W The treatment assignment (may be binary or real).
#' @param Z The instrument (may be binary or real).
#' @param ... All other arguments passed to \code{\link[grf]{instrumental_forest}},
#'   e.g. \code{Y.hat}, \code{W.hat}, \code{Z.hat}, \code{num.trees},
#'   \code{sample.weights}, \code{clusters}, \code{sample.fraction}, \code{mtry},
#'   \code{min.node.size}, \code{honesty}, \code{honesty.fraction},
#'   \code{honesty.prune.leaves}, \code{alpha}, \code{imbalance.penalty},
#'   \code{stabilize.splits}, \code{ci.group.size}, \code{reduced.form.weight},
#'   \code{tune.parameters}, \code{compute.oob.predictions}, \code{num.threads},
#'   \code{seed}.
#' @return A trained \code{instrumental_forest} object.
#'
#' @references
#' Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.
#' \emph{Annals of Statistics}, 47(2), 1148-1178.
#'
#' @seealso \code{\link[grf]{instrumental_forest}}, \code{\link{predict.instrumental_forest}},
#' \code{\link{grf_average_treatment_effect}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # instrumental_forest(...)
#' }
#' @export
instrumental_forest <- function(X, Y, W, Z, ...) {
  .check_grf()
  grf::instrumental_forest(X, Y, W, Z, ...)
}

#' Predict with an instrumental forest (grf)
#'
#' Re-export of \code{\link[grf]{predict.instrumental_forest}}. Gets estimates
#' of tau(x) using a trained instrumental forest. Requires the \pkg{grf} package.
#'
#' @param object The trained forest (from \code{\link{instrumental_forest}}).
#' @param newdata Points for prediction. If NULL, OOB predictions on training set.
#' @param num.threads Number of threads. Default NULL.
#' @param estimate.variance Whether to estimate variance for confidence intervals.
#' @param ... Additional arguments passed to \code{\link[grf]{predict.instrumental_forest}}.
#' @return A data frame with \code{predictions} and optionally \code{debiased.error}.
#'
#' @seealso \code{\link{instrumental_forest}}, \code{\link[grf]{predict.instrumental_forest}}
#' @method predict instrumental_forest
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.instrumental_forest(...)
#' }
#' @export
predict.instrumental_forest <- function(object, newdata = NULL,
                                        num.threads = NULL,
                                        estimate.variance = FALSE,
                                        ...) {
  .check_grf()
  getFromNamespace("predict.instrumental_forest", "grf")(object, newdata = newdata,
                                                          num.threads = num.threads,
                                                          estimate.variance = estimate.variance,
                                                          ...)
}

#' Multi-arm causal forest (grf)
#'
#' Re-export of \code{\link[grf]{multi_arm_causal_forest}} from the \pkg{grf}
#' package. Trains a causal forest for multiple treatment arms. Requires the
#' \pkg{grf} package.
#'
#' @param X The covariates.
#' @param Y The outcome (vector or matrix for multiple outcomes).
#' @param W The treatment assignment (factor).
#' @param ... All other arguments passed to \code{\link[grf]{multi_arm_causal_forest}}.
#' @return A trained \code{multi_arm_causal_forest} object.
#'
#' @references
#' Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.
#' \emph{Annals of Statistics}, 47(2), 1148-1178.
#'
#' Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous
#' treatment effects. \emph{Biometrika}, 108(2), 299-319.
#'
#' @seealso \code{\link[grf]{multi_arm_causal_forest}}, \code{\link{grf_average_treatment_effect}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # multi_arm_causal_forest(...)
#' }
#' @export
multi_arm_causal_forest <- function(X, Y, W, ...) {
  .check_grf()
  grf::multi_arm_causal_forest(X, Y, W, ...)
}

#' Rank-weighted average treatment effect (grf)
#'
#' Re-export of \code{\link[grf]{rank_average_treatment_effect}} from the
#' \pkg{grf} package. Estimates the rank-weighted average treatment effect
#' (RATE) and targeting operator characteristic (TOC). Requires the \pkg{grf}
#' package.
#'
#' @param forest An evaluation forest from \pkg{grf}.
#' @param priorities Treatment prioritization scores.
#' @param ... All other arguments passed to \code{\link[grf]{rank_average_treatment_effect}}.
#' @return An object of class \code{rank_average_treatment_effect} with
#'   \code{estimate}, \code{std.err}, \code{target}, and \code{TOC}.
#'
#' @references
#' Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., & Wager, S. (2025).
#' Evaluating treatment prioritization rules via rank-weighted average
#' treatment effects. \emph{Journal of the American Statistical Association},
#' 120(549), 1-12.
#'
#' @seealso \code{\link[grf]{rank_average_treatment_effect}}, \code{\link{plot.rank_average_treatment_effect}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # rank_average_treatment_effect(...)
#' }
#' @export
rank_average_treatment_effect <- function(forest, priorities, ...) {
  .check_grf()
  grf::rank_average_treatment_effect(forest, priorities, ...)
}

#' Rank-weighted average treatment effect fit (grf)
#'
#' Re-export of \code{\link[grf]{rank_average_treatment_effect.fit}} from the
#' \pkg{grf} package. Computes RATE with user-supplied doubly robust scores.
#' Requires the \pkg{grf} package.
#'
#' @param DR.scores Doubly robust evaluation scores.
#' @param priorities Treatment prioritization scores.
#' @param ... All other arguments passed to \code{\link[grf]{rank_average_treatment_effect.fit}}.
#' @return An object of class \code{rank_average_treatment_effect}.
#'
#' @references
#' Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., & Wager, S. (2025).
#' Evaluating treatment prioritization rules via rank-weighted average
#' treatment effects. \emph{JASA}, 120(549), 1-12.
#'
#' @seealso \code{\link[grf]{rank_average_treatment_effect.fit}}, \code{\link{rank_average_treatment_effect}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # rank_average_treatment_effect.fit(...)
#' }
#' @export
rank_average_treatment_effect.fit <- function(DR.scores, priorities, ...) {
  .check_grf()
  grf::rank_average_treatment_effect.fit(DR.scores, priorities, ...)
}

# =============================================================================
# S3 plot methods from grf (so plot(rate) and plot(tree) work when using
# our re-exported functions)
# =============================================================================

#' Plot rank_average_treatment_effect (grf)
#'
#' Re-export of \code{\link[grf]{plot.rank_average_treatment_effect}}. Plots the
#' targeting operator characteristic (TOC) curve. Requires the \pkg{grf} package.
#'
#' @param x Output of \code{\link{rank_average_treatment_effect}}.
#' @param ... Additional arguments passed to the underlying plot.
#' @param ci.args List of arguments passed to \code{points} for confidence bars (e.g. \code{lty = 2}).
#' @param abline.args List of arguments passed to \code{abline} (e.g. \code{h = 0, lty = 3}).
#' @param legend.args List of arguments passed to \code{legend} (e.g. \code{x = "topright"}).
#' @seealso \code{\link{rank_average_treatment_effect}}, \code{\link[grf]{plot.rank_average_treatment_effect}}
#' @return
#' Object returned by \code{plot.rank_average_treatment_effect}.
#' @examples
#' \dontrun{
#' # Basic usage
#' # plot.rank_average_treatment_effect(...)
#' }
#' @export
plot.rank_average_treatment_effect <- function(x, ..., ci.args = list(), abline.args = list(), legend.args = list()) {
  .check_grf()
  getFromNamespace("plot.rank_average_treatment_effect", "grf")(x, ..., ci.args = ci.args, abline.args = abline.args, legend.args = legend.args)
}

#' Plot grf tree (grf)
#'
#' Re-export of \code{\link[grf]{plot.grf_tree}}. Plots a single tree from a
#' grf forest (requires \pkg{DiagrammeR}). NA path is shown by arrow fill when
#' the forest was trained with missing values. Requires the \pkg{grf} package.
#'
#' @param x A tree from \code{\link{get_tree}}.
#' @param include.na.path Whether to show the path of missing values. Defaults to
#'   whether the forest was trained with NAs.
#' @param ... Additional arguments (currently ignored).
#' @seealso \code{\link{get_tree}}, \code{\link[grf]{plot.grf_tree}}
#' @return
#' Object returned by \code{plot.grf_tree}.
#' @examples
#' \dontrun{
#' # Basic usage
#' # plot.grf_tree(...)
#' }
#' @export
plot.grf_tree <- function(x, include.na.path = NULL, ...) {
  .check_grf()
  getFromNamespace("plot.grf_tree", "grf")(x, include.na.path = include.na.path, ...)
}

# =============================================================================
# Forest summary / calibration from grf (test_calibration, best_linear_projection)
# =============================================================================

#' Test calibration of a grf forest
#'
#' Re-export of \code{\link[grf]{test_calibration}} from the \pkg{grf} package.
#' Omnibus evaluation of the quality of the random forest estimates via
#' calibration. Computes the best linear fit of the target estimand using the
#' forest prediction (on held-out data) and the mean forest prediction as
#' regressors. A coefficient of 1 for \code{mean.forest.prediction} suggests
#' the mean forest prediction is correct; a coefficient of 1 for
#' \code{differential.forest.prediction} suggests heterogeneity estimates are
#' well calibrated. The p-value of \code{differential.forest.prediction} also
#' acts as an omnibus test for the presence of heterogeneity. Requires the
#' \pkg{grf} package.
#'
#' @param forest A trained forest from \pkg{grf} (e.g. \code{grf_causal_forest},
#'   \code{regression_forest}).
#' @param vcov.type Optional covariance type for standard errors (e.g. "HC0" to
#'   "HC3"). Default "HC3" (recommended in small samples).
#' @return A heteroskedasticity-consistent test of calibration (matrix with
#'   coefficients, SEs, t, p-value).
#'
#' @references
#' Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to
#' cluster-robust inference. \emph{Journal of Human Resources}, 50(2), 317-372.
#' Chernozhukov, V., Demirer, M., Duflo, E., & Fernandez-Val, I. (2017). Generic
#' machine learning inference on heterogenous treatment effects in randomized
#' experiments. arXiv:1712.04802.
#' MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
#' covariance matrix estimators with improved finite sample properties.
#' \emph{Journal of Econometrics}, 29(3), 305-325.
#'
#' @seealso \code{\link[grf]{test_calibration}}, \code{\link{grf_causal_forest}},
#' \code{\link{best_linear_projection}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # test_calibration(...)
#' }
#' @export
test_calibration <- function(forest, vcov.type = "HC3") {
  .check_grf()
  grf::test_calibration(forest, vcov.type = vcov.type)
}

#' Best linear projection of CATE (grf)
#'
#' Re-export of \code{\link[grf]{best_linear_projection}} from the \pkg{grf}
#' package. Estimates the best linear projection of the conditional average
#' treatment effect tau(X) onto user-provided covariates A, i.e. fits
#' tau(Xi) ~ beta_0 + Ai * beta via regression of doubly robust scores from the
#' forest on A. The case of the null model is equivalent to average treatment
#' effect via AIPW. Requires the \pkg{grf} package.
#'
#' @param forest A trained forest from \pkg{grf} (e.g. \code{grf_causal_forest},
#'   \code{instrumental_forest}, \code{causal_survival_forest}).
#' @param A Covariates to project the CATE onto (matrix or NULL for intercept-only).
#' @param subset Optional subset of training indices. Define only using features
#'   Xi, not treatment or outcome.
#' @param debiasing.weights Optional vector of debiasing weights; if NULL,
#'   obtained via doubly robust score construction.
#' @param compliance.score For instrumental forests only: estimate of effect of
#'   Z on W; if not provided, estimated via auxiliary causal forest.
#' @param num.trees.for.weights Number of trees for auxiliary forests when
#'   computing debiasing weights (default 500).
#' @param vcov.type Covariance type for standard errors (default "HC3").
#' @param target.sample \code{"all"} (default) or \code{"overlap"} (weights
#'   e(X)(1 - e(X))).
#' @return Coefficient estimates and cluster- and heteroskedasticity-robust
#'   standard errors.
#'
#' @references
#' Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to
#' cluster-robust inference. \emph{Journal of Human Resources}, 50(2), 317-372.
#' Cui, Y., Kosorok, M. R., Sverdrup, E., Wager, S., & Zhu, R. (2023).
#' Estimating heterogeneous treatment effects with right-censored data via
#' causal survival forests. \emph{JRSS Series B}, 85(2), 497-523.
#' MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
#' covariance matrix estimators. \emph{Journal of Econometrics}, 29(3), 305-325.
#' Semenova, V., & Chernozhukov, V. (2021). Debiased machine learning of
#' conditional average treatment effects and other causal functions.
#' \emph{The Econometrics Journal}, 24(2).
#'
#' @seealso \code{\link[grf]{best_linear_projection}}, \code{\link{grf_causal_forest}},
#' \code{\link{grf_average_treatment_effect}}, \code{\link{test_calibration}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # best_linear_projection(...)
#' }
#' @export
best_linear_projection <- function(forest,
                                   A = NULL,
                                   subset = NULL,
                                   debiasing.weights = NULL,
                                   compliance.score = NULL,
                                   num.trees.for.weights = 500,
                                   vcov.type = "HC3",
                                   target.sample = c("all", "overlap")) {
  .check_grf()
  grf::best_linear_projection(forest,
                              A = A,
                              subset = subset,
                              debiasing.weights = debiasing.weights,
                              compliance.score = compliance.score,
                              num.trees.for.weights = num.trees.for.weights,
                              vcov.type = vcov.type,
                              target.sample = target.sample)
}

# =============================================================================
# Analysis tools from grf (get_tree, split_frequencies, grf_variable_importance,
# get_forest_weights, get_leaf_node)
# =============================================================================

#' Get a single tree from a grf forest
#'
#' Re-export of \code{\link[grf]{get_tree}}. Retrieves a single tree from a
#' trained grf forest for inspection or plotting. Requires the \pkg{grf} package.
#'
#' @param forest A trained forest from \pkg{grf}.
#' @param index The 1-based index of the tree to retrieve.
#' @return A \code{grf_tree} object with \code{nodes}, \code{columns}, etc.
#'
#' @seealso \code{\link[grf]{get_tree}}, \code{\link{plot.grf_tree}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # get_tree(...)
#' }
#' @export
get_tree <- function(forest, index) {
  .check_grf()
  grf::get_tree(forest, index)
}

#' Split frequencies of a grf forest
#'
#' Re-export of \code{\link[grf]{split_frequencies}}. For each depth, counts how
#' often each feature was split on. Requires the \pkg{grf} package.
#'
#' @param forest A trained forest from \pkg{grf}.
#' @param max.depth Maximum depth to consider. Default 4.
#' @return A matrix (depth x feature) of split counts.
#'
#' @seealso \code{\link[grf]{split_frequencies}}, \code{\link{grf_variable_importance}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # split_frequencies(...)
#' }
#' @export
split_frequencies <- function(forest, max.depth = 4) {
  .check_grf()
  grf::split_frequencies(forest, max.depth)
}

#' Variable importance for a grf forest
#'
#' Re-export of \code{\link[grf]{variable_importance}} from the \pkg{grf} package.
#' Simple importance measure from weighted split frequencies by depth. Use
#' \code{\link{variable_importance}} for RCausalML's built-in causal forest.
#' Requires the \pkg{grf} package.
#'
#' @param forest A trained forest from \pkg{grf}.
#' @param decay.exponent Weight given to depth (default 2).
#' @param max.depth Maximum depth to consider. Default 4.
#' @return A vector of importance values per feature.
#'
#' @seealso \code{\link[grf]{variable_importance}}, \code{\link{split_frequencies}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # grf_variable_importance(...)
#' }
#' @export
grf_variable_importance <- function(forest, decay.exponent = 2, max.depth = 4) {
  .check_grf()
  grf::variable_importance(forest, decay.exponent, max.depth)
}

#' Forest kernel weights (grf)
#'
#' Re-export of \code{\link[grf]{get_forest_weights}}. Computes the kernel weights
#' (alpha in the GRF paper) for each test point. Requires the \pkg{grf} package.
#'
#' @param forest A trained forest from \pkg{grf}.
#' @param newdata Points at which to compute weights. If NULL, OOB weights on
#'   the training set.
#' @param num.threads Number of threads. Default NULL.
#' @return A sparse matrix: rows = test samples, columns = training samples.
#'
#' @seealso \code{\link[grf]{get_forest_weights}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # get_forest_weights(...)
#' }
#' @export
get_forest_weights <- function(forest, newdata = NULL, num.threads = NULL) {
  .check_grf()
  grf::get_forest_weights(forest, newdata, num.threads)
}

#' Leaf node for each sample in a grf tree
#'
#' Re-export of \code{\link[grf]{get_leaf_node}}. Given a \code{grf_tree}, returns
#' the leaf index each test sample falls into. Requires the \pkg{grf} package.
#'
#' @param tree A \code{grf_tree} from \code{\link{get_tree}}.
#' @param newdata Matrix of test points.
#' @param node.id If TRUE (default), return leaf node id per sample; if FALSE,
#'   return list of sample indices per node.
#' @return Vector of leaf indices, or list of sample indices per node.
#'
#' @seealso \code{\link[grf]{get_leaf_node}}, \code{\link{get_tree}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # get_leaf_node(...)
#' }
#' @export
get_leaf_node <- function(tree, newdata, node.id = TRUE) {
  .check_grf()
  grf::get_leaf_node(tree, newdata, node.id)
}
