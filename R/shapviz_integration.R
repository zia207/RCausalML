# =============================================================================
# SHAP / shapviz integration for RCausalML
# =============================================================================
# Use kernelshap to compute SHAP values for any RCausalML model with predict(),
# then pass the result to shapviz() to get all shapviz plots (importance,
# dependence, waterfall, force, etc.). Works with meta-learners, DML, causal
# forest, uplift trees, policy learners, neural CATE models, and interpreters.
# =============================================================================

#' Prediction helper for RCausalML models (for use with kernelshap)
#'
#' Extracts a numeric vector of predictions from \code{predict(object, newdata)}.
#' Handles models that return a list with a \code{predictions} component
#' (e.g. \code{causal_forest}) or a plain vector/matrix.
#'
#' @param object Fitted RCausalML model (e.g. SLearner, causal_forest, ...).
#' @param newdata Feature matrix or data.frame passed to \code{predict(object, newdata)}.
#' @return Numeric vector of length \code{nrow(newdata)}.
#' @noRd
rcausalml_predict_numeric <- function(object, newdata) {
  p <- predict(object, newdata = newdata)
  if (is.list(p) && "predictions" %in% names(p)) {
    p <- as.numeric(p[["predictions"]])
  } else if (is.list(p) && "pred" %in% names(p)) {
    p <- as.numeric(p[["pred"]])
  } else {
    p <- as.numeric(p)
  }
  # kernelshap/shapviz require a finite prediction vector; replace any NA/Inf/NaN
  # with the mean of the finite values (or 0 if all are non-finite)
  bad <- !is.finite(p)
  if (any(bad)) {
    fallback <- if (any(!bad)) mean(p[!bad]) else 0
    p[bad] <- fallback
  }
  p
}

#' Compute SHAP values for any RCausalML CATE model
#'
#' Uses \code{kernelshap} (or \code{permshap}) to compute SHAP values for the
#' model's CATE predictions, so you can then pass the result to \code{shapviz()}
#' and use all shapviz functions: \code{sv_importance()}, \code{sv_dependence()},
#' \code{sv_waterfall()}, \code{sv_force()}, \code{sv_dependence2D()}, etc.
#'
#' Works with all RCausalML models that have a \code{predict(object, newdata)}
#' returning CATE (or a list with \code{predictions}), including:
#' \itemize{
#'   \item Meta-learners: \code{SLearner}, \code{TLearner}, \code{XLearner}, \code{RLearner}, \code{DRLearner}
#'   \item DML: \code{LinearDML}, \code{SparseLinearDML}, \code{KernelDML}, \code{NonParamDML}, \code{CausalForestDML}, \code{DynamicDMLearner}, \code{OrthoIVLearner}, \code{DMLIVLearner}, \code{NonParamDMLIVLearner}
#'   \item Trees/forests: \code{causal_forest}, \code{uplift_rf_*}, \code{uplift_tree_*}, \code{interaction_tree}, \code{causal_inference_tree}
#'   \item Policy: \code{policy_learner}, \code{DRPolicyTree}, \code{DRPolicyForest}
#'   \item Interpreters: \code{SingleTreeCateInterpreter}, \code{SingleTreePolicyInterpreter}
#'   \item Neural: \code{cevae}, \code{dragonnet}, \code{tarnet}, \code{cfrnet}, \code{ganite}
#'   \item Other: \code{AutomatedMLModel}, \code{driv_learner}
#' }
#'
#' @param object Fitted RCausalML model (any of the above).
#' @param X Feature matrix or data.frame to explain (same covariates as used for fitting).
#' @param bg_X Optional background dataset for kernelshap; if \code{NULL}, a small sample of \code{X} may be used by kernelshap (see \code{?kernelshap::kernelshap}).
#' @param n_samples Optional number of rows to explain (for speed). If \code{NULL}, all rows of \code{X} are used.
#' @param use_permshap If \code{TRUE}, use \code{kernelshap::permshap()} instead of \code{kernelshap::kernelshap()} (faster, approximate). Recommended when \code{ncol(X) > 8}.
#' @param pred_fun Optional custom prediction function of the form
#'   \code{function(object, newdata)} returning a numeric vector. If \code{NULL},
#'   a default wrapper around \code{predict(object, newdata)} is used.
#' @param ... Passed to \code{kernelshap::kernelshap()} or \code{kernelshap::permshap()}
#'   (e.g. \code{n_permutations}, \code{verbose}).
#' @return An object of class \code{kernelshap}. Pass it to \code{shapviz()} to get a \code{shapviz} object, then use \code{sv_importance()}, \code{sv_dependence()}, \code{sv_waterfall()}, \code{sv_force()}, etc.
#' @seealso \code{\link[kernelshap]{kernelshap}}, \code{\link[kernelshap]{permshap}}; \code{shapviz::shapviz()}, \code{shapviz::sv_importance()}, \code{shapviz::sv_dependence()}.
#' @examples
#' \dontrun{
#' library(RCausalML)
#' library(kernelshap)
#' library(shapviz)
#'
#' # Fit any CATE model
#' set.seed(42)
#' d <- synthetic_data(mode = 1, n = 500, p = 5, sigma = 1)
#' sl <- SLearner(learner = "ranger")
#' sl <- fit(sl, d$X, d$w, d$y)
#'
#' # Compute SHAP for CATE predictions
#' X_explain <- d$X[1:100, ]   # optional: subset for speed
#' ks <- explain_cate(sl, X_explain, bg_X = d$X, use_permshap = TRUE)
#'
#' # Build shapviz object and use all shapviz functions
#' shp <- shapviz(ks)
#' sv_importance(shp)
#' sv_dependence(shp, "X1")
#' sv_waterfall(shp, 1)
#' sv_force(shp, 1)
#' }
#' @export
explain_cate <- function(object,
                         X,
                         bg_X = NULL,
                         n_samples = NULL,
                         use_permshap = FALSE,
                         pred_fun = NULL,
                         ...) {
  if (!requireNamespace("kernelshap", quietly = TRUE)) {
    stop("Package 'kernelshap' is required for explain_cate(). Install with: install.packages(\"kernelshap\")")
  }
  X <- as.data.frame(X)
  if (!is.null(n_samples) && nrow(X) > n_samples) {
    X <- X[seq_len(n_samples), , drop = FALSE]
  }
  if (!is.null(bg_X)) bg_X <- as.data.frame(bg_X)
  if (is.null(pred_fun)) {
    pred_fun <- function(m, x) rcausalml_predict_numeric(m, as.data.frame(x))
  }
  if (use_permshap) {
    ks <- kernelshap::permshap(object, X = X, pred_fun = pred_fun, bg_X = bg_X, ...)
  } else {
    ks <- kernelshap::kernelshap(object, X = X, pred_fun = pred_fun, bg_X = bg_X, ...)
  }
  ks
}
