#' Causal Tree (Honest or Adaptive)
#'
#' Fits a causal tree for heterogeneous treatment effects using recursive partitioning.
#' Supports both **adaptive** (original `causalTree` style) and **honest** estimation
#' (recommended; Athey & Imbens 2016) in one function.
#'
#' @param formula Formula: response ~ predictors (no interactions)
#' @param data Data frame containing the variables in `formula`
#' @param treatment Binary vector (0 = control, 1 = treated), same length as rows in `data`
#' @param honest Logical. If `TRUE` (default), uses honest estimation:
#'   tree structure built on training sample, leaf means estimated on held-out estimation sample.
#' @param est_fraction Fraction of data used for honest estimation (when `honest = TRUE`).
#'   Ignored if `est_idx` is provided. Default: 0.5
#' @param est_idx Optional integer vector: row indices to use for honest estimation.
#'   If provided, overrides random split and `est_fraction`.
#' @param split.Rule Splitting rule: `"TOT"`, `"CT"`, `"fit"`, `"tstats"` (or with `"D"` suffix for bucketed)
#' @param cv.option Cross-validation method: `"none"`, `"TOT"`, `"matching"`, `"CT"`, `"fit"`, `"user"`, `"policy"`
#' @param minsize Minimum number of treated **and** control units per leaf. Default: 5L
#' @param propensity Scalar propensity score (constant). Defaults to observed mean(treatment)
#' @param split.alpha Parameter for honest splitting risk function (01). Default: 0.5
#' @param cv.alpha Parameter for CV risk function (01). Default: 0.5
#' @param split.Bucket Logical. Use bucketed (discrete) splitting?
#' @param bucketNum,bucketMax Bucket parameters (when `split.Bucket = TRUE`)
#' @param xval Number of cross-validation folds. Default: 5
#' @param cp Complexity parameter (can be 0 for no pruning during growth)
#' @param ... Passed to `rpart.control()` (e.g. `minsplit`, `maxdepth`, `xval`)
#'
#' @return Object of class `c("causalTree", "rpart")`
#'   - Use `rpart.plot::rpart.plot()` to visualize
#'   - Use `prune()` to prune via cross-validation error
#'   - Use `predict.causalTree()` (from previous response) for predictions
#'
#' @references
#' Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects.
#' Proceedings of the National Academy of Sciences.
#'
#' Original implementation: <https://github.com/susanathey/causalTree>
#' Maintained fork: **htetree** on CRAN
#'
#' @examples
#' \dontrun{
#'   # Honest causal tree (recommended)
#'   ct_honest <- causal_tree(
#'     y ~ x1 + x2 + x3 + x4,
#'     data = mydata,
#'     treatment = mydata$trt,
#'     honest = TRUE,
#'     est_fraction = 0.4,
#'     split.Rule = "CT",
#'     cv.option = "CT",
#'     minsize = 10L,
#'     xval = 5
#'   )
#'
#'   # Prune using CV error
#'   cp_best <- ct_honest$cptable[which.min(ct_honest$cptable[,"xerror"]), "CP"]
#'   pruned <- prune(ct_honest, cp = cp_best)
#'
#'   rpart.plot::rpart.plot(pruned, roundint = FALSE)
#'
#'   # Predict CATE on new data
#'   tau_new <- predict(pruned, newdata = testdata, type = "treatment.effect")
#' }
#'
#' @seealso
#' \code{\link{RCausalML-package}}
#' @export
causal_tree <- function(
    formula,
    data,
    treatment,
    honest          = TRUE,
    est_fraction    = 0.5,
    est_idx         = NULL,
    split.Rule      = "CT",
    cv.option       = "CT",
    minsize         = 5L,
    propensity      = NULL,
    split.alpha     = 0.5,
    cv.alpha        = 0.5,
    split.Bucket    = FALSE,
    bucketNum       = 5L,
    bucketMax       = 100L,
    xval            = 5L,
    cp              = 0,
    ...) {

  Call <- match.call()

  #  Input validation 
  if (!inherits(data, "data.frame")) {
    stop("'data' must be a data.frame")
  }
  mf <- model.frame(formula, data)
  n <- nrow(mf)

  if (missing(treatment)) stop("'treatment' is required (0/1 vector)")
  treatment <- as.integer(treatment)
  if (length(treatment) != n || !all(treatment %in% 0:1)) {
    stop("'treatment' must be a 0/1 vector of length nrow(data)")
  }

  if (is.null(propensity)) propensity <- mean(treatment)

  #  Handle honest estimation split 
  train_idx <- seq_len(n)
  est_sample_size <- NULL

  if (honest) {
    if (!is.null(est_idx)) {
      est_idx <- as.integer(est_idx)
      if (!all(est_idx %in% train_idx)) stop("Invalid est_idx")
      train_idx <- setdiff(train_idx, est_idx)
    } else {
      est_fraction <- max(0.05, min(0.95, est_fraction))
      est_n <- round(n * est_fraction)
      est_idx <- sample(seq_len(n), est_n)
      train_idx <- setdiff(seq_len(n), est_idx)
    }
    est_sample_size <- length(est_idx)
    message(sprintf("Honest mode: %d obs for tree structure, %d for estimation",
                    length(train_idx), est_sample_size))
  } else {
    message("Adaptive mode (honest = FALSE)  leaf estimates from training data only")
  }

  # Subset data for tree-building phase
  data_train <- data[train_idx, , drop = FALSE]
  treatment_train <- treatment[train_idx]

  #  Prepare arguments for causalTree backend 
  args <- list(
    formula       = formula,
    data          = data_train,
    weights       = NULL,           # can be extended later
    treatment     = treatment_train,
    subset        = NULL,
    na.action     = htetree::na.causalTree,
    split.Rule    = split.Rule,
    split.Honest  = if (honest) TRUE else FALSE,
    HonestSampleSize = if (honest) est_sample_size else nrow(data_train),
    split.Bucket  = split.Bucket,
    bucketNum     = as.integer(bucketNum),
    bucketMax     = as.integer(bucketMax),
    cv.option     = cv.option,
    cv.Honest     = honest,
    minsize       = as.integer(minsize),
    x             = FALSE,
    y             = TRUE,
    propensity    = propensity,
    control       = rpart::rpart.control(cp = cp, xval = xval, ...),
    split.alpha   = split.alpha,
    cv.alpha      = cv.alpha,
    ...
  )

  #  Fit tree structure (on training sample) 
  tree <- do.call(htetree::causalTree, args)   # or causalTree::causalTree

  #  Honest leaf estimation (if requested) 
  if (honest && length(est_idx) > 0) {
    # Use honest estimation on held-out sample
    est_data <- data[est_idx, , drop = FALSE]
    est_treatment <- treatment[est_idx]

    # htetree::honest.est.causalTree(fit, x, wt, treatment, y) expects
    # fit = tree, x = covariate matrix, wt = weights, treatment = vector, y = outcome
    mf_est <- model.frame(formula, est_data)
    x_est <- model.matrix(formula, mf_est)
    if (ncol(x_est) > 1L && colnames(x_est)[1L] == "(Intercept)")
      x_est <- x_est[, -1L, drop = FALSE]
    y_est <- model.response(mf_est)
    wt_est <- rep(1, nrow(est_data))

    tree_honest <- htetree::honest.est.causalTree(
      fit = tree,
      x = x_est,
      wt = wt_est,
      treatment = est_treatment,
      y = y_est
    )

    # Replace the original with honest version
    tree <- tree_honest
  }

  #  Finalize object 
  class(tree) <- c("causalTree", class(tree))
  tree$call <- Call
  tree$honest <- honest
  if (honest) {
    tree$train_idx <- train_idx
    tree$est_idx   <- est_idx
  }

  tree
}
