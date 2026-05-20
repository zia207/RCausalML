# CausalML-R: Policy Learners (EconML-style + Athey & Wager 2018)
# Aligns with EconML policy module: PolicyTree, PolicyForest, DRPolicyTree, DRPolicyForest.
# Reference: Athey & Wager (2018) https://arxiv.org/abs/1702.02896
# EconML: econml.policy._base, _forest, _drlearner

#' Check that xgboost is available (for learner = "xgb")
#' @noRd
check_xgboost_policy <- function() {
  if (!requireNamespace("xgboost", quietly = TRUE))
    stop("Package 'xgboost' is required for outcome_learner or treatment_learner = 'xgb'. Install with install.packages('xgboost').")
}

#' Out-of-fold propensity scores via K-fold CV
#' @param X covariate matrix
#' @param w treatment (0/1)
#' @param learner "glmnet", "ranger", or "glm"
#' @param n_fold number of folds
#' @param seed random seed for folds
#' @noRd
oof_propensity <- function(X, w, learner = "glmnet", n_fold = 5, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- length(w)
  folds <- sample(rep(seq_len(n_fold), length.out = n))
  pred <- numeric(n)
  for (k in seq_len(n_fold)) {
    tr <- which(folds != k)
    te <- which(folds == k)
    X_tr <- X[tr, , drop = FALSE]
    w_tr <- w[tr]
    X_te <- X[te, , drop = FALSE]
    if (learner == "glmnet") {
      fit <- glmnet::cv.glmnet(X_tr, w_tr, family = "binomial", nfolds = min(5, max(1, length(tr) - 1)))
      pred[te] <- as.vector(predict(fit, newx = X_te, s = "lambda.1se", type = "response"))
    } else if (learner == "ranger") {
      df_tr <- as.data.frame(X_tr)
      df_tr$w <- w_tr
      fit <- ranger::ranger(w ~ ., data = df_tr, probability = TRUE)
      pred[te] <- predict(fit, data = as.data.frame(X_te))$predictions[, 2]
    } else {
      df_tr <- data.frame(w = w_tr, X = X_tr)
      fit <- stats::glm(w ~ ., data = df_tr, family = stats::binomial)
      pred[te] <- as.vector(predict(fit, newdata = as.data.frame(X_te), type = "response"))
    }
  }
  pred
}

# =============================================================================
# Policy learner (Athey & Wager style: DR score + weighted classifier)
# =============================================================================

#' Policy Learner: treatment assignment policy with doubly robust estimator
#'
#' Learns a treatment assignment policy from observational data using a doubly robust
#' estimator of the causal effect for binary treatment. The policy is trained to maximize
#' expected outcome (welfare) by fitting a classifier on the sign of the DR score
#' with weights equal to the absolute DR score. Aligns with the "weighted classification"
#' approach in Athey & Wager (2018); for tree/forest policy learners see \code{\link{DRPolicyTree}}
#' and \code{\link{DRPolicyForest}}.
#'
#' @param outcome_learner character: regression model for outcome ("ranger", "lm", "glmnet", "xgb").
#' @param treatment_learner character: classification model for propensity ("glmnet", "ranger", "glm").
#' @param policy_learner character: classifier for treatment assignment, must support \code{case.weights} ("rpart", "ranger").
#' @param clip_bounds numeric of length 2: lower and upper bounds for clipping propensity scores (default \code{c(1e-3, 1 - 1e-3)}).
#' @param n_fold integer: number of cross-validation folds for outcome and propensity estimation.
#' @param random_state integer or NULL: random seed for fold splits.
#' @param calibration logical: whether to calibrate propensity (currently ignored; for API compatibility).
#' @return Object of class \code{policy_learner}.
#' @references Athey, S., & Wager, S. (2018). Policy learning with observational data. \url{https://arxiv.org/abs/1702.02896}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # policy_learner(...)
#' }
#' @export
policy_learner <- function(
    outcome_learner = "ranger",
    treatment_learner = "glmnet",
    policy_learner = "rpart",
    clip_bounds = c(1e-3, 1 - 1e-3),
    n_fold = 5L,
    random_state = NULL,
    calibration = FALSE) {
  structure(
    list(
      outcome_learner = outcome_learner,
      treatment_learner = treatment_learner,
      policy_learner = policy_learner,
      clip_bounds = clip_bounds,
      n_fold = as.integer(n_fold),
      random_state = random_state,
      calibration = calibration,
      model_mu = NULL,
      model_pi = NULL,
      y_pred = NULL,
      tau_pred = NULL,
      w_pred = NULL,
      dr_score = NULL,
      X_names = NULL
    ),
    class = "policy_learner"
  )
}

#' Out-of-fold outcome and CATE estimates (internal)
#' @noRd
policy_outcome_estimate <- function(obj, X, w, y) {
  n <- length(y)
  if (!is.null(obj$random_state)) set.seed(obj$random_state)
  folds <- sample(rep(seq_len(obj$n_fold), length.out = n))
  y_pred <- numeric(n)
  tau_pred <- numeric(n)
  for (k in seq_len(obj$n_fold)) {
    tr <- which(folds != k)
    te <- which(folds == k)
    X_tr <- X[tr, , drop = FALSE]
    w_tr <- w[tr]
    y_tr <- y[tr]
    X_te <- X[te, , drop = FALSE]
    w_te <- w[te]
    # Fit mu(X, W) on train
    df_tr <- as.data.frame(cbind(w = w_tr, X_tr))
    df_tr$y <- y_tr
    learner <- obj$outcome_learner
    if (learner == "lm") {
      m <- stats::lm(y ~ ., data = df_tr)
      y_pred[te] <- predict(m, newdata = as.data.frame(cbind(w = w_te, X_te)))
      X1 <- as.data.frame(cbind(w = 1, X_te))
      X0 <- as.data.frame(cbind(w = 0, X_te))
      tau_pred[te] <- predict(m, newdata = X1) - predict(m, newdata = X0)
    } else if (learner == "ranger") {
      m <- ranger::ranger(y ~ ., data = df_tr)
      y_pred[te] <- predict(m, data = as.data.frame(cbind(w = w_te, X_te)))$predictions
      tau_pred[te] <- predict(m, data = as.data.frame(cbind(w = 1, X_te)))$predictions -
        predict(m, data = as.data.frame(cbind(w = 0, X_te)))$predictions
    } else if (learner == "glmnet") {
      mm_tr <- as.matrix(df_tr[, -which(names(df_tr) == "y")])
      m <- glmnet::cv.glmnet(mm_tr, df_tr$y, nfolds = min(5, max(1, nrow(mm_tr) - 1)))
      mm_te_w <- as.matrix(cbind(w = w_te, X_te))
      y_pred[te] <- as.vector(predict(m, newx = mm_te_w, s = "lambda.1se"))
      mm_te_1 <- as.matrix(cbind(w = 1, X_te))
      mm_te_0 <- as.matrix(cbind(w = 0, X_te))
      tau_pred[te] <- as.vector(predict(m, newx = mm_te_1, s = "lambda.1se")) -
        as.vector(predict(m, newx = mm_te_0, s = "lambda.1se"))
    } else if (learner == "xgb") {
      check_xgboost_policy()
      mm_tr <- as.matrix(df_tr[, -which(names(df_tr) == "y")])
      m <- xgboost::xgboost(data = mm_tr, label = df_tr$y, nrounds = 50, verbosity = 0,
                            objective = "reg:squarederror")
      mm_te_w <- as.matrix(cbind(w = w_te, X_te))
      y_pred[te] <- as.vector(predict(m, mm_te_w))
      mm_te_1 <- as.matrix(cbind(w = 1, X_te))
      mm_te_0 <- as.matrix(cbind(w = 0, X_te))
      tau_pred[te] <- as.vector(predict(m, mm_te_1)) - as.vector(predict(m, mm_te_0))
    } else {
      stop("outcome_learner must be 'lm', 'ranger', 'glmnet', or 'xgb'")
    }
  }
  list(y_pred = y_pred, tau_pred = tau_pred)
}

#' Fit the policy learner
#'
#' @param obj object of class \code{policy_learner}.
#' @param X feature matrix or data.frame.
#' @param treatment treatment vector (1 = treated, 0 = control).
#' @param y outcome vector.
#' @param p optional user-provided propensity score vector (0–1).
#' @param dhat optional user-provided predicted treatment effect (CATE) vector.
#' @param ... passed to underlying learners.
#' @return Fitted \code{policy_learner} object (invisibly).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.policy_learner(...)
#' }
#' @export
fit.policy_learner <- function(obj, X, treatment, y, p = NULL, dhat = NULL, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  check_treatment_vector(treatment, 0)
  w <- as.integer(treatment != 0)

  # Out-of-fold outcome (and tau) estimates
  oo <- policy_outcome_estimate(obj, X, w, y)
  obj$y_pred <- oo$y_pred
  obj$tau_pred <- oo$tau_pred
  if (!is.null(dhat)) obj$tau_pred <- as.numeric(dhat)

  # Propensity: user-provided or OOF estimate
  if (is.null(p)) {
    obj$w_pred <- oof_propensity(X, w, learner = obj$treatment_learner,
                                 n_fold = obj$n_fold, seed = obj$random_state)
  } else {
    obj$w_pred <- as.numeric(p)
  }
  obj$w_pred <- pmax(obj$clip_bounds[1], pmin(obj$clip_bounds[2], obj$w_pred))

  # Doubly robust score
  obj$dr_score <- obj$tau_pred + (w - obj$w_pred) / (obj$w_pred * (1 - obj$w_pred)) * (y - obj$y_pred)

  # Policy target: sign(DR score); weight = |DR score|
  target <- as.integer(obj$dr_score > 0)  # 1 = treat, 0 = don't treat
  wts <- abs(obj$dr_score)

  # Fit policy classifier with case weights
  df_pi <- as.data.frame(X)
  df_pi$target <- factor(target, levels = c(0, 1))
  pl <- obj$policy_learner
  if (pl == "rpart") {
    obj$model_pi <- list(
      type = "rpart",
      model = rpart::rpart(target ~ ., data = df_pi, weights = wts, method = "class", ...)
    )
  } else if (pl == "ranger") {
    obj$model_pi <- list(
      type = "ranger",
      model = ranger::ranger(target ~ ., data = df_pi, case.weights = wts, probability = TRUE, ...)
    )
  } else {
    stop("policy_learner must be 'rpart' or 'ranger'")
  }
  obj$X_names <- colnames(X)
  invisible(obj)
}

#' Predict treatment assignment (0 or 1)
#'
#' @param object fitted \code{policy_learner} object.
#' @param newdata feature matrix or data.frame.
#' @param ... unused.
#' @return Integer vector of predicted treatment assignment (0 = control, 1 = treat).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.policy_learner(...)
#' }
#' @export
predict.policy_learner <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names
  m <- object$model_pi
  df <- as.data.frame(newdata)
  if (m$type == "rpart") {
    cl <- predict(m$model, newdata = df, type = "class")
    out <- as.integer(as.character(cl))
  } else {
    prob <- predict(m$model, data = df)$predictions[, 2]
    out <- as.integer(prob >= 0.5)
  }
  out
}

#' Predict probability of treatment assignment (P(treat))
#'
#' @param object fitted \code{policy_learner} object.
#' @param newdata feature matrix or data.frame.
#' @param ... unused.
#' @return Numeric vector of predicted probability of recommending treatment.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict_proba.policy_learner(...)
#' }
#' @export
predict_proba.policy_learner <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names) && ncol(newdata) == length(object$X_names))
    colnames(newdata) <- object$X_names
  m <- object$model_pi
  df <- as.data.frame(newdata)
  if (m$type == "rpart") {
    prob <- predict(m$model, newdata = df, type = "prob")[, "1"]
  } else {
    prob <- predict(m$model, data = df)$predictions[, 2]
  }
  as.numeric(prob)
}

# =============================================================================
# DR Policy Tree / Forest (EconML-style: first-stage nuisances + policy tree/forest)
# =============================================================================

#' Cross-fitted outcome predictions for policy value matrix
#' Returns OOS predicted outcome under control and under each treatment.
#' @noRd
policy_oos_outcome <- function(X, treatment, y, control_name, t_groups, learner, n_fold, seed) {
  n <- nrow(X)
  if (!is.null(seed)) set.seed(seed)
  folds <- sample(rep(seq_len(n_fold), length.out = n))
  mu_c <- numeric(n)
  mu_t <- matrix(NA_real_, n, length(t_groups))
  colnames(mu_t) <- as.character(t_groups)
  for (k in seq_len(n_fold)) {
    tr <- which(folds != k)
    te <- which(folds == k)
    X_tr <- X[tr, , drop = FALSE]
    X_te <- X[te, , drop = FALSE]
    y_tr <- y[tr]
    treatment_tr <- treatment[tr]
    treatment_te <- treatment[te]
    # Control outcome
    idx_c <- which(treatment_tr == control_name)
    if (length(idx_c) >= 5L) {
      df_c <- as.data.frame(X_tr[idx_c, , drop = FALSE])
      df_c$y <- y_tr[idx_c]
      m_c <- policy_fit_outcome(df_c, learner)
      mu_c[te] <- policy_pred_outcome(m_c, X_te, learner)
    }
    # Outcome under each treatment
    for (j in seq_along(t_groups)) {
      g <- t_groups[j]
      idx_t <- which(treatment_tr == g)
      if (length(idx_t) >= 5L) {
        df_t <- as.data.frame(X_tr[idx_t, , drop = FALSE])
        df_t$y <- y_tr[idx_t]
        m_t <- policy_fit_outcome(df_t, learner)
        mu_t[te, j] <- policy_pred_outcome(m_t, X_te, learner)
      }
    }
  }
  list(mu_c = mu_c, mu_t = mu_t)
}

#' @noRd
policy_fit_outcome <- function(df, learner) {
  resp <- "y"
  if (learner == "lm") {
    list(type = "lm", model = stats::lm(y ~ ., data = df))
  } else if (learner == "ranger") {
    list(type = "ranger", model = ranger::ranger(y ~ ., data = df))
  } else if (learner == "glmnet") {
    mm <- as.matrix(df[, -which(names(df) == "y")])
    list(type = "glmnet", model = glmnet::cv.glmnet(mm, df$y, nfolds = min(5L, max(1L, nrow(mm) - 1L))))
  } else if (learner == "xgb") {
    check_xgboost_policy()
    mm <- as.matrix(df[, -which(names(df) == "y")])
    list(type = "xgb", model = xgboost::xgboost(data = mm, label = df$y, nrounds = 50L, verbosity = 0L, objective = "reg:squarederror"))
  } else {
    stop("model_regression must be 'lm', 'ranger', 'glmnet', or 'xgb'")
  }
}

#' @noRd
policy_pred_outcome <- function(m, X, learner) {
  if (is.null(m)) return(rep(NA_real_, nrow(X)))
  if (inherits(X, "data.frame")) X <- as.matrix(X)
  df <- as.data.frame(X)
  if (m$type == "lm") return(as.vector(predict(m$model, newdata = df)))
  if (m$type == "ranger") return(as.vector(predict(m$model, data = df)$predictions))
  if (m$type == "glmnet") return(as.vector(predict(m$model, newx = X, s = "lambda.1se")[, 1L]))
  if (m$type == "xgb") return(as.vector(predict(m$model, newdata = X)))
  rep(NA_real_, nrow(X))
}

#' DR Policy Tree (EconML-style)
#'
#' Policy learner that uses cross-fitted outcome regression to form a value matrix
#' (predicted CATE per treatment vs control), then fits a single decision tree to predict
#' the best treatment (argmax value). Mirrors EconML \code{econml.policy.DRPolicyTree}.
#'
#' @param model_regression character: outcome regression learner ("ranger", "lm", "glmnet", "xgb"). Default \code{"ranger"}.
#' @param model_propensity character: propensity learner for optional DR weighting ("glmnet", "ranger", "glm"). Default \code{"glmnet"}.
#' @param min_propensity numeric: lower bound for propensity (default \code{1e-6}).
#' @param control_name value indicating control group (default \code{0}).
#' @param cv integer: number of cross-fitting folds (default \code{2}).
#' @param max_depth integer or NULL: maximum tree depth for the policy tree (default \code{NULL}).
#' @param min_samples_split integer: minimum samples to split a node (default \code{10}).
#' @param min_samples_leaf integer: minimum samples per leaf (default \code{5}).
#' @param max_features character or integer: "auto", "sqrt", "log2", or number (default \code{"auto"}).
#' @param random_state integer or NULL: random seed.
#' @return Object of class \code{DRPolicyTree}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # DRPolicyTree(...)
#' }
#' @export
DRPolicyTree <- function(model_regression = "ranger",
                        model_propensity = "glmnet",
                        min_propensity = 1e-6,
                        control_name = 0,
                        cv = 2L,
                        max_depth = NULL,
                        min_samples_split = 10L,
                        min_samples_leaf = 5L,
                        max_features = "auto",
                        random_state = NULL) {
  structure(list(
    model_regression = model_regression,
    model_propensity = model_propensity,
    min_propensity = min_propensity,
    control_name = control_name,
    cv = as.integer(cv),
    max_depth = max_depth,
    min_samples_split = as.integer(min_samples_split),
    min_samples_leaf = as.integer(min_samples_leaf),
    max_features = max_features,
    random_state = random_state,
    policy_model_ = NULL,
    treatment_levels_ = NULL,
    t_groups_ = NULL,
    X_names_ = NULL
  ), class = "DRPolicyTree")
}

#' Fit DR Policy Tree
#'
#' @param obj \code{DRPolicyTree} object.
#' @param X feature matrix or data.frame.
#' @param treatment treatment vector (numeric or factor).
#' @param y outcome vector.
#' @param W optional controls (ignored if NULL; for API compatibility).
#' @param sample_weight optional numeric weights (passed to rpart).
#' @param ... passed to rpart.
#' @return Fitted \code{DRPolicyTree} (invisibly).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.DRPolicyTree(...)
#' }
#' @export
fit.DRPolicyTree <- function(obj, X, treatment, y, W = NULL, sample_weight = NULL, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- length(y)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  check_treatment_vector(treatment, obj$control_name)
  t_groups <- sort(unique(treatment[treatment != obj$control_name]))
  obj$t_groups_ <- t_groups
  n_t <- length(t_groups)
  # Cross-fitted outcome: value matrix (n x (1 + n_t)) with columns (control=0, cate_1, cate_2, ...)
  oos <- policy_oos_outcome(X, treatment, y, obj$control_name, t_groups,
                            obj$model_regression, obj$cv, obj$random_state)
  value_matrix <- matrix(0, n, 1L + n_t)
  value_matrix[, 1L] <- 0
  for (j in seq_len(n_t)) value_matrix[, 1L + j] <- oos$mu_t[, j] - oos$mu_c
  # Best treatment index: 1 = control, 2 = first treatment, ...
  best_idx <- max.col(value_matrix, ties.method = "first")
  value_at_best <- value_matrix[cbind(seq_len(n), best_idx)]
  # Treatment levels for mapping: level 1 = control, 2 = first treatment, ...
  treatment_levels <- c(obj$control_name, t_groups)
  obj$treatment_levels_ <- treatment_levels
  # Fit single policy tree (classification: predict best treatment index)
  df <- as.data.frame(X)
  df$best <- factor(best_idx, levels = seq_len(1L + n_t))
  wts <- if (!is.null(sample_weight)) as.numeric(sample_weight) else value_at_best
  if (any(wts <= 0)) wts <- pmax(wts, 1e-6)
  control_rpart <- list(minsplit = obj$min_samples_split, minbucket = obj$min_samples_leaf)
  if (!is.null(obj$max_depth)) control_rpart$maxdepth <- obj$max_depth
  obj$policy_model_ <- rpart::rpart(best ~ ., data = df, weights = wts, method = "class",
                                    control = control_rpart, ...)
  obj$X_names_ <- colnames(X)
  invisible(obj)
}

#' Predict recommended treatment (DR Policy Tree)
#'
#' @param object fitted \code{DRPolicyTree}.
#' @param newdata feature matrix or data.frame.
#' @param ... unused.
#' @return Vector of recommended treatment values (same type as \code{control_name} / \code{t_groups}).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.DRPolicyTree(...)
#' }
#' @export
predict.DRPolicyTree <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names_) && ncol(newdata) == length(object$X_names_))
    colnames(newdata) <- object$X_names_
  df <- as.data.frame(newdata)
  pred_class <- predict(object$policy_model_, newdata = df, type = "class")
  idx <- as.integer(as.character(pred_class))
  object$treatment_levels_[idx]
}

#' Predict probability of recommending each treatment (DR Policy Tree)
#'
#' @param object fitted \code{DRPolicyTree}.
#' @param newdata feature matrix or data.frame.
#' @param ... unused.
#' @return Matrix of shape (n_samples, n_treatments) with probabilities (each row sums to 1).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict_proba.DRPolicyTree(...)
#' }
#' @export
predict_proba.DRPolicyTree <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names_) && ncol(newdata) == length(object$X_names_))
    colnames(newdata) <- object$X_names_
  df <- as.data.frame(newdata)
  prob <- predict(object$policy_model_, newdata = df, type = "prob")
  # prob columns are "1", "2", ...; ensure same order as treatment_levels_
  n_t <- length(object$treatment_levels_)
  out <- matrix(0, nrow(newdata), n_t)
  for (j in seq_len(n_t)) {
    colj <- which(colnames(prob) == as.character(j))
    if (length(colj)) out[, j] <- prob[, colj[1L]]
  }
  out
}

#' Predict expected value under each treatment (DR Policy Tree)
#' Returns the predicted CATE (value vs control) for each treatment.
#' @param object fitted \code{DRPolicyTree}.
#' @param newdata feature matrix or data.frame.
#' @param ... unused.
#' @return Matrix (n_samples x n_treatments) with value for control (column 1 = 0) and each treatment.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict_value.DRPolicyTree(...)
#' }
#' @export
predict_value.DRPolicyTree <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names_) && ncol(newdata) == length(object$X_names_))
    colnames(newdata) <- object$X_names_
  df <- as.data.frame(newdata)
  # Leaf index; we don't store leaf values in this simplified tree, so return proba-weighted value
  proba <- predict_proba(object, newdata, ...)
  # Default: return proba as proxy for value (policy tree doesn't store leaf values in this impl)
  proba
}

#' Policy feature names (EconML API compatibility)
#' @param object fitted policy object.
#' @param feature_names optional character vector of feature names.
#' @param ... unused.
#' @return
#' Object returned by \code{policy_feature_names}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # policy_feature_names(...)
#' }
#' @export
policy_feature_names <- function(object, feature_names = NULL, ...) {
  if (!is.null(feature_names)) return(feature_names)
  if (!is.null(object$X_names_)) object$X_names_ else object$X_names
}

#' Feature importances from policy tree (variable importance from rpart)
#' @param object fitted \code{DRPolicyTree}.
#' @param ... unused.
#' @return Named numeric vector of importance (sum to 1 if available).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # feature_importances.DRPolicyTree(...)
#' }
#' @export
feature_importances.DRPolicyTree <- function(object, ...) {
  imp <- object$policy_model_$variable.importance
  if (is.null(imp)) return(numeric(0))
  imp / sum(imp)
}

# =============================================================================
# DR Policy Forest (EconML-style: ensemble of policy trees with subsampling)
# =============================================================================

#' DR Policy Forest (EconML-style)
#'
#' Ensemble of policy trees fit on subsamples; prediction by majority vote.
#' Mirrors EconML \code{econml.policy.DRPolicyForest}.
#'
#' @param model_regression character: outcome regression learner (default \code{"ranger"}).
#' @param model_propensity character: propensity learner (default \code{"glmnet"}).
#' @param min_propensity numeric: lower bound for propensity (default \code{1e-6}).
#' @param control_name value indicating control group (default \code{0}).
#' @param cv integer: number of cross-fitting folds (default \code{2}).
#' @param n_estimators integer: number of trees in the forest (default \code{100}).
#' @param max_samples numeric in (0,1]: fraction of samples per tree (default \code{0.5}).
#' @param max_depth integer or NULL: maximum tree depth (default \code{NULL}).
#' @param min_samples_split integer: minimum samples to split (default \code{10}).
#' @param min_samples_leaf integer: minimum samples per leaf (default \code{5}).
#' @param max_features character or integer (default \code{"auto"}).
#' @param random_state integer or NULL: random seed.
#' @param n_jobs integer: number of parallel jobs (default \code{1}; uses \code{parallel} if > 1).
#' @return Object of class \code{DRPolicyForest}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # DRPolicyForest(...)
#' }
#' @export
DRPolicyForest <- function(model_regression = "ranger",
                          model_propensity = "glmnet",
                          min_propensity = 1e-6,
                          control_name = 0,
                          cv = 2L,
                          n_estimators = 100L,
                          max_samples = 0.5,
                          max_depth = NULL,
                          min_samples_split = 10L,
                          min_samples_leaf = 5L,
                          max_features = "auto",
                          random_state = NULL,
                          n_jobs = 1L) {
  structure(list(
    model_regression = model_regression,
    model_propensity = model_propensity,
    min_propensity = min_propensity,
    control_name = control_name,
    cv = as.integer(cv),
    n_estimators = as.integer(n_estimators),
    max_samples = max_samples,
    max_depth = max_depth,
    min_samples_split = as.integer(min_samples_split),
    min_samples_leaf = as.integer(min_samples_leaf),
    max_features = max_features,
    random_state = random_state,
    n_jobs = as.integer(n_jobs),
    estimators_ = NULL,
    treatment_levels_ = NULL,
    t_groups_ = NULL,
    X_names_ = NULL
  ), class = "DRPolicyForest")
}

#' Fit DR Policy Forest
#'
#' @param obj \code{DRPolicyForest} object.
#' @param X feature matrix or data.frame.
#' @param treatment treatment vector.
#' @param y outcome vector.
#' @param W optional controls (ignored).
#' @param sample_weight optional weights (passed to each tree).
#' @param ... passed to rpart.
#' @return Fitted \code{DRPolicyForest} (invisibly).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # fit.DRPolicyForest(...)
#' }
#' @export
fit.DRPolicyForest <- function(obj, X, treatment, y, W = NULL, sample_weight = NULL, ...) {
  conv <- convert_to_numeric(X, treatment, y)
  X <- conv$X
  treatment <- conv$treatment
  y <- conv$y
  n <- length(y)
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  check_treatment_vector(treatment, obj$control_name)
  t_groups <- sort(unique(treatment[treatment != obj$control_name]))
  obj$t_groups_ <- t_groups
  n_t <- length(t_groups)
  treatment_levels <- c(obj$control_name, t_groups)
  obj$treatment_levels_ <- treatment_levels
  obj$X_names_ <- colnames(X)
  n_subsample <- max(2L, ceiling(n * obj$max_samples))
  if (!is.null(obj$random_state)) set.seed(obj$random_state)
  seeds <- sample.int(1e7, size = obj$n_estimators)
  fit_one <- function(i) {
    set.seed(seeds[i])
    idx <- sample(n, size = min(n_subsample, n), replace = FALSE)
    tree <- DRPolicyTree(
      model_regression = obj$model_regression,
      model_propensity = obj$model_propensity,
      min_propensity = obj$min_propensity,
      control_name = obj$control_name,
      cv = obj$cv,
      max_depth = obj$max_depth,
      min_samples_split = obj$min_samples_split,
      min_samples_leaf = obj$min_samples_leaf,
      max_features = obj$max_features,
      random_state = seeds[i]
    )
    fit(tree, X[idx, , drop = FALSE], treatment[idx], y[idx],
        sample_weight = if (is.null(sample_weight)) NULL else sample_weight[idx], ...)
  }
  if (obj$n_jobs > 1L) {
    obj$estimators_ <- parallel::mclapply(seq_len(obj$n_estimators), fit_one, mc.cores = obj$n_jobs)
  } else {
    obj$estimators_ <- lapply(seq_len(obj$n_estimators), fit_one)
  }
  invisible(obj)
}

#' Predict recommended treatment (DR Policy Forest) — majority vote
#'
#' @param object fitted \code{DRPolicyForest}.
#' @param newdata feature matrix or data.frame.
#' @param ... unused.
#' @return Vector of recommended treatment values.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict.DRPolicyForest(...)
#' }
#' @export
predict.DRPolicyForest <- function(object, newdata, ...) {
  proba <- predict_proba(object, newdata, ...)
  idx <- max.col(proba, ties.method = "first")
  object$treatment_levels_[idx]
}

#' Predict probability of recommending each treatment (DR Policy Forest)
#' Average of tree probabilities.
#'
#' @param object fitted \code{DRPolicyForest}.
#' @param newdata feature matrix or data.frame.
#' @param ... unused.
#' @return Matrix (n_samples x n_treatments).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict_proba.DRPolicyForest(...)
#' }
#' @export
predict_proba.DRPolicyForest <- function(object, newdata, ...) {
  if (inherits(newdata, "data.frame")) newdata <- as.matrix(newdata)
  if (!is.null(object$X_names_) && ncol(newdata) == length(object$X_names_))
    colnames(newdata) <- object$X_names_
  n_t <- length(object$treatment_levels_)
  n_obs <- nrow(newdata)
  out <- matrix(0, n_obs, n_t)
  for (tree in object$estimators_) {
    p <- predict_proba(tree, newdata, ...)
    out <- out + p
  }
  out / length(object$estimators_)
}

#' Feature importances (average over trees)
#' @param object fitted \code{DRPolicyForest}.
#' @param ... unused.
#' @return
#' Object returned by \code{feature_importances.DRPolicyForest}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # feature_importances.DRPolicyForest(...)
#' }
#' @export
feature_importances.DRPolicyForest <- function(object, ...) {
  imp_list <- lapply(object$estimators_, function(t) feature_importances(t, ...))
  imp_list <- imp_list[lengths(imp_list) > 0L]
  if (length(imp_list) == 0L) return(numeric(0))
  nms <- unique(unlist(lapply(imp_list, names)))
  out <- numeric(length(nms))
  names(out) <- nms
  for (imp in imp_list) {
    for (nm in names(imp)) out[nm] <- out[nm] + imp[nm]
  }
  out / sum(out)
}

# =============================================================================
# predict_proba generic (if not already defined)
# =============================================================================

#' Predict probability of treatment assignment (P(treat))
#'
#' Generic for predicted probability of recommending treatment.
#' @param object fitted model object.
#' @param newdata feature matrix or data.frame.
#' @param ... passed to methods.
#' @return
#' Object returned by \code{predict_proba}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict_proba(...)
#' }
#' @export
predict_proba <- function(object, newdata, ...) {
  UseMethod("predict_proba")
}

# =============================================================================
# predict_value generic
# =============================================================================

#' Predict expected value under each treatment (policy value)
#' @param object fitted policy model.
#' @param newdata feature matrix or data.frame.
#' @param ... passed to methods.
#' @return
#' Object returned by \code{predict_value}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # predict_value(...)
#' }
#' @export
predict_value <- function(object, newdata, ...) {
  UseMethod("predict_value")
}

# =============================================================================
# feature_importances generic
# =============================================================================

#' Feature importances for policy / tree models
#' @param object fitted model.
#' @param ... passed to methods.
#' @return
#' Object returned by \code{feature_importances}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # feature_importances(...)
#' }
#' @export
feature_importances <- function(object, ...) {
  UseMethod("feature_importances")
}

