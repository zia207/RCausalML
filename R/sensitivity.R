# CausalML-R: Sensitivity analysis (Placebo Treatment, Random Cause, Subset Data, Random Replace, Selection Bias)
# Reference: https://github.com/microsoft/dowhy/blob/master/dowhy/causal_refuters/
# Selection bias: Blackwell, Matthew. "A selection bias approach to sensitivity analysis
# for causal effects." Political Analysis 22.2 (2014): 169-182.
# https://www.mattblackwell.org/files/papers/causalsens.pdf

SUMMARY_COLS <- c("Method", "ATE", "New ATE", "New ATE LB", "New ATE UB")

# --- Confounding functions (Blackwell 2014) ---

#' One-sided confounding function (ATE)
#' @param alpha scalar or vector; confounding strength
#' @param p propensity score vector (0-1)
#' @param treatment treatment vector (1 = treated, 0 = control)
#' @return adjustment vector of length length(p)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # one_sided(...)
#' }
#' @export
one_sided <- function(alpha, p, treatment) {
  stopifnot(length(p) == length(treatment))
  alpha * (1 - p) * treatment - alpha * p * (1 - treatment)
}

#' Alignment confounding function (ATE)
#' @param alpha scalar or vector
#' @param p propensity score vector
#' @param treatment treatment vector
#' @return adjustment vector
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # alignment(...)
#' }
#' @export
alignment <- function(alpha, p, treatment) {
  stopifnot(length(p) == length(treatment))
  alpha * (1 - p) * treatment + alpha * p * (1 - treatment)
}

#' One-sided confounding function for ATT
#' @param alpha scalar or vector
#' @param p propensity score vector
#' @param treatment treatment vector
#' @return adjustment vector
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # one_sided_att(...)
#' }
#' @export
one_sided_att <- function(alpha, p, treatment) {
  stopifnot(length(p) == length(treatment))
  alpha * (1 - treatment)
}

#' Alignment confounding function for ATT
#' @param alpha scalar or vector
#' @param p propensity score vector
#' @param treatment treatment vector
#' @return adjustment vector
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # alignment_att(...)
#' }
#' @export
alignment_att <- function(alpha, p, treatment) {
  stopifnot(length(p) == length(treatment))
  alpha * (1 - treatment)
}

# --- Sensitivity object and internal helpers ---

#' Create a sensitivity analysis object
#'
#' @param df data.frame with outcome, treatment, propensity, and inference features
#' @param inference_features character vector of column names used for inference
#' @param p_col column name of propensity score
#' @param treatment_col column name of treatment (0/1)
#' @param outcome_col column name of outcome
#' @param learner fitted or unfitted meta-learner (e.g. from \code{XLearner(learner = "lm")}) that supports \code{fit()}, \code{predict()}, and \code{estimate_ate()}
#' @return list with components \code{df}, \code{inference_features}, \code{p_col}, \code{treatment_col}, \code{outcome_col}, \code{learner}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # sensitivity(...)
#' }
#' @export
sensitivity <- function(df,
                        inference_features,
                        p_col,
                        treatment_col,
                        outcome_col,
                        learner) {
  structure(list(
    df = df,
    inference_features = inference_features,
    p_col = p_col,
    treatment_col = treatment_col,
    outcome_col = outcome_col,
    learner = learner
  ), class = "sensitivity")
}

#' Get CATE predictions from learner (fit then predict)
#' @param sens sensitivity object
#' @param X feature matrix
#' @param p propensity vector
#' @param treatment treatment vector
#' @param y outcome vector
#' @return numeric vector of CATE predictions
#' @noRd
get_prediction_sensitivity <- function(sens, X, p, treatment, y) {
  sens$learner <- fit(sens$learner, X = X, treatment = treatment, y = y, p = p)
  obj <- sens$learner
  preds <- predict(obj, X)
  if (is.matrix(preds)) preds <- preds[, 1]
  as.numeric(preds)
}

#' Get ATE and confidence interval from learner
#' @param sens sensitivity object
#' @param X feature matrix
#' @param p propensity vector
#' @param treatment treatment vector
#' @param y outcome vector
#' @return list with ate, ate_lb, ate_ub
#' @noRd
get_ate_ci_sensitivity <- function(sens, X, p, treatment, y) {
  sens$learner <- fit(sens$learner, X = X, treatment = treatment, y = y, p = p)
  res <- estimate_ate(sens$learner, X, treatment, y, p = p, return_ci = TRUE)
  list(ate = res$ate, ate_lb = res$ate_lb, ate_ub = res$ate_ub)
}

# --- Method implementations ---

#' Placebo treatment: replace treatment with random assignment
#' @param sens sensitivity object
#' @return list with ate_new, ate_new_lower, ate_new_upper
#' @noRd
sensitivity_placebo_treatment <- function(sens) {
  n <- nrow(sens$df)
  X <- as.matrix(sens$df[sens$inference_features])
  p <- sens$df[[sens$p_col]]
  treatment_new <- sample(0:1, size = n, replace = TRUE)
  y <- sens$df[[sens$outcome_col]]
  get_ate_ci_sensitivity(sens, X, p, treatment_new, y)
}

#' Random cause: add irrelevant random covariate
#' @param sens sensitivity object
#' @return list with ate_new, ate_new_lower, ate_new_upper
#' @noRd
sensitivity_random_cause <- function(sens) {
  n <- nrow(sens$df)
  X <- as.matrix(sens$df[sens$inference_features])
  X_new <- cbind(X, rnorm(n))
  p <- sens$df[[sens$p_col]]
  treatment <- sens$df[[sens$treatment_col]]
  y <- sens$df[[sens$outcome_col]]
  get_ate_ci_sensitivity(sens, X_new, p, treatment, y)
}

#' Random replace: replace one covariate with random values
#' @param sens sensitivity object
#' @param replaced_feature column name to replace; if NULL, chosen at random
#' @return list with ate_new, ate_new_lower, ate_new_upper
#' @noRd
sensitivity_random_replace <- function(sens, replaced_feature = NULL) {
  n <- nrow(sens$df)
  if (is.null(replaced_feature)) {
    idx <- sample(length(sens$inference_features), 1L)
    replaced_feature <- sens$inference_features[idx]
  }
  df_new <- sens$df
  df_new[[replaced_feature]] <- rnorm(n)
  X_new <- as.matrix(df_new[sens$inference_features])
  p_new <- df_new[[sens$p_col]]
  treatment_new <- df_new[[sens$treatment_col]]
  y_new <- df_new[[sens$outcome_col]]
  get_ate_ci_sensitivity(sens, X_new, p_new, treatment_new, y_new)
}

#' Subset data: use a random subset
#' @param sens sensitivity object
#' @param sample_size fraction of data to keep (0-1)
#' @return list with ate_new, ate_new_lower, ate_new_upper
#' @noRd
sensitivity_subset_data <- function(sens, sample_size) {
  if (is.null(sample_size)) stop("Subset Data requires sample_size (fraction in 0-1).")
  n <- nrow(sens$df)
  idx <- sample.int(n, size = floor(n * sample_size))
  df_new <- sens$df[idx, ]
  X_new <- as.matrix(df_new[sens$inference_features])
  p_new <- df_new[[sens$p_col]]
  treatment_new <- df_new[[sens$treatment_col]]
  y_new <- df_new[[sens$outcome_col]]
  get_ate_ci_sensitivity(sens, X_new, p_new, treatment_new, y_new)
}

#' Confounding function by name
#' @noRd
confounding_functions <- function() {
  list(
    one_sided = one_sided,
    alignment = alignment,
    one_sided_att = one_sided_att,
    alignment_att = alignment_att
  )
}

#' Selection bias: causalsens over alpha range
#' @param sens sensitivity object
#' @param confound character: "one_sided", "alignment", "one_sided_att", "alignment_att"
#' @param alpha_range numeric vector of alpha values; if NULL, derived from outcome IQR
#' @param sensitivity_features character vector of features for partial R²; if NULL use inference_features
#' @return list with sens_df (data.frame: alpha, rsqs, New ATE, New ATE LB, New ATE UB), partial_rsqs_df (feature, partial_rsqs)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # causalsens_selection_bias(...)
#' }
#' @export
causalsens_selection_bias <- function(sens,
                                     confound = "one_sided",
                                     alpha_range = NULL,
                                     sensitivity_features = NULL) {
  cf <- confounding_functions()
  if (!confound %in% names(cf))
    stop("confound must be one of: ", paste(names(cf), collapse = ", "))
  confound_fun <- cf[[confound]]

  if (is.null(sensitivity_features))
    sensitivity_features <- sens$inference_features

  y <- sens$df[[sens$outcome_col]]
  if (is.null(alpha_range)) {
    iqr <- unname(quantile(y, 0.75) - quantile(y, 0.25))
    alpha_range <- seq(-iqr/2, iqr/2, length.out = 11)
    if (!0 %in% alpha_range)
      alpha_range <- c(alpha_range, 0)
  }
  alpha_range <- sort(alpha_range)

  X <- as.matrix(sens$df[sens$inference_features])
  p <- sens$df[[sens$p_col]]
  treatment <- sens$df[[sens$treatment_col]]

  preds <- get_prediction_sensitivity(sens, X, p, treatment, y)
  ate_at_0 <- mean(preds, na.rm = TRUE)

  sens_rows <- list()
  for (a in alpha_range) {
    adj <- confound_fun(a, p, treatment)
    preds_adj <- y - adj
    s_preds <- get_prediction_sensitivity(sens, X, p, treatment, preds_adj)
    ci <- get_ate_ci_sensitivity(sens, X, p, treatment, preds_adj)
    s_preds_residual <- preds_adj - s_preds
    rsqs <- (a^2 * var(treatment)) / max(1e-10, var(s_preds_residual))
    sens_rows[[length(sens_rows) + 1]] <- data.frame(
      alpha = a, rsqs = rsqs,
      "New ATE" = ci$ate, "New ATE LB" = ci$ate_lb, "New ATE UB" = ci$ate_ub,
      check.names = FALSE
    )
  }
  sens_df <- do.call(rbind, sens_rows)

  rss <- sum((y - preds)^2)
  partial_rsqs <- numeric(length(sensitivity_features))
  for (i in seq_along(sensitivity_features)) {
    feat <- sensitivity_features[i]
    drop_col <- setdiff(sens$inference_features, feat)
    if (length(drop_col) == 0) next
    X_reduced <- as.matrix(sens$df[drop_col])
    y_pred_reduced <- get_prediction_sensitivity(sens, X_reduced, p, treatment, y)
    rss_new <- sum((y - y_pred_reduced)^2)
    partial_rsqs[i] <- (rss_new - rss) / max(1e-10, rss)
  }
  partial_rsqs_df <- data.frame(
    feature = sensitivity_features,
    partial_rsqs = partial_rsqs,
    stringsAsFactors = FALSE
  )

  list(sens_df = sens_df, partial_rsqs_df = partial_rsqs_df)
}

#' Summary for a single sensitivity method (non–Selection Bias)
#' @param sens sensitivity object
#' @param method_name character label for the method
#' @return data.frame with columns Method, ATE, New ATE, New ATE LB, New ATE UB (one row)
#' @noRd
summary_sensitivity_single <- function(sens, method_name, ate_new, ate_new_lower, ate_new_upper) {
  X <- as.matrix(sens$df[sens$inference_features])
  p <- sens$df[[sens$p_col]]
  treatment <- sens$df[[sens$treatment_col]]
  y <- sens$df[[sens$outcome_col]]
  preds <- get_prediction_sensitivity(sens, X, p, treatment, y)
  ate <- mean(preds, na.rm = TRUE)
  data.frame(
    Method = method_name,
    ATE = ate,
    "New ATE" = ate_new,
    "New ATE LB" = ate_new_lower,
    "New ATE UB" = ate_new_upper,
    check.names = FALSE
  )
}

#' Run sensitivity analysis for one or more methods
#'
#' @param sens sensitivity object from \code{sensitivity()}
#' @param methods character vector of method names: "Placebo Treatment", "Random Cause", "Subset Data", "Random Replace", "Selection Bias"
#' @param sample_size fraction for "Subset Data" (e.g. 0.5); required when "Subset Data" is in \code{methods}
#' @param confound confounding function for "Selection Bias": "one_sided", "alignment", "one_sided_att", "alignment_att"
#' @param alpha_range numeric vector of alpha values for Selection Bias; if NULL, derived from outcome IQR
#' @param sensitivity_features character vector for partial R² in Selection Bias; if NULL use inference features
#' @param replaced_feature for "Random Replace", optional column name to replace; if NULL a random one is chosen
#' @return data.frame with columns \code{Method}, \code{ATE}, \code{New ATE}, \code{New ATE LB}, \code{New ATE UB}. For "Selection Bias" one row per alpha value.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # sensitivity_analysis(...)
#' }
#' @export
sensitivity_analysis <- function(sens,
                                 methods = c("Placebo Treatment", "Random Cause", "Subset Data", "Random Replace", "Selection Bias"),
                                 sample_size = NULL,
                                 confound = "one_sided",
                                 alpha_range = NULL,
                                 sensitivity_features = NULL,
                                 replaced_feature = NULL) {
  valid_methods <- c("Placebo Treatment", "Random Cause", "Subset Data", "Random Replace", "Selection Bias")
  bad <- setdiff(methods, valid_methods)
  if (length(bad))
    stop("Invalid methods: ", paste(bad, collapse = ", "), ". Choose from: ", paste(valid_methods, collapse = ", "))

  if ("Subset Data" %in% methods && is.null(sample_size))
    stop("sample_size is required when 'Subset Data' is in methods.")

  if (is.null(alpha_range) && "Selection Bias" %in% methods) {
    y <- sens$df[[sens$outcome_col]]
    iqr <- unname(quantile(y, 0.75) - quantile(y, 0.25))
    alpha_range <- seq(-iqr/2, iqr/2, length.out = 11)
    if (!0 %in% alpha_range)
      alpha_range <- c(alpha_range, 0)
    alpha_range <- sort(alpha_range)
  }

  summary_list <- list()
  for (method in methods) {
    if (method == "Placebo Treatment") {
      res <- sensitivity_placebo_treatment(sens)
      summary_list[[length(summary_list) + 1]] <- summary_sensitivity_single(
        sens, method, res$ate, res$ate_lb, res$ate_ub)
    } else if (method == "Random Cause") {
      res <- sensitivity_random_cause(sens)
      summary_list[[length(summary_list) + 1]] <- summary_sensitivity_single(
        sens, method, res$ate, res$ate_lb, res$ate_ub)
    } else if (method == "Subset Data") {
      res <- sensitivity_subset_data(sens, sample_size)
      label <- paste0("Subset Data (sample size @", sample_size, ")")
      summary_list[[length(summary_list) + 1]] <- summary_sensitivity_single(
        sens, label, res$ate, res$ate_lb, res$ate_ub)
    } else if (method == "Random Replace") {
      res <- sensitivity_random_replace(sens, replaced_feature)
      summary_list[[length(summary_list) + 1]] <- summary_sensitivity_single(
        sens, method, res$ate, res$ate_lb, res$ate_ub)
    } else if (method == "Selection Bias") {
      out <- causalsens_selection_bias(sens, confound = confound, alpha_range = alpha_range,
                                      sensitivity_features = sensitivity_features)
      sens_df <- out$sens_df
      ate_at_0 <- sens_df[sens_df$alpha == 0, "New ATE", drop = TRUE]
      if (length(ate_at_0) == 0) ate_at_0 <- sens_df[1, "New ATE"]
      sens_df$Method <- paste0("Selection Bias (alpha@", round(sens_df$alpha, 5), ", with r-square:", round(sens_df$rsqs, 5), ")")
      sens_df$ATE <- ate_at_0[1]
      summary_list[[length(summary_list) + 1]] <- sens_df[, SUMMARY_COLS]
    }
  }

  do.call(rbind, summary_list)
}

#' Plot sensitivity analysis (Selection Bias causalsens output)
#'
#' @param sens_df data.frame from \code{causalsens_selection_bias()$sens_df} with columns \code{alpha}, \code{rsqs}, \code{New ATE}, \code{New ATE LB}, \code{New ATE UB}
#' @param partial_rsqs_df optional data.frame from \code{causalsens_selection_bias()$partial_rsqs_df} (feature, partial_rsqs)
#' @param type "raw" (x = alpha) or "r.squared" (x = rsqs)
#' @param ci whether to draw confidence interval ribbon
#' @param partial_rsqs whether to add partial R² points when \code{type = "r.squared"} and \code{partial_rsqs_df} is provided
#' @return ggplot object if ggplot2 is available, otherwise base plot (invisible)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # plot_sensitivity(...)
#' }
#' @export
plot_sensitivity <- function(sens_df,
                             partial_rsqs_df = NULL,
                             type = c("raw", "r.squared"),
                             ci = FALSE,
                             partial_rsqs = FALSE) {
  type <- match.arg(type)
  if (!all(c("alpha", "rsqs", "New ATE", "New ATE LB", "New ATE UB") %in% names(sens_df)))
    stop("sens_df must contain alpha, rsqs, 'New ATE', 'New ATE LB', 'New ATE UB'.")

  y_max <- round(max(sens_df[["New ATE UB"]], na.rm = TRUE) * 1.1, 4)
  y_min <- round(min(sens_df[["New ATE LB"]], na.rm = TRUE) * 0.9, 4)
  ate_col <- "New ATE"
  lb_col <- "New ATE LB"
  ub_col <- "New ATE UB"

  if (type == "raw") {
    x_vals <- sens_df$alpha
    x_max <- round(max(x_vals, na.rm = TRUE) * 1.1, 4)
    x_min <- round(min(x_vals, na.rm = TRUE) * 0.9, 4)
  } else {
    x_vals <- sens_df$rsqs
    x_max <- max(x_vals, na.rm = TRUE) * 1.1
    x_min <- min(x_vals, na.rm = TRUE) * 0.9
  }

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    x_name <- if (type == "raw") "alpha" else "rsqs"
    p <- ggplot2::ggplot(sens_df, ggplot2::aes(x = .data[[x_name]], y = .data[[ate_col]]))
    if (ci)
      p <- p + ggplot2::geom_ribbon(ggplot2::aes(ymin = .data[[lb_col]], ymax = .data[[ub_col]]), fill = "gray", alpha = 0.5)
    p <- p + ggplot2::geom_line(linewidth = 1) +
      ggplot2::ylim(y_min, y_max)
    if (type == "raw")
      p <- p + ggplot2::xlim(x_min, x_max) + ggplot2::labs(x = "Alpha", y = "Adjusted ATE")
    else
      p <- p + ggplot2::labs(x = "R-squared", y = "Adjusted ATE")
    if (partial_rsqs && type == "r.squared" && !is.null(partial_rsqs_df) && nrow(partial_rsqs_df) > 0) {
      ate_at_0 <- sens_df[sens_df$alpha == 0, ate_col, drop = TRUE]
      if (length(ate_at_0) == 0) ate_at_0 <- sens_df[1, ate_col]
      pt_df <- data.frame(
        x = partial_rsqs_df$partial_rsqs,
        y = rep(ate_at_0[1], nrow(partial_rsqs_df))
      )
      p <- p + ggplot2::geom_point(data = pt_df, ggplot2::aes(x = .data$x, y = .data$y),
                                   shape = 4, color = "red", size = 3)
    }
    return(p)
  }

  # Base R fallback
  plot(x_vals, sens_df[[ate_col]], type = "l", xlim = c(x_min, x_max), ylim = c(y_min, y_max),
       xlab = if (type == "raw") "Alpha" else "R-squared", ylab = "Adjusted ATE")
  if (ci)
    polygon(c(x_vals, rev(x_vals)), c(sens_df[[ub_col]], rev(sens_df[[lb_col]])), col = "gray", border = NA)
  lines(x_vals, sens_df[[ate_col]], lwd = 2)
  if (partial_rsqs && type == "r.squared" && !is.null(partial_rsqs_df)) {
    ate_at_0 <- sens_df[sens_df$alpha == 0, ate_col, drop = TRUE]
    if (length(ate_at_0) == 0) ate_at_0 <- sens_df[1, ate_col]
    points(partial_rsqs_df$partial_rsqs, rep(ate_at_0[1], nrow(partial_rsqs_df)), pch = 4, col = "red", cex = 2)
  }
  invisible(NULL)
}

#' Map partial R² of a feature to confounding amount (alpha range)
#'
#' Finds alpha values in \code{sens_df} whose \code{rsqs} is within \code{range} of \code{partial_rsqs_value}.
#'
#' @param sens_df data.frame from \code{causalsens_selection_bias()$sens_df}
#' @param feature_name label for the feature (for messages)
#' @param partial_rsqs_value partial R² value of the feature
#' @param range fraction around \code{partial_rsqs_value} to search (default 0.01 = 1\%)
#' @return numeric vector of length 2: (confounding_min, confounding_max), or NULL if none in range
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # partial_rsqs_confounding(...)
#' }
#' @export
partial_rsqs_confounding <- function(sens_df, feature_name, partial_rsqs_value, range = 0.01) {
  lo <- partial_rsqs_value - partial_rsqs_value * range
  hi <- partial_rsqs_value + partial_rsqs_value * range
  idx <- sens_df$rsqs >= lo & sens_df$rsqs <= hi
  if (!any(idx)) {
    message("Cannot find corresponding R-squared value within the range; try a larger range or different alpha_range.")
    return(NULL)
  }
  alphas <- sens_df$alpha[idx]
  confounding_min <- min(alphas)
  confounding_max <- max(alphas)
  message("For feature ", feature_name, " with partial R-squared ", partial_rsqs_value,
          " confounding amount (alpha) in [", confounding_min, ", ", confounding_max, "]")
  c(confounding_min, confounding_max)
}
