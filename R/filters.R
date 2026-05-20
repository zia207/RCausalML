# CausalML-R: Filter feature selection for uplift modeling
# For classification: outcome is binary. Port of Python CausalML FilterSelect.
# Reference: Zhao et al. (2020) "Feature Selection Methods for Uplift Modeling"

# --- F-test filter (interaction between treatment and feature) ---

#' F-test of treatment-feature interaction for one feature (internal)
#' @param data data.frame with outcome, treatment, and feature
#' @param treatment_indicator column name for binary treatment (1) or control (0)
#' @param feature_name column name of the feature
#' @param y_name column name of the outcome
#' @param order 1=linear, 2=quadratic+linear, 3=cubic+quadratic+linear
#' @return data.frame with feature, method, score (F), p_value, misc
#' @noRd
filter_F_one_feature <- function(data, treatment_indicator, feature_name, y_name, order = 1) {
  if (!order %in% c(1, 2, 3))
    stop("order must be 1, 2, or 3")

  Y <- data[[y_name]]
  tr <- data[[treatment_indicator]]
  x <- data[[feature_name]]

  # Build design: intercept, treatment, feature, treatment*feature; optional higher orders
  X_df <- data.frame(
    const = 1,
    tr = tr,
    x = x,
    tr_x = tr * x
  )
  if (order >= 2) {
    x_o2 <- x^2
    X_df$x_o2 <- x_o2
    X_df$tr_x_o2 <- tr * x_o2
  }
  if (order >= 3) {
    x_o3 <- x^3
    X_df$x_o3 <- x_o3
    X_df$tr_x_o3 <- tr * x_o3
  }

  # Full model (all terms)
  full_form <- stats::as.formula(paste0("Y ~ . - const"))
  full_lm <- stats::lm(full_form, data = cbind(Y = Y, X_df))

  # Restricted model (no treatment*feature interaction terms)
  if (order == 1) {
    restr_form <- stats::as.formula("Y ~ tr + x")
  } else if (order == 2) {
    restr_form <- stats::as.formula("Y ~ tr + x + x_o2")
  } else {
    restr_form <- stats::as.formula("Y ~ tr + x + x_o2 + x_o3")
  }
  restr_lm <- stats::lm(restr_form, data = cbind(Y = Y, X_df))

  aa <- stats::anova(restr_lm, full_lm)
  fval <- aa$`F`[2]
  pval <- aa$`Pr(>F)`[2]
  df_num <- aa$Df[2]
  df_denom <- full_lm$df.residual

  data.frame(
    feature = feature_name,
    method = paste0("F", order, " Filter"),
    score = fval,
    p_value = pval,
    misc = sprintf("df_num: %s, df_denom: %s, order:%s", df_num, df_denom, order),
    stringsAsFactors = FALSE
  )
}

#' Rank features by F-statistic of treatment-feature interaction
#'
#' F-test of the interaction between treatment and each feature (linear, quadratic, or cubic).
#'
#' @param data data.frame containing outcome, features, and treatment
#' @param treatment_indicator column name for binary treatment (1) or control (0)
#' @param features character vector of feature column names
#' @param y_name column name of the outcome variable
#' @param order 1=linear, 2=quadratic+linear, 3=cubic+quadratic+linear
#' @return data.frame with columns feature, method, score, p_value, misc, rank (sorted by score descending)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # filter_F(...)
#' }
#' @export
filter_F <- function(data, treatment_indicator, features, y_name, order = 1) {
  if (!order %in% c(1, 2, 3))
    stop("order must be 1, 2, or 3")

  all_result <- do.call(rbind, lapply(features, function(fname) {
    filter_F_one_feature(
      data = data,
      treatment_indicator = treatment_indicator,
      feature_name = fname,
      y_name = y_name,
      order = order
    )
  }))
  all_result <- all_result[order(-all_result$score), ]
  all_result$rank <- rank(-all_result$score, ties.method = "first")
  row.names(all_result) <- NULL
  all_result
}

# --- LR (Likelihood Ratio) filter ---

#' LR test of treatment-feature interaction for one feature (internal)
#' @param data data.frame with outcome, treatment, and feature
#' @param treatment_indicator column name for binary treatment
#' @param feature_name column name of the feature
#' @param y_name column name of the outcome
#' @param order 1, 2, or 3
#' @param disp unused (kept for API compatibility)
#' @return data.frame with feature, method, score (LR stat), p_value, misc
#' @noRd
filter_LR_one_feature <- function(data, treatment_indicator, feature_name, y_name, order = 1, disp = TRUE) {
  if (!order %in% c(1, 2, 3))
    stop("order must be 1, 2, or 3")

  Y <- data[[y_name]]
  tr <- data[[treatment_indicator]]
  x <- data[[feature_name]]

  X_df <- data.frame(
    const = 1,
    tr = tr,
    x = x,
    tr_x = tr * x
  )
  if (order >= 2) {
    x_o2 <- x^2
    X_df$x_o2 <- x_o2
    X_df$tr_x_o2 <- tr * x_o2
  }
  if (order >= 3) {
    x_o3 <- x^3
    X_df$x_o3 <- x_o3
    X_df$tr_x_o3 <- tr * x_o3
  }

  # Restricted: no interaction terms
  if (order == 1) {
    restr_form <- stats::as.formula("Y ~ tr + x")
  } else if (order == 2) {
    restr_form <- stats::as.formula("Y ~ tr + x + x_o2")
  } else {
    restr_form <- stats::as.formula("Y ~ tr + x + x_o2 + x_o3")
  }
  restr_glm <- stats::glm(restr_form, data = cbind(Y = Y, X_df), family = stats::binomial())

  full_form <- stats::as.formula(paste0("Y ~ . - const"))
  full_glm <- stats::glm(full_form, data = cbind(Y = Y, X_df), family = stats::binomial())

  ll_r <- stats::logLik(restr_glm)
  ll_f <- stats::logLik(full_glm)
  LR_stat <- as.numeric(-2 * (ll_r - ll_f))
  LR_df <- attr(ll_f, "df") - attr(ll_r, "df")
  LR_pvalue <- stats::pchisq(LR_stat, df = LR_df, lower.tail = FALSE)

  data.frame(
    feature = feature_name,
    method = paste0("LR", order, " Filter"),
    score = LR_stat,
    p_value = LR_pvalue,
    misc = sprintf("df: %s, order: %s", LR_df, order),
    stringsAsFactors = FALSE
  )
}

#' Rank features by likelihood-ratio statistic of treatment-feature interaction
#'
#' Compares logistic regression with vs without treatment-by-feature interaction(s).
#'
#' @param data data.frame containing outcome, features, and treatment
#' @param treatment_indicator column name for binary treatment (1 or 0)
#' @param features character vector of feature column names
#' @param y_name column name of the outcome variable
#' @param order 1=linear, 2=quadratic+linear, 3=cubic+quadratic+linear
#' @param disp unused; kept for API compatibility with Python
#' @return data.frame with feature, method, score, p_value, misc, rank (sorted by score descending)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # filter_LR(...)
#' }
#' @export
filter_LR <- function(data, treatment_indicator, features, y_name, order = 1, disp = TRUE) {
  if (!order %in% c(1, 2, 3))
    stop("order must be 1, 2, or 3")

  all_result <- do.call(rbind, lapply(features, function(fname) {
    filter_LR_one_feature(
      data = data,
      treatment_indicator = treatment_indicator,
      feature_name = fname,
      y_name = y_name,
      order = order,
      disp = disp
    )
  }))
  all_result <- all_result[order(-all_result$score), ]
  all_result$rank <- rank(-all_result$score, ties.method = "first")
  row.names(all_result) <- NULL
  all_result
}

# --- Node summary and divergence (for bin-based filters) ---

#' Node summary: conversion counts and probabilities by treatment group
#'
#' Used for uplift tree node split evaluation (bin-based divergence filters).
#'
#' @param data data.frame with experiment group and outcome columns
#' @param experiment_group_column column name for treatment group
#' @param y_name column name for binary outcome
#' @param smooth if TRUE, add 1 to counts to avoid zero divisions
#' @return list with \code{results} (counts by group and outcome) and \code{nodeSummary} (conversion prob and n by group)
#' @noRd
get_node_summary <- function(data,
                             experiment_group_column = "treatment_group_key",
                             y_name = "conversion",
                             smooth = TRUE) {
  tab <- table(data[[experiment_group_column]], data[[y_name]])
  treatment_group_keys <- rownames(tab)
  y_keys <- as.character(colnames(tab))

  results <- list()
  for (ti in treatment_group_keys) {
    results[[ti]] <- list()
    for (yi in y_keys) {
      count <- tab[ti, yi]
      if (smooth && (is.na(count) || count == 0))
        count <- 1
      results[[ti]][[yi]] <- count
    }
  }

  nodeSummary <- list()
  for (tg in names(results)) {
    n1 <- results[[tg]][["1"]] %||% 0
    n0 <- results[[tg]][["0"]] %||% 0
    n_total <- n0 + n1
    y_mean <- n_total
    if (n_total > 0) y_mean <- n1 / n_total else y_mean <- 0
    nodeSummary[[tg]] <- c(y_mean, n_total)
  }
  list(results = results, nodeSummary = nodeSummary)
}

# %||% for default when NULL
`%||%` <- function(x, y) if (is.null(x)) y else x

#' KL divergence for binary classification
#' @param pk probability of class 1 in treatment group
#' @param qk probability of class 1 in control group
#' @return scalar
#' @noRd
kl_divergence <- function(pk, qk) {
  qk <- max(1e-6, min(1 - 1e-6, qk))
  pk * log(pk / qk) + (1 - pk) * log((1 - pk) / (1 - qk))
}

#' Multi-treatment unconditional D with KL divergence (one node)
#' @param nodeSummary list of [conversion_prob, n] by treatment group
#' @param control_group name of control group
#' @return scalar
#' @noRd
evaluate_KL <- function(nodeSummary, control_group = "control") {
  if (!control_group %in% names(nodeSummary)) return(0)
  pc <- nodeSummary[[control_group]][1]
  d_res <- 0
  for (tg in names(nodeSummary)) {
    if (tg != control_group)
      d_res <- d_res + kl_divergence(nodeSummary[[tg]][1], pc)
  }
  d_res
}

#' Multi-treatment unconditional D with Euclidean distance (one node)
#' @param nodeSummary list of [conversion_prob, n] by treatment group
#' @param control_group name of control group
#' @return scalar
#' @noRd
evaluate_ED <- function(nodeSummary, control_group = "control") {
  if (!control_group %in% names(nodeSummary)) return(0)
  pc <- nodeSummary[[control_group]][1]
  d_res <- 0
  for (tg in names(nodeSummary)) {
    if (tg != control_group)
      d_res <- d_res + 2 * (nodeSummary[[tg]][1] - pc)^2
  }
  d_res
}

#' Multi-treatment unconditional D with Chi-Square (one node)
#' @param nodeSummary list of [conversion_prob, n] by treatment group
#' @param control_group name of control group
#' @return scalar
#' @noRd
evaluate_Chi <- function(nodeSummary, control_group = "control") {
  if (!control_group %in% names(nodeSummary)) return(0)
  pc <- nodeSummary[[control_group]][1]
  d_res <- 0
  for (tg in names(nodeSummary)) {
    if (tg != control_group) {
      pt <- nodeSummary[[tg]][1]
      d_res <- d_res + (pt - pc)^2 / max(1e-6, pc) +
        (pt - pc)^2 / max(1e-6, 1 - pc)
    }
  }
  d_res
}

#' Divergence score for one feature (bin-based uplift filter)
#' @param data data.frame with outcome, feature, and experiment group
#' @param feature_name column name of the feature
#' @param y_name column name of the outcome
#' @param n_bins number of quantile-based bins
#' @param method "KL", "ED", or "Chi"
#' @param control_group value for control in experiment_group_column
#' @param experiment_group_column column name for treatment assignment
#' @param null_impute "mean", "median", or "most_frequent"; if NULL and NA present, error
#' @return data.frame with feature, method, score, p_value (NA), misc
#' @noRd
filter_D_one_feature <- function(data,
                                 feature_name,
                                 y_name,
                                 n_bins = 10,
                                 method = "KL",
                                 control_group = "control",
                                 experiment_group_column = "treatment_group_key",
                                 null_impute = NULL) {
  if (!method %in% c("KL", "ED", "Chi"))
    stop("method must be 'KL', 'ED', or 'Chi'")

  eval_fun <- switch(method,
    KL = evaluate_KL,
    ED = evaluate_ED,
    Chi = evaluate_Chi,
    stop("method must be KL, ED, or Chi")
  )

  x_vec <- data[[feature_name]]
  if (any(is.na(x_vec))) {
    if (is.null(null_impute))
      stop("NA present in column '", feature_name, "'. Impute or set null_impute.")
    if (null_impute == "mean") {
      x_vec[is.na(x_vec)] <- mean(x_vec, na.rm = TRUE)
    } else if (null_impute == "median") {
      x_vec[is.na(x_vec)] <- stats::median(x_vec, na.rm = TRUE)
    } else if (null_impute == "most_frequent") {
      u <- unique(na.omit(x_vec))
      mode_val <- u[which.max(tabulate(match(x_vec, u)))]
      x_vec[is.na(x_vec)] <- mode_val
    } else {
      stop("null_impute must be 'mean', 'median', or 'most_frequent'")
    }
  }

  # Quantile-based bins, drop duplicate edges (as in Python qcut(..., duplicates="drop"))
  probs <- seq(0, 1, length.out = n_bins + 1)
  breaks <- unique(stats::quantile(x_vec, probs = probs, na.rm = TRUE))
  if (length(breaks) < 2) {
    x_bin <- rep(0, length(x_vec))
    n_actual_bins <- 1
  } else {
    x_bin <- as.integer(cut(x_vec, breaks = breaks, include.lowest = TRUE, labels = FALSE))
    x_bin[is.na(x_bin)] <- 1
    n_actual_bins <- max(x_bin, na.rm = TRUE)
  }

  totalSize <- nrow(data)
  d_children <- 0
  data_temp <- data
  data_temp[[feature_name]] <- x_vec
  data_temp$.x_bin <- x_bin

  for (i_bin in seq_len(n_actual_bins)) {
    sub <- data_temp[data_temp$.x_bin == i_bin, ]
    if (nrow(sub) == 0) next
    ns <- get_node_summary(sub, experiment_group_column = experiment_group_column, y_name = y_name)[[2]]
    nodeScore <- eval_fun(ns, control_group = control_group)
    nodeSize <- sum(vapply(ns, function(x) x[2], 0))
    d_children <- d_children + nodeScore * nodeSize / totalSize
  }

  parentNS <- get_node_summary(data_temp, experiment_group_column = experiment_group_column, y_name = y_name)[[2]]
  d_parent <- eval_fun(parentNS, control_group = control_group)
  d_res <- d_children - d_parent

  data.frame(
    feature = feature_name,
    method = method,
    score = d_res,
    p_value = NA_real_,
    misc = sprintf("number_of_bins: %s", n_actual_bins),
    stringsAsFactors = FALSE
  )
}

#' Rank features by divergence measure (bin-based uplift filter)
#'
#' Bins each feature by quantiles and computes KL, Euclidean distance, or Chi-Square
#' divergence between treatment and control conversion rates (weighted by children vs parent).
#'
#' @param data data.frame containing outcome, features, and experiment group
#' @param features character vector of feature column names
#' @param y_name column name of the outcome variable
#' @param n_bins number of quantile-based bins
#' @param method "KL", "ED", or "Chi"
#' @param control_group value in \code{experiment_group_column} for control
#' @param experiment_group_column column name for treatment assignment
#' @param null_impute "mean", "median", or "most_frequent"; if NULL and NA present, error
#' @return data.frame with feature, method, score, p_value, misc, rank (sorted by score descending)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # filter_D(...)
#' }
#' @export
filter_D <- function(data,
                     features,
                     y_name,
                     n_bins = 10,
                     method = "KL",
                     control_group = "control",
                     experiment_group_column = "treatment_group_key",
                     null_impute = NULL) {
  all_result <- do.call(rbind, lapply(features, function(fname) {
    filter_D_one_feature(
      data = data,
      feature_name = fname,
      y_name = y_name,
      n_bins = n_bins,
      method = method,
      control_group = control_group,
      experiment_group_column = experiment_group_column,
      null_impute = null_impute
    )
  }))
  all_result <- all_result[order(-all_result$score), ]
  all_result$rank <- rank(-all_result$score, ties.method = "first")
  row.names(all_result) <- NULL
  all_result
}

# --- Main entry point ---

#' Feature importance for uplift modeling (filter methods)
#'
#' Ranks features by strength of treatment-by-feature interaction or by bin-based
#' divergence. For binary outcome (classification) only.
#'
#' Methods:
#' - \code{F}: F-test of treatment-feature interaction (linear/quadratic/cubic).
#' - \code{LR}: Likelihood-ratio test (logistic regression) of interaction.
#' - \code{KL}, \code{ED}, \code{Chi}: Bin-based divergence (KL, Euclidean, Chi-Square).
#'
#' @param data data.frame containing outcome, features, and experiment group column
#' @param features character vector of feature column names
#' @param y_name column name of the outcome variable (binary)
#' @param method one of \code{"F"}, \code{"LR"}, \code{"KL"}, \code{"ED"}, \code{"Chi"}
#' @param experiment_group_column column name for treatment assignment (e.g. \code{"treatment_group_key"})
#' @param control_group value in that column for control (e.g. \code{"control"})
#' @param treatment_group value in that column for treatment (e.g. \code{"treatment1"})
#' @param n_bins number of bins for KL/ED/Chi (default 5)
#' @param null_impute for KL/ED/Chi: \code{"mean"}, \code{"median"}, or \code{"most_frequent"}; if NULL and NA present, error
#' @param order for F and LR: 1=linear, 2=quadratic+linear, 3=cubic+quadratic+linear (default 1)
#' @param disp unused; kept for API compatibility
#' @return data.frame with columns \code{method}, \code{feature}, \code{rank}, \code{score}, \code{p_value}, \code{misc}
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # get_importance(...)
#' }
#' @export
get_importance <- function(data,
                          features,
                          y_name,
                          method,
                          experiment_group_column = "treatment_group_key",
                          control_group = "control",
                          treatment_group = "treatment",
                          n_bins = 5,
                          null_impute = NULL,
                          order = 1,
                          disp = FALSE) {
  if (!method %in% c("F", "LR", "KL", "ED", "Chi"))
    stop("method must be 'F', 'LR', 'KL', 'ED', or 'Chi'")

  if (method == "F" || method == "LR") {
    data <- data[data[[experiment_group_column]] %in% c(control_group, treatment_group), ]
    data$treatment_indicator <- 0
    data[data[[experiment_group_column]] == treatment_group, "treatment_indicator"] <- 1

    if (method == "F") {
      all_result <- filter_F(
        data = data,
        treatment_indicator = "treatment_indicator",
        features = features,
        y_name = y_name,
        order = order
      )
    } else {
      all_result <- filter_LR(
        data = data,
        treatment_indicator = "treatment_indicator",
        features = features,
        y_name = y_name,
        order = order,
        disp = disp
      )
    }
  } else {
    all_result <- filter_D(
      data = data,
      features = features,
      y_name = y_name,
      n_bins = n_bins,
      method = method,
      control_group = control_group,
      experiment_group_column = experiment_group_column,
      null_impute = null_impute
    )
  }

  all_result$method <- paste0(method, " filter")
  all_result[, c("method", "feature", "rank", "score", "p_value", "misc")]
}
