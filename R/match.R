# CausalML-R: Propensity score matching
# Port of causalml.match (NearestNeighborMatch, create_table_one, MatchOptimizer)

#' Standardized Mean Difference (SMD) of a feature between treatment and control
#'
#' Definition: (mean_t - mean_c) / sqrt(0.5 * (var_t + var_c)).
#' See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/#s11title
#'
#' @param feature numeric vector (one covariate)
#' @param treatment binary vector (1 = treatment, 0 = control)
#' @return numeric SMD value
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # smd(...)
#' }
#' @export
smd <- function(feature, treatment) {
  t_vals <- feature[treatment == 1]
  c_vals <- feature[treatment == 0]
  mt <- mean(t_vals, na.rm = TRUE)
  mc <- mean(c_vals, na.rm = TRUE)
  vt <- stats::var(t_vals, na.rm = TRUE)
  vc <- stats::var(c_vals, na.rm = TRUE)
  pooled_sd <- sqrt(0.5 * (vt + vc))
  if (is.na(pooled_sd) || pooled_sd <= 0) return(NA_real_)
  (mt - mc) / pooled_sd
}


#' Create Table One: balance table for treatment vs control
#'
#' Reports means (and optionally SD) by group and SMD for each feature.
#' References: R tableone (CRAN), Python tableone (PyPi).
#'
#' @param data data.frame (total or matched sample)
#' @param treatment_col character; column name for treatment (0/1)
#' @param features character vector; column names of covariates
#' @param with_std logical; if TRUE, format as "mean (sd)"
#' @param with_counts logical; if TRUE, include a row "n" with group counts
#' @return data.frame with columns Control, Treatment, SMD; rows = n (optional) + features
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # create_table_one(...)
#' }
#' @export
create_table_one <- function(data,
                             treatment_col = "w",
                             features,
                             with_std = TRUE,
                             with_counts = TRUE) {
  trt <- data[[treatment_col]]
  if (is.character(trt) || is.factor(trt)) {
    trt <- as.integer(trt != levels(trt)[1L])
  } else {
    trt <- as.integer(trt)
  }
  control_vals <- data[trt == 0, features, drop = FALSE]
  treated_vals <- data[trt == 1, features, drop = FALSE]

  fmt <- function(x, std) {
    if (std) sprintf("%.2f (%.2f)", mean(x, na.rm = TRUE), stats::sd(x, na.rm = TRUE))
    else sprintf("%.2f", mean(x, na.rm = TRUE))
  }

  control_col <- vapply(features, function(f) fmt(control_vals[[f]], with_std), character(1))
  treated_col <- vapply(features, function(f) fmt(treated_vals[[f]], with_std), character(1))
  smd_col <- vapply(features, function(f) smd(data[[f]], trt), numeric(1))

  t1 <- data.frame(
    Variable = features,
    Control = control_col,
    Treatment = treated_col,
    SMD = round(smd_col, 4),
    stringsAsFactors = FALSE
  )
  rownames(t1) <- features

  if (with_counts) {
    n_control <- sum(trt == 0)
    n_treated <- sum(trt == 1)
    n_row <- data.frame(
      Variable = "n",
      Control = as.character(n_control),
      Treatment = as.character(n_treated),
      SMD = "",
      stringsAsFactors = FALSE
    )
    rownames(n_row) <- "n"
    t1 <- rbind(n_row, t1)
  }
  t1
}


#' Nearest neighbor propensity score matching
#'
#' Matches on propensity (and optionally other columns). Supports caliper,
#' with/without replacement, ratio, shuffle, and matching treatment-to-control
#' or control-to-treatment. When \code{replace=TRUE}, multiple \code{score_cols}
#' are supported (with scaling); when \code{replace=FALSE}, only a single
#' score column is allowed.
#'
#' @param data data.frame containing treatment and score columns
#' @param treatment_col name of treatment column (0/1)
#' @param score_cols character vector; column name(s) for matching (e.g. propensity)
#' @param caliper threshold in SD units (or in scaled units when replace=TRUE) to accept a match (default 0.2)
#' @param replace logical; allow reuse of controls (default FALSE)
#' @param ratio integer; number of control units per treated (default 1)
#' @param shuffle logical; shuffle treatment group before matching (default TRUE)
#' @param treatment_to_control logical; if TRUE match treatment to control, else control to treatment (default TRUE)
#' @param random_state integer or NULL; seed for reproducibility
#' @return data.frame with matched sample (treated + matched controls)
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # nearest_neighbor_match(...)
#' }
#' @export
nearest_neighbor_match <- function(data,
                                   treatment_col = "w",
                                   score_cols = "e",
                                   caliper = 0.2,
                                   replace = FALSE,
                                   ratio = 1L,
                                   shuffle = TRUE,
                                   treatment_to_control = TRUE,
                                   random_state = NULL) {
  if (!is.character(score_cols) && !is.null(score_cols)) score_cols <- as.character(score_cols)
  if (length(score_cols) == 0) stop("score_cols must be a non-empty character vector")
  if (!is.null(random_state)) set.seed(random_state)

  trt <- data[[treatment_col]]
  if (is.character(trt) || is.factor(trt)) {
    trt_num <- as.integer(trt != levels(trt)[1L])
  } else {
    trt_num <- as.integer(trt)
  }
  treated_idx <- which(trt_num == 1)
  control_idx <- which(trt_num == 0)
  if (length(treated_idx) == 0 || length(control_idx) == 0) {
    stop("Need both treated and control units.")
  }

  match_from_df <- data[treated_idx, score_cols, drop = FALSE]
  match_to_df   <- data[control_idx, score_cols, drop = FALSE]
  rownames(match_from_df) <- treated_idx
  rownames(match_to_df)   <- control_idx

  if (!treatment_to_control) {
    match_from_df <- data[control_idx, score_cols, drop = FALSE]
    match_to_df   <- data[treated_idx, score_cols, drop = FALSE]
    rownames(match_from_df) <- control_idx
    rownames(match_to_df)   <- treated_idx
  }

  n_from <- nrow(match_from_df)
  n_to   <- nrow(match_to_df)
  p      <- length(score_cols)

  if (replace) {
    # Scale using full data (treatment + control) for consistent units
    all_scores <- data[, score_cols, drop = FALSE]
    sc <- scale(all_scores)
    center <- attr(sc, "scaled:center")
    scale_attr <- attr(sc, "scaled:scale")
    scale_attr[scale_attr == 0] <- 1
    match_from_scaled <- scale(match_from_df, center = center, scale = scale_attr)
    match_to_scaled   <- scale(match_to_df,   center = center, scale = scale_attr)
    sdcal <- caliper  # in scaled space, SD = 1
    from_idx_matched <- integer(0)
    to_idx_matched   <- integer(0)
    from_rownames <- rownames(match_from_df)
    to_rownames   <- rownames(match_to_df)

    for (i in seq_len(n_from)) {
      from_row <- match_from_scaled[i, , drop = TRUE]
      dist_sq <- colSums((t(match_to_scaled) - from_row)^2)
      dist <- sqrt(dist_sq)
      ord <- order(dist)
      k <- min(ratio, length(ord))
      any_within <- FALSE
      for (j in seq_len(k)) {
        idx <- ord[j]
        if ((dist[idx] / sqrt(p)) < sdcal) {
          any_within <- TRUE
          to_idx_matched <- c(to_idx_matched, as.integer(to_rownames[idx]))
        }
      }
      if (any_within) from_idx_matched <- c(from_idx_matched, as.integer(from_rownames[i]))
    }
    from_idx_matched <- unique(from_idx_matched)
    keep_idx <- c(from_idx_matched, to_idx_matched)
  } else {
    if (length(score_cols) != 1) {
      stop("Matching on multiple columns is only supported with replace=TRUE. Set replace=TRUE for multi-column matching.")
    }
    score_col <- score_cols[1]
    sdcal <- caliper * stats::sd(data[[score_col]], na.rm = TRUE)
    match_from_vec <- match_from_df[[1]]
    match_to_vec   <- match_to_df[[1]]
    names(match_from_vec) <- rownames(match_from_df)
    names(match_to_vec)   <- rownames(match_to_df)

    if (shuffle) {
      from_indices <- sample(seq_along(match_from_vec))
    } else {
      from_indices <- seq_along(match_from_vec)
    }
    from_idx_matched <- integer(0)
    to_idx_matched   <- integer(0)
    unmatched <- rep(TRUE, n_to)
    to_rownames <- rownames(match_to_df)

    which_unmatched_idx <- function() which(unmatched)

    for (ii in from_indices) {
      from_idx <- as.integer(names(match_from_vec)[ii])
      from_val <- match_from_vec[ii]
      wu <- which_unmatched_idx()
      if (length(wu) == 0) next
      dist <- abs(match_to_vec[wu] - from_val)
      ord <- order(dist)
      k <- min(ratio, length(ord))
      for (j in seq_len(k)) {
        pos_in_unmatched <- ord[j]
        to_global <- wu[pos_in_unmatched]
        d <- dist[pos_in_unmatched]
        if (d <= sdcal) {
          if (j == 1) from_idx_matched <- c(from_idx_matched, from_idx)
          to_idx_matched <- c(to_idx_matched, as.integer(to_rownames[to_global]))
          unmatched[to_global] <- FALSE
        }
      }
    }
    keep_idx <- c(from_idx_matched, to_idx_matched)
  }

  data_matched <- data[sort(unique(keep_idx)), , drop = FALSE]
  attr(data_matched, "matched_control_idx") <- if (treatment_to_control) to_idx_matched else from_idx_matched
  attr(data_matched, "treated_idx") <- if (treatment_to_control) from_idx_matched else to_idx_matched
  data_matched
}


#' Stratified nearest neighbor matching by group
#'
#' Runs nearest neighbor matching within each level of \code{groupby_col},
#' then combines the matched samples.
#'
#' @param data data.frame with treatment, score, and group columns
#' @param treatment_col name of treatment column
#' @param score_cols character vector; columns to match on
#' @param groupby_col name of column defining strata
#' @param ... arguments passed to \code{\link{nearest_neighbor_match}} (e.g. caliper, replace, ratio)
#' @return data.frame of matched units from all groups
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # match_by_group(...)
#' }
#' @export
match_by_group <- function(data,
                          treatment_col = "w",
                          score_cols = "e",
                          groupby_col,
                          ...) {
  grps <- split(data, data[[groupby_col]])
  out_list <- lapply(grps, function(g) {
    nearest_neighbor_match(g, treatment_col = treatment_col, score_cols = score_cols, ...)
  })
  do.call(rbind, out_list)
}


#' Match optimizer: search for best caliper / pihat / score_cols
#'
#' Finds parameters that minimize a balance score: number of features with
#' SMD above \code{max_smd} plus deviation penalty for key variables.
#' Searches over max propensity threshold, which covariates to include in
#' matching, and caliper.
#'
#' @param df data.frame with treatment, propensity, and covariate columns
#' @param treatment_col name of treatment column (default "is_treatment")
#' @param ps_col name of propensity score column (default "pihat")
#' @param user_col optional column for counting unique users per group (default NULL)
#' @param matching_covariates character vector; covariates to assess balance (default "pihat")
#' @param max_smd maximum acceptable absolute SMD (default 0.1)
#' @param max_deviation maximum acceptable deviation for key variables (default 0.1)
#' @param caliper_range numeric vector or length-2 range for caliper search (default c(0.01, 0.5))
#' @param max_pihat_range numeric vector or length-2 range for max propensity filter (default c(0.95, 0.999))
#' @param max_iter_per_param number of values to try per parameter (default 5)
#' @param min_users_per_group minimum units per group in matched set (default 1000)
#' @param smd_cols character vector; covariates that contribute extra to score when SMD > max_smd (default "pihat")
#' @param dev_cols_transformations named list of functions (e.g. mean) for deviation (default pihat = mean)
#' @param dev_factor weight for deviation in score (default 1)
#' @param verbose logical; print progress (default TRUE)
#' @return data.frame; best matched sample
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # match_optimizer(...)
#' }
#' @export
match_optimizer <- function(df,
                            treatment_col = "is_treatment",
                            ps_col = "pihat",
                            user_col = NULL,
                            matching_covariates = "pihat",
                            max_smd = 0.1,
                            max_deviation = 0.1,
                            caliper_range = c(0.01, 0.5),
                            max_pihat_range = c(0.95, 0.999),
                            max_iter_per_param = 5L,
                            min_users_per_group = 1000L,
                            smd_cols = "pihat",
                            dev_cols_transformations = NULL,
                            dev_factor = 1.0,
                            verbose = TRUE) {
  if (is.null(dev_cols_transformations)) {
    dev_cols_transformations <- setNames(list(mean), ps_col)
  }
  if (length(caliper_range) == 2) {
    caliper_range <- seq(caliper_range[1], caliper_range[2], length.out = max_iter_per_param)
  }
  if (length(max_pihat_range) == 2) {
    max_pihat_range <- seq(max_pihat_range[1], max_pihat_range[2], length.out = max_iter_per_param)
  }
  trt <- df[[treatment_col]]
  if (is.character(trt) || is.factor(trt)) trt <- as.integer(trt != levels(trt)[1])
  treated_df <- df[trt == 1, , drop = FALSE]
  original_stats <- list()
  for (col in names(dev_cols_transformations)) {
    original_stats[[col]] <- dev_cols_transformations[[col]](treated_df[[col]])
  }

  best_score <- 1e7
  best_params <- list()
  best_matched <- NULL
  pass_all <- FALSE
  cols_to_fix <- character(0)

  single_match <- function(score_cols, pihat_threshold, caliper) {
    sub <- df[df[[ps_col]] < pihat_threshold, , drop = FALSE]
    nearest_neighbor_match(sub,
                           treatment_col = treatment_col,
                           score_cols = score_cols,
                           caliper = caliper,
                           replace = TRUE)
  }

  check_table_one <- function(tableone, matched, score_cols, pihat_threshold, caliper) {
    smd_rows <- tableone[rownames(tableone) != "n", , drop = FALSE]
    smd_vals <- abs(as.numeric(smd_rows$SMD))
    names(smd_vals) <- smd_rows$Variable
    num_cols_over_smd <- sum(smd_vals >= max_smd, na.rm = TRUE)
    cols_over <- smd_vals[smd_vals >= max_smd]
    cols_to_fix <<- names(sort(cols_over, decreasing = TRUE))

    if (is.null(user_col)) {
      n_per_group <- min(
        sum(matched[[treatment_col]] == 0),
        sum(matched[[treatment_col]] == 1)
      )
    } else {
      n_per_group <- min(
        sum(matched[matched[[treatment_col]] == 0, user_col], na.rm = TRUE),
        sum(matched[matched[[treatment_col]] == 1, user_col], na.rm = TRUE)
      )
    }
    matched_treated <- matched[matched[[treatment_col]] == 1, , drop = FALSE]
    deviations <- numeric(length(dev_cols_transformations))
    for (i in seq_along(dev_cols_transformations)) {
      col <- names(dev_cols_transformations)[i]
      deviations[i] <- abs(original_stats[[col]] / mean(matched_treated[[col]], na.rm = TRUE) - 1)
    }
    score <- num_cols_over_smd
    for (col in smd_cols) {
      if (col %in% names(smd_vals) && !is.na(smd_vals[col]) && smd_vals[col] >= max_smd) {
        score <- score + 1
      }
    }
    score <- score + sum(deviations * 10 * dev_factor)

    if (score < best_score && n_per_group > min_users_per_group) {
      best_score   <<- score
      best_params  <<- list(score_cols = score_cols, pihat = pihat_threshold, caliper = caliper)
      best_matched <<- matched
    }
    if (verbose) {
      message(sprintf("\tScore: %.3f (Best: %.3f)\n", score, best_score))
    }
    pass_all <<- (n_per_group > min_users_per_group &&
                    num_cols_over_smd == 0 &&
                    all(deviations < max_deviation))
    invisible(NULL)
  }

  match_and_check <- function(score_cols, pihat_threshold, caliper) {
    if (verbose) {
      message(sprintf("Preparing match: caliper=%.3f, pihat_threshold=%.3f, score_cols=%s",
                      caliper, pihat_threshold, paste(score_cols, collapse = ",")))
    }
    df_matched <- single_match(score_cols, pihat_threshold, caliper)
    tableone <- create_table_one(df_matched, treatment_col, matching_covariates)
    check_table_one(tableone, df_matched, score_cols, pihat_threshold, caliper)
  }

  # Search best max pihat
  if (verbose) message("SEARCHING FOR BEST PIHAT")
  score_cols <- ps_col
  caliper <- caliper_range[length(caliper_range)]
  for (pihat_threshold in max_pihat_range) {
    match_and_check(score_cols, pihat_threshold, caliper)
  }

  # Search best score_cols (add covariates if SMD still high)
  if (verbose) message("SEARCHING FOR BEST SCORE_COLS")
  if (length(best_params) > 0) {
    pihat_threshold <- best_params$pihat
    caliper <- caliper_range[max(1, length(caliper_range) %/% 2)]
    score_cols <- c(ps_col)
    while (!pass_all) {
      if (length(cols_to_fix) == 0) break
      if (length(intersect(cols_to_fix, score_cols)) > 0) break
      score_cols <- c(score_cols, cols_to_fix[1])
      match_and_check(score_cols, pihat_threshold, caliper)
    }

    # Search best caliper
    if (verbose) message("SEARCHING FOR BEST CALIPER")
    score_cols <- best_params$score_cols
    pihat_threshold <- best_params$pihat
    for (cal in caliper_range) {
      match_and_check(score_cols, pihat_threshold, cal)
    }
  }

  if (verbose && length(best_params) > 0) {
    message("\n-----\nBest params:\n", paste(capture.output(print(best_params)), collapse = "\n"))
  }
  if (is.null(best_matched)) {
    stop("No matching run met min_users_per_group. Try relaxing constraints or increasing data.")
  }
  best_matched
}
