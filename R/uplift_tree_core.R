# CausalML-R: Core uplift tree building (KL, ED, Chi, DDP, IDDP, IT, CIT, CTS)
# Aligned with Python causalml.inference.tree.uplift_tree_classifier (UpliftTreeClassifier, UpliftRandomForestClassifier).
# Supports multi-treatment (n_class >= 2); DDP/IDDP/IT/CIT binary treatment only.
# Single tree; forests bag multiple trees.

# --- KL divergence (single distribution pair): sum_k p_k * log(p_k/q_k) ---
# Python: kl_divergence(pk, qk). Returns KL(P||Q) with eps clipping.
kl_divergence_scalar <- function(pk, qk) {
  eps <- 1e-6
  if (abs(qk) < eps) return(0)
  qk <- max(eps, min(1 - eps, qk))
  if (abs(pk) < eps) return(-log(1 - qk))
  if (abs(pk - 1) < eps) return(-log(qk))
  pk * log(pk / qk) + (1 - pk) * log((1 - pk) / (1 - qk))
}

# --- Entropy H(p) = -p*log(p) or H(p,q) = -p*log(q) (Python entropyH) ---
entropy_h <- function(p, q = -1) {
  if (q < 0 && p > 0) return(-p * log(p))
  if (q > 0) return(-p * log(q))
  0
}

# --- Node-level evaluation (single node): summary = list of c(p, n) per treatment, p = P(Y=1|T) ---
# Python: evaluate_KL(nodeSummary) with nodeSummary[i] = c(p, n)
evaluate_KL_node <- function(summary_p) {
  p_c <- summary_p[1]
  d_res <- 0
  for (i in seq_along(summary_p)[-1])
    d_res <- d_res + kl_divergence_scalar(summary_p[i], p_c)
  d_res
}

evaluate_ED_node <- function(summary_p) {
  p_c <- summary_p[1]
  d_res <- 0
  for (i in seq_along(summary_p)[-1])
    d_res <- d_res + 2 * (summary_p[i] - p_c)^2
  d_res
}

evaluate_Chi_node <- function(summary_p) {
  p_c <- summary_p[1]
  eps <- 1e-6
  d_res <- 0
  for (i in seq_along(summary_p)[-1]) {
    diff_sq <- (summary_p[i] - p_c)^2
    d_res <- d_res + diff_sq / max(eps, p_c) + diff_sq / max(eps, 1 - p_c)
  }
  d_res
}

evaluate_DDP_node <- function(summary_p) {
  p_c <- summary_p[1]
  d_res <- 0
  for (i in seq_along(summary_p)[-1])
    d_res <- d_res + (summary_p[i] - p_c)
  d_res
}

evaluate_CTS_node <- function(summary_p) {
  -max(summary_p)  # Python: -max([stat[0] for stat in nodeSummary])
}

# --- Legacy branch-level divergence (kept for compatibility) ---
divergence_kl <- function(pT_left, pC_left, pT_right, pC_right) {
  pT_left <- pmax(1e-10, pmin(1 - 1e-10, pT_left))
  pC_left <- pmax(1e-10, pmin(1 - 1e-10, pC_left))
  pT_right <- pmax(1e-10, pmin(1 - 1e-10, pT_right))
  pC_right <- pmax(1e-10, pmin(1 - 1e-10, pC_right))
  (pT_left * log(pT_left / pC_left) + (1 - pT_left) * log((1 - pT_left) / (1 - pC_left)) +
   pT_right * log(pT_right / pC_right) + (1 - pT_right) * log((1 - pT_right) / (1 - pC_right)))
}

divergence_ed <- function(pT_left, pC_left, pT_right, pC_right) {
  (pT_left - pC_left)^2 + (pT_right - pC_right)^2
}

divergence_chi <- function(pT_left, pC_left, pT_right, pC_right) {
  pC_left <- pmax(1e-10, pC_left)
  pC_right <- pmax(1e-10, pC_right)
  (pT_left - pC_left)^2 / pC_left + (pT_right - pC_right)^2 / pC_right
}

ddp_score <- function(pT_left, pC_left, pT_right, pC_right) {
  abs((pT_left - pC_left) - (pT_right - pC_right))
}

entropy2 <- function(p, q) {
  p <- pmax(1e-10, pmin(1 - 1e-10, p))
  q <- pmax(1e-10, q)
  -p * log2(p) - (1 - p) * log2(1 - p)
}

# --- IDDP: DDP* / I(phi, phi_l, phi_r). DDP* = DDP - |E[Y(1)-Y(0)]| in parent. ---
iddp_score <- function(n_t, n_c, n, n_tL, n_tR, n_cL, n_cR, ddp, ate_parent) {
  ddp_star <- max(0, ddp - abs(ate_parent))
  pt <- n_t / n
  pc <- n_c / n
  H_parent <- entropy2(pt, pc)
  H_t <- entropy2(n_tL / n_t, n_tR / n_t)
  H_c <- entropy2(n_cL / n_c, n_cR / n_c)
  I_val <- H_parent * 2 * (1 + ddp_star) / 3 + pt * H_t + pc * H_c + 0.5
  if (I_val < 1e-10) return(0)
  ddp_star / I_val
}

# --- Normalization factor (Python normI / arr_normI). n_c, n_c_left scalars; n_t, n_t_left vectors of treatment counts. ---
norm_i <- function(n_c, n_c_left, n_t, n_t_left, alpha = 0.9, current_divergence = 0, eval_iddp = FALSE) {
  sum_n_t <- sum(n_t)
  pt_a <- sum(n_t_left) / (sum_n_t + 0.1)
  pc_a <- n_c_left / (n_c + 0.1)
  norm_res <- 0
  if (eval_iddp) {
    norm_res <- norm_res + entropy_h(sum_n_t / (sum_n_t + n_c), n_c / (sum_n_t + n_c)) * current_divergence
    norm_res <- norm_res + (sum_n_t / (sum_n_t + n_c)) * entropy_h(pt_a)
  } else {
    norm_res <- norm_res + alpha * entropy_h(sum_n_t / (sum_n_t + n_c), n_c / (sum_n_t + n_c)) * kl_divergence_scalar(pt_a, pc_a)
    for (i in seq_along(n_t)) {
      pt_a_i <- n_t_left[i] / (n_t[i] + 0.1)
      norm_res <- norm_res + (1 - alpha) * entropy_h(n_t[i] / (n_t[i] + n_c), n_c / (n_t[i] + n_c)) * kl_divergence_scalar(pt_a_i, pc_a)
      norm_res <- norm_res + (n_t[i] / (sum_n_t + n_c)) * entropy_h(pt_a_i)
    }
  }
  norm_res <- norm_res + (n_c / (sum_n_t + n_c)) * entropy_h(pc_a)
  norm_res + 0.5
}

# --- IT: squared t-statistic (Python evaluate_IT / arr_evaluate_IT) ---
it_score <- function(y1_L, s2_1, n1, y0_L, s2_2, n2, y1_R, s2_3, n3, y0_R, s2_4, n4) {
  num <- (y1_L - y0_L) - (y1_R - y0_R)
  w <- (n1 - 1) + (n2 - 1) + (n3 - 1) + (n4 - 1)
  if (w < 1e-6) return(0)
  sigma2 <- ((n1 - 1) * s2_1 + (n2 - 1) * s2_2 + (n3 - 1) * s2_3 + (n4 - 1) * s2_4) / w
  sigma2 <- pmax(1e-10, sigma2)
  denom <- sqrt(sigma2 * (1/n1 + 1/n2 + 1/n3 + 1/n4))
  if (denom < 1e-10) return(0)
  t_val <- num / denom
  t_val^2
}

# --- CIT: likelihood ratio (Python arr_evaluate_CIT; binary treatment) ---
cit_score <- function(cur_summary_p, cur_summary_n, left_summary_p, left_summary_n, right_summary_p, right_summary_n) {
  n_l_t_0 <- left_summary_n[1]
  n_r_t_0 <- right_summary_n[1]
  n_l_t_1 <- left_summary_n[2]
  n_r_t_1 <- right_summary_n[2]
  n_l_t <- n_l_t_1 + n_l_t_0
  n_r_t <- n_r_t_1 + n_r_t_0
  n_t <- n_l_t + n_r_t
  n_t_1 <- n_l_t_1 + n_r_t_1
  n_t_0 <- n_l_t_0 + n_r_t_0
  sse_tau_l <- n_l_t_0 * left_summary_p[1] * (1 - left_summary_p[1]) + n_l_t_1 * left_summary_p[2] * (1 - left_summary_p[2])
  sse_tau_r <- n_r_t_0 * right_summary_p[1] * (1 - right_summary_p[1]) + n_r_t_1 * right_summary_p[2] * (1 - right_summary_p[2])
  sse_tau <- n_t_0 * cur_summary_p[1] * (1 - cur_summary_p[1]) + n_t_1 * cur_summary_p[2] * (1 - cur_summary_p[2])
  i_tau_l <- -(n_l_t / 2) * log(n_l_t * sse_tau_l) + n_l_t_1 * log(max(1, n_l_t_1)) + n_l_t_0 * log(max(1, n_l_t_0))
  i_tau_r <- -(n_r_t / 2) * log(n_r_t * sse_tau_r) + n_r_t_1 * log(max(1, n_r_t_1)) + n_r_t_0 * log(max(1, n_r_t_0))
  i_tau <- -(n_t / 2) * log(n_t * sse_tau) + n_t_1 * log(max(1, n_t_1)) + n_t_0 * log(max(1, n_t_0))
  2 * (i_tau_l + i_tau_r - i_tau)
}

# --- Group counts: for each treatment i, count N(Y=0,T=i) and N(Y=1,T=i). ---
# Returns list of c(n_neg, n_pos) per treatment (n_neg + n_pos = N(T=i)).
group_unique_counts <- function(treatment_idx, y) {
  n_class <- length(unique(treatment_idx))
  res <- vector("list", n_class)
  for (i in seq_len(n_class)) {
    filt <- treatment_idx == (i - 1L)
    n_pos <- sum(y[filt])
    n_neg <- sum(filt) - n_pos
    res[[i]] <- c(n_neg, n_pos)
  }
  res
}

# --- Tree node summary: [[P(Y=1|T), N(T)], ...] with optional parent regularization (Python tree_node_summary). ---
tree_node_summary <- function(treatment_idx, y, min_samples_treatment = 10, n_reg = 100, parent_summary_p = NULL) {
  counts <- group_unique_counts(treatment_idx, y)
  n_class <- length(counts)
  node_summary <- vector("list", n_class)
  for (i in seq_len(n_class)) {
    n_pos <- counts[[i]][2]
    n <- counts[[i]][1] + counts[[i]][2]
    if (is.null(parent_summary_p)) {
      p <- if (n > 0) n_pos / n else 0
    } else {
      if (n > min_samples_treatment)
        p <- (n_pos + parent_summary_p[i] * n_reg) / (n + n_reg)
      else
        p <- parent_summary_p[i]
    }
    node_summary[[i]] <- c(p, n)
  }
  node_summary
}

# --- Summary to vectors (p and n) for easy use ---
summary_to_vectors <- function(node_summary) {
  list(
    p = vapply(node_summary, function(x) as.numeric(x[1]), 0),
    n = vapply(node_summary, function(x) as.integer(x[2]), 0L)
  )
}

# --- Node summary for continuous outcome: (mean(y), var(y), n) per treatment (for IT/CIT with continuous y). ---
tree_node_summary_continuous <- function(treatment_idx, y) {
  n_class <- length(unique(treatment_idx))
  res <- vector("list", n_class)
  for (i in seq_len(n_class)) {
    filt <- treatment_idx == (i - 1L)
    yi <- y[filt]
    ni <- length(yi)
    if (ni > 0) {
      m <- mean(yi)
      v <- if (ni >= 2) var(yi) else 0
      if (is.na(v) || v < 1e-10) v <- 1e-10
    } else {
      m <- 0
      v <- 1e-10
      ni <- 0L
    }
    res[[i]] <- c(mean = m, var = v, n = ni)
  }
  res
}

# --- Summary continuous to vectors (mean, var, n) for easy use ---
summary_continuous_to_vectors <- function(node_summary_cont) {
  list(
    mean = vapply(node_summary_cont, function(x) as.numeric(x["mean"]), 0),
    var = vapply(node_summary_cont, function(x) as.numeric(x["var"]), 0),
    n = vapply(node_summary_cont, function(x) as.integer(x["n"]), 0L)
  )
}

# --- Divide set: left = (X[, col] >= value) for numeric, (X[, col] == value) for categorical (match Python). ---
divide_set <- function(X, treatment_idx, y, column, value) {
  xcol <- X[, column]
  if (is.numeric(xcol))
    filt <- xcol >= value
  else
    filt <- xcol == value
  list(
    X_l = X[filt, , drop = FALSE], X_r = X[!filt, , drop = FALSE],
    w_l = treatment_idx[filt], w_r = treatment_idx[!filt],
    y_l = y[filt], y_r = y[!filt]
  )
}

# --- Percentile split values (Python: c_num_percentiles, c_cat_percentiles) ---
split_candidates <- function(xcol, num_percentiles = c(3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97), cat_percentiles = c(10, 50, 90)) {
  u <- unique(na.omit(xcol))
  if (length(u) < 2) return(numeric(0))
  if (is.numeric(xcol)) {
    if (length(u) > 10)
      vals <- quantile(xcol, num_percentiles / 100, names = FALSE, na.rm = TRUE)
    else
      vals <- quantile(u, cat_percentiles / 100, names = FALSE, na.rm = TRUE)
    unique(vals)
  } else {
    u
  }
}

# --- Evaluate split for uplift (binary y) using node summaries ---
eval_split_binary <- function(y, w, idx_left, idx_right, criterion = c("kl", "ed", "chi", "ddp", "iddp")) {
  criterion <- match.arg(criterion)
  yL <- y[idx_left]; wL <- w[idx_left]
  yR <- y[idx_right]; wR <- w[idx_right]
  n_tL <- sum(wL == 1); n_cL <- sum(wL == 0)
  n_tR <- sum(wR == 1); n_cR <- sum(wR == 0)
  pT_left <- if (n_tL > 0) mean(yL[wL == 1]) else 0.5
  pC_left <- if (n_cL > 0) mean(yL[wL == 0]) else 0.5
  pT_right <- if (n_tR > 0) mean(yR[wR == 1]) else 0.5
  pC_right <- if (n_cR > 0) mean(yR[wR == 0]) else 0.5
  n_t <- n_tL + n_tR
  n_c <- n_cL + n_cR
  n <- length(idx_left) + length(idx_right)
  if (criterion == "kl") return(divergence_kl(pT_left, pC_left, pT_right, pC_right))
  if (criterion == "ed") return(divergence_ed(pT_left, pC_left, pT_right, pC_right))
  if (criterion == "chi") return(divergence_chi(pT_left, pC_left, pT_right, pC_right))
  if (criterion == "ddp") return(ddp_score(pT_left, pC_left, pT_right, pC_right))
  ate_parent <- (sum(y[w == 1]) / max(1, n_t) - sum(y[w == 0]) / max(1, n_c))
  if (criterion == "iddp") return(iddp_score(n_t, n_c, n, n_tL, n_tR, n_cL, n_cR,
    ddp_score(pT_left, pC_left, pT_right, pC_right), ate_parent))
  NA_real_
}

eval_split_it <- function(y, w, idx_left, idx_right) {
  yL <- y[idx_left]; wL <- w[idx_left]
  yR <- y[idx_right]; wR <- w[idx_right]
  n1 <- sum(wL == 1); n2 <- sum(wL == 0); n3 <- sum(wR == 1); n4 <- sum(wR == 0)
  if (min(n1, n2, n3, n4) < 2) return(-Inf)
  y1_L <- mean(yL[wL == 1]); s2_1 <- var(yL[wL == 1]); if (is.na(s2_1)) s2_1 <- 0
  y0_L <- mean(yL[wL == 0]); s2_2 <- var(yL[wL == 0]); if (is.na(s2_2)) s2_2 <- 0
  y1_R <- mean(yR[wR == 1]); s2_3 <- var(yR[wR == 1]); if (is.na(s2_3)) s2_3 <- 0
  y0_R <- mean(yR[wR == 0]); s2_4 <- var(yR[wR == 0]); if (is.na(s2_4)) s2_4 <- 0
  it_score(y1_L, s2_1, n1, y0_L, s2_2, n2, y1_R, s2_3, n3, y0_R, s2_4, n4)
}

eval_split_cit <- function(y, w, idx_left, idx_right) {
  yL <- y[idx_left]; wL <- w[idx_left]
  yR <- y[idx_right]; wR <- w[idx_right]
  n_L <- length(yL); n_L1 <- sum(wL == 1); n_L0 <- sum(wL == 0)
  n_R <- length(yR); n_R1 <- sum(wR == 1); n_R0 <- sum(wR == 0)
  if (n_L1 < 1 || n_L0 < 1 || n_R1 < 1 || n_R0 < 1) return(-Inf)
  ybar_L1 <- mean(yL[wL == 1]); ybar_L0 <- mean(yL[wL == 0])
  ybar_R1 <- mean(yR[wR == 1]); ybar_R0 <- mean(yR[wR == 0])
  sse_L <- sum((yL[wL == 1] - ybar_L1)^2) + sum((yL[wL == 0] - ybar_L0)^2)
  sse_R <- sum((yR[wR == 1] - ybar_R1)^2) + sum((yR[wR == 0] - ybar_R0)^2)
  cur_summary_n <- c(n_L0 + n_L1, n_R0 + n_R1)
  cur_summary_p <- c(mean(c(yL[wL == 0], yR[wR == 0])), mean(c(yL[wL == 1], yR[wR == 1])))
  left_summary_p <- c(ybar_L0, ybar_L1)
  left_summary_n <- c(n_L0, n_L1)
  right_summary_p <- c(ybar_R0, ybar_R1)
  right_summary_n <- c(n_R0, n_R1)
  cit_score(cur_summary_p, cur_summary_n, left_summary_p, left_summary_n, right_summary_p, right_summary_n)
}

eval_split_cts <- function(y, w, idx_left, idx_right) {
  yL <- y[idx_left]; wL <- w[idx_left]
  yR <- y[idx_right]; wR <- w[idx_right]
  y_all <- c(y[idx_left], y[idx_right])
  w_all <- c(w[idx_left], w[idx_right])
  y0_P <- mean(y_all[w_all == 0]); y1_P <- mean(y_all[w_all == 1])
  y0_L <- mean(yL[wL == 0]); y1_L <- mean(yL[wL == 1])
  y0_R <- mean(yR[wR == 0]); y1_R <- mean(yR[wR == 1])
  if (is.nan(y0_L)) y0_L <- y0_P
  if (is.nan(y1_L)) y1_L <- y1_P
  if (is.nan(y0_R)) y0_R <- y0_P
  if (is.nan(y1_R)) y1_R <- y1_P
  n_L <- length(yL)
  n_R <- length(yR)
  n <- n_L + n_R
  pL <- n_L / n
  pR <- n_R / n
  pL * max(y0_L, y1_L) + pR * max(y0_R, y1_R) - max(y0_P, y1_P)
}

# --- Uplift classification results at leaf: P(Y=1|T=i) for each treatment (Python uplift_classification_results). ---
uplift_classification_results <- function(treatment_idx, y) {
  counts <- group_unique_counts(treatment_idx, y)
  res <- numeric(length(counts))
  for (i in seq_along(counts)) {
    n <- counts[[i]][1] + counts[[i]][2]
    res[i] <- if (n > 0) counts[[i]][2] / n else 0
  }
  res
}

# --- Build single uplift tree (Python growDecisionTreeFrom logic). ---
# treatment_idx: 0 = control, 1,2,... = treatments. Same length as y.
# Returns tree: list(leaf, split_var, split_val, left, right, results, node_summary, backup_results, best_treatment, uplift_score, ...)
# When leaf=TRUE: results = P(Y=1|T) per group, tau = best_treatment uplift vs control.
build_uplift_tree <- function(X, y, w, criterion = "kl", min_node_size = 10, min_samples_treatment = 10,
                              max_depth = 5, depth = 0, n_reg = 100, normalization = TRUE,
                              max_features = NULL, random_state = NULL,
                              X_val = NULL, w_val = NULL, y_val = NULL, early_stopping_eval_diff_scale = 1) {
  n <- length(y)
  n_class <- length(unique(w))
  if (n_class < 2) n_class <- 2
  # treatment_idx 0-based for group_unique_counts (we pass w as 0/1/2...)
  treatment_idx <- as.integer(w)
  if (min(treatment_idx) > 0) treatment_idx <- treatment_idx - min(treatment_idx)

  # For IT/CIT with continuous outcome use mean(y) per group and tau = E[Y|T=1]-E[Y|T=0]
  y_continuous <- criterion %in% c("it", "cit") && !all(y %in% c(0L, 1L))

  if (n < 2 * min_node_size || depth >= max_depth) {
    if (y_continuous) {
      idx0 <- treatment_idx == 0
      idx1 <- treatment_idx == 1
      n0 <- sum(idx0)
      n1 <- sum(idx1)
      mean0 <- if (n0 > 0) mean(y[idx0]) else 0
      mean1 <- if (n1 > 0) mean(y[idx1]) else 0
      tau <- mean1 - mean0
      results <- c(mean0, mean1)
      node_summary <- tree_node_summary_continuous(treatment_idx, y)
      return(list(leaf = TRUE, tau = tau, n = n, p0 = mean0, p1 = mean1,
                  results = results, node_summary = node_summary, best_treatment = 2L, uplift_score = c(tau, NA)))
    }
    node_summary <- tree_node_summary(treatment_idx, y, min_samples_treatment = 0, n_reg = 0)
    sv <- summary_to_vectors(node_summary)
    results <- sv$p
    p_c <- sv$p[1]
    best_treatment <- 1L
    max_diff <- -Inf
    for (i in seq_along(sv$p)[-1]) {
      d <- sv$p[i] - p_c
      if (d > max_diff) { max_diff <- d; best_treatment <- i }
    }
    if (max_diff <= -Inf) max_diff <- 0
    uplift_score <- c(max_diff, NA)
    tau <- if (length(results) > 1) results[best_treatment] - results[1] else 0
    return(list(leaf = TRUE, tau = tau, n = n, p0 = results[1], p1 = if (length(results) > 1) results[2] else results[1],
                results = results, node_summary = node_summary, best_treatment = best_treatment, uplift_score = uplift_score))
  }

  p <- ncol(X)
  if (is.null(max_features) || max_features <= 0 || max_features > p) max_features <- p
  if (!is.null(random_state)) set.seed(random_state)
  try_cols <- sample(seq_len(p), min(max_features, p), replace = FALSE)

  best_gain <- -Inf
  best_var <- NA
  best_val <- NA
  best_left <- NULL
  best_right <- NULL

  cur_summary <- tree_node_summary(treatment_idx, y, min_samples_treatment, n_reg, NULL)
  cur_sv <- summary_to_vectors(cur_summary)
  cur_summary_p <- cur_sv$p
  cur_summary_n <- cur_sv$n

  if (criterion %in% c("it", "cit")) current_score <- 0 else current_score <- switch(criterion,
    kl = evaluate_KL_node(cur_summary_p),
    ed = evaluate_ED_node(cur_summary_p),
    chi = evaluate_Chi_node(cur_summary_p),
    ddp = evaluate_DDP_node(cur_summary_p),
    iddp = evaluate_DDP_node(cur_summary_p),
    cts = evaluate_CTS_node(cur_summary_p),
    evaluate_KL_node(cur_summary_p))

  for (j in try_cols) {
    xj <- X[, j]
    vals <- split_candidates(xj)
    for (v in vals) {
      div <- divide_set(X, treatment_idx, y, j, v)
      len_L <- nrow(div$X_l)
      len_R <- n - len_L
      if (len_L < min_node_size || len_R < min_node_size) next

      left_summary <- tree_node_summary(div$w_l, div$y_l, min_samples_treatment, n_reg, cur_summary_p)
      right_summary <- tree_node_summary(div$w_r, div$y_r, min_samples_treatment, n_reg, cur_summary_p)
      left_sv <- summary_to_vectors(left_summary)
      right_sv <- summary_to_vectors(right_summary)
      if (min(left_sv$n) < min_samples_treatment || min(right_sv$n) < min_samples_treatment) next

      if (!is.null(X_val) && !is.null(y_val) && early_stopping_eval_diff_scale < Inf) {
        val_div <- divide_set(X_val, w_val, y_val, j, v)
        val_left_sv <- summary_to_vectors(tree_node_summary(val_div$w_l, val_div$y_l, 0, 0, NULL))
        val_right_sv <- summary_to_vectors(tree_node_summary(val_div$w_r, val_div$y_r, 0, 0, NULL))
        early_stop <- FALSE
        for (k in seq_along(cur_summary_p)) {
          if (abs(val_left_sv$p[k] - left_sv$p[k]) > min(val_left_sv$p[k], left_sv$p[k]) / early_stopping_eval_diff_scale ||
              abs(val_right_sv$p[k] - right_sv$p[k]) > min(val_right_sv$p[k], right_sv$p[k]) / early_stopping_eval_diff_scale) {
            early_stop <- TRUE
            break
          }
        }
        if (early_stop) next
      }

      p_node <- len_L / n
      if (criterion == "cts") {
        left_score <- evaluate_CTS_node(left_sv$p)
        right_score <- evaluate_CTS_node(right_sv$p)
        gain <- current_score - p_node * left_score - (1 - p_node) * right_score
      } else if (criterion == "ddp") {
        left_score <- evaluate_DDP_node(left_sv$p)
        right_score <- evaluate_DDP_node(right_sv$p)
        gain <- abs(left_score - right_score)
      } else if (criterion == "iddp") {
        left_score <- evaluate_DDP_node(left_sv$p)
        right_score <- evaluate_DDP_node(right_sv$p)
        gain <- abs(left_score - right_score) - abs(current_score)
        if (normalization) {
          current_divergence <- 2 * (gain + 1) / 3
          norm_factor <- norm_i(cur_summary_n[1], left_sv$n[1], cur_summary_n[-1], left_sv$n[-1], 0.9, current_divergence, TRUE)
        } else norm_factor <- 1
        gain <- gain / norm_factor
      } else if (criterion == "it") {
        if (n_class != 2) next
        yL <- div$y_l; wL <- div$w_l; yR <- div$y_r; wR <- div$w_r
        n1 <- sum(wL == 1); n2 <- sum(wL == 0); n3 <- sum(wR == 1); n4 <- sum(wR == 0)
        if (min(n1, n2, n3, n4) < 2) next
        if (y_continuous) {
          y1_L <- mean(yL[wL == 1]); s2_1 <- var(yL[wL == 1]); if (is.na(s2_1)) s2_1 <- 1e-10
          y0_L <- mean(yL[wL == 0]); s2_2 <- var(yL[wL == 0]); if (is.na(s2_2)) s2_2 <- 1e-10
          y1_R <- mean(yR[wR == 1]); s2_3 <- var(yR[wR == 1]); if (is.na(s2_3)) s2_3 <- 1e-10
          y0_R <- mean(yR[wR == 0]); s2_4 <- var(yR[wR == 0]); if (is.na(s2_4)) s2_4 <- 1e-10
          gain <- it_score(y1_L, s2_1, n1, y0_L, s2_2, n2, y1_R, s2_3, n3, y0_R, s2_4, n4)
        } else {
          s2_1 <- left_sv$p[2] * (1 - left_sv$p[2]); s2_2 <- left_sv$p[1] * (1 - left_sv$p[1])
          s2_3 <- right_sv$p[2] * (1 - right_sv$p[2]); s2_4 <- right_sv$p[1] * (1 - right_sv$p[1])
          gain <- it_score(left_sv$p[2], s2_1, max(1, left_sv$n[2]), left_sv$p[1], s2_2, max(1, left_sv$n[1]),
                          right_sv$p[2], s2_3, max(1, right_sv$n[2]), right_sv$p[1], s2_4, max(1, right_sv$n[1]))
        }
      } else if (criterion == "cit") {
        if (n_class != 2) next
        if (y_continuous) {
          yL <- div$y_l; wL <- div$w_l; yR <- div$y_r; wR <- div$w_r
          n1 <- sum(wL == 1); n2 <- sum(wL == 0); n3 <- sum(wR == 1); n4 <- sum(wR == 0)
          if (min(n1, n2, n3, n4) < 2) next
          y1_L <- mean(yL[wL == 1]); s2_1 <- var(yL[wL == 1]); if (is.na(s2_1)) s2_1 <- 1e-10
          y0_L <- mean(yL[wL == 0]); s2_2 <- var(yL[wL == 0]); if (is.na(s2_2)) s2_2 <- 1e-10
          y1_R <- mean(yR[wR == 1]); s2_3 <- var(yR[wR == 1]); if (is.na(s2_3)) s2_3 <- 1e-10
          y0_R <- mean(yR[wR == 0]); s2_4 <- var(yR[wR == 0]); if (is.na(s2_4)) s2_4 <- 1e-10
          gain <- it_score(y1_L, s2_1, n1, y0_L, s2_2, n2, y1_R, s2_3, n3, y0_R, s2_4, n4)
        } else {
          gain <- cit_score(cur_summary_p, cur_summary_n, left_sv$p, left_sv$n, right_sv$p, right_sv$n)
        }
      } else {
        left_score <- switch(criterion, kl = evaluate_KL_node(left_sv$p), ed = evaluate_ED_node(left_sv$p), chi = evaluate_Chi_node(left_sv$p), evaluate_KL_node(left_sv$p))
        right_score <- switch(criterion, kl = evaluate_KL_node(right_sv$p), ed = evaluate_ED_node(right_sv$p), chi = evaluate_Chi_node(right_sv$p), evaluate_KL_node(right_sv$p))
        gain <- (p_node * left_score + (1 - p_node) * right_score - current_score)
        if (normalization) {
          norm_factor <- norm_i(cur_summary_n[1], left_sv$n[1], cur_summary_n[-1], left_sv$n[-1], 0.9)
        } else norm_factor <- 1
        gain <- gain / max(norm_factor, 1e-10)
      }

      if (is.finite(gain) && gain > best_gain) {
        best_gain <- gain
        best_var <- j
        best_val <- v
        best_left <- list(X = div$X_l, y = div$y_l, w = div$w_l)
        best_right <- list(X = div$X_r, y = div$y_r, w = div$w_r)
      }
    }
  }

  if (is.na(best_var)) {
    if (y_continuous) {
      idx0 <- treatment_idx == 0
      idx1 <- treatment_idx == 1
      n0 <- sum(idx0)
      n1 <- sum(idx1)
      mean0 <- if (n0 > 0) mean(y[idx0]) else 0
      mean1 <- if (n1 > 0) mean(y[idx1]) else 0
      tau <- mean1 - mean0
      results <- c(mean0, mean1)
      node_summary <- tree_node_summary_continuous(treatment_idx, y)
      return(list(leaf = TRUE, tau = tau, n = n, p0 = mean0, p1 = mean1,
                  results = results, node_summary = node_summary, best_treatment = 2L, uplift_score = c(tau, NA)))
    }
    results <- uplift_classification_results(treatment_idx, y)
    p_c <- results[1]
    best_treatment <- 1L
    max_diff <- -Inf
    for (i in seq_along(results)[-1]) {
      d <- results[i] - p_c
      if (d > max_diff) { max_diff <- d; best_treatment <- i }
    }
    uplift_score <- c(max_diff, NA)
    tau <- if (length(results) > 1) results[best_treatment] - results[1] else 0
    return(list(leaf = TRUE, tau = tau, n = n, p0 = results[1], p1 = if (length(results) > 1) results[2] else results[1],
                results = results, node_summary = cur_summary, best_treatment = best_treatment, uplift_score = uplift_score))
  }

  left_tree <- build_uplift_tree(best_left$X, best_left$y, best_left$w, criterion = criterion,
    min_node_size = min_node_size, min_samples_treatment = min_samples_treatment, max_depth = max_depth, depth = depth + 1,
    n_reg = n_reg, normalization = normalization, max_features = max_features, random_state = NULL,
    X_val = NULL, w_val = NULL, y_val = NULL, early_stopping_eval_diff_scale = Inf)
  right_tree <- build_uplift_tree(best_right$X, best_right$y, best_right$w, criterion = criterion,
    min_node_size = min_node_size, min_samples_treatment = min_samples_treatment, max_depth = max_depth, depth = depth + 1,
    n_reg = n_reg, normalization = normalization, max_features = max_features, random_state = NULL,
    X_val = NULL, w_val = NULL, y_val = NULL, early_stopping_eval_diff_scale = Inf)

  backup_results <- uplift_classification_results(treatment_idx, y)
  p_c <- cur_summary_p[1]
  best_treatment <- 1L
  max_diff_treatment <- 1L
  max_diff_sign <- 0
  max_abs_diff <- 0
  max_diff <- -Inf
  for (i in seq_along(cur_summary_p)[-1]) {
    d <- cur_summary_p[i] - p_c
    if (abs(d) > max_abs_diff) { max_abs_diff <- abs(d); max_diff_treatment <- i; max_diff_sign <- sign(d) }
    if (d > max_diff) { max_diff <- d; best_treatment <- i }
  }
  if (!is.finite(max_diff)) max_diff <- 0
  uplift_score <- c(max_diff, NA)

  list(leaf = FALSE, split_var = best_var, split_val = best_val, left = left_tree, right = right_tree,
       results = NULL, node_summary = cur_summary, backup_results = backup_results,
       best_treatment = best_treatment, max_diff_treatment = max_diff_treatment, max_diff_sign = max_diff_sign,
       uplift_score = uplift_score)
}

# --- Predict from uplift tree (Python classify). Split rule: left = (x >= value) for numeric, (x == value) for categorical. ---
# If full_output=TRUE and tree has results (P(Y=1|T) per treatment), return list(tau=, p0=, p1=, results=).
predict_uplift_tree <- function(tree, X, full_output = FALSE) {
  if (tree$leaf) {
    n <- nrow(X)
    if (full_output && !is.null(tree$results)) {
      return(list(
        tau = rep(tree$tau, n),
        p0 = rep(tree$p0, n),
        p1 = rep(if (!is.null(tree$p1)) tree$p1 else tree$p0, n),
        results = matrix(rep(tree$results, each = n), nrow = n, byrow = TRUE)
      ))
    }
    return(rep(tree$tau, n))
  }
  xcol <- X[, tree$split_var]
  if (is.numeric(xcol))
    go_left <- xcol >= tree$split_val
  else
    go_left <- xcol == tree$split_val
  if (full_output) {
    left_out <- predict_uplift_tree(tree$left, X[go_left, , drop = FALSE], full_output = TRUE)
    right_out <- predict_uplift_tree(tree$right, X[!go_left, , drop = FALSE], full_output = TRUE)
    n <- nrow(X)
    n_left <- sum(go_left)
    out_tau <- out_p0 <- out_p1 <- numeric(n)
    out_tau[go_left] <- left_out$tau
    out_tau[!go_left] <- right_out$tau
    out_p0[go_left] <- left_out$p0
    out_p0[!go_left] <- right_out$p0
    out_p1[go_left] <- left_out$p1
    out_p1[!go_left] <- right_out$p1
    n_res <- if (is.matrix(left_out$results)) ncol(left_out$results) else length(left_out$results)
    if (n_res < 1) n_res <- if (is.matrix(right_out$results)) ncol(right_out$results) else length(right_out$results)
    out_results <- matrix(NA_real_, n, n_res)
    if (n_left > 0 && n_res > 0) {
      lr <- as.matrix(left_out$results)
      if (ncol(lr) != n_res) lr <- t(lr)
      if (nrow(lr) != n_left) lr <- matrix(rep(as.vector(lr), length.out = n_left * n_res), nrow = n_left, ncol = n_res)
      out_results[go_left, seq_len(n_res)] <- lr
    }
    if (n_left < n && n_res > 0) {
      n_right <- n - n_left
      rr <- as.matrix(right_out$results)
      if (ncol(rr) != n_res) rr <- t(rr)
      if (nrow(rr) != n_right) rr <- matrix(rep(as.vector(rr), length.out = n_right * n_res), nrow = n_right, ncol = n_res)
      out_results[!go_left, seq_len(n_res)] <- rr
    }
    return(list(tau = out_tau, p0 = out_p0, p1 = out_p1, results = out_results))
  }
  out <- numeric(nrow(X))
  out[go_left] <- predict_uplift_tree(tree$left, X[go_left, , drop = FALSE])
  out[!go_left] <- predict_uplift_tree(tree$right, X[!go_left, , drop = FALSE])
  out
}
