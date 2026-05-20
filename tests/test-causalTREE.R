# Test: causalTREE.R with lung-like data (X, Y, D)
# Run from package root: Rscript tests/test-causalTREE.R
# Or: source("tests/test-causalTREE.R")
#
# Uses lung data from {survival}: covariates (X), outcome (Y), censoring (D),
# and a simulated binary treatment (W) for causal tree fitting.

pkg_root <- if (file.exists("R/causalTree.R")) "." else
  if (file.exists("../R/causalTree.R")) ".." else stop("Run from RCausalML package root")

# test-causalTREE.R uses legacy causalTree_CV / causalTree_prune helpers that
# are not part of the current package API.  Skip gracefully.
if (!exists("causalTree_CV", mode = "function")) {
  message("SKIP: Legacy causalTree_CV/causalTree_prune helpers not available in this version.")
  message("Use causal_tree() + rpart::prune() from test-causalTree.R instead.")
  quit(status = 0, save = "no")
}

# Load dependencies needed when sourcing directly
library(rpart)
if (requireNamespace("htetree", quietly = TRUE)) library(htetree)
# Load causal_tree function
source(file.path(pkg_root, "R/causalTree.R"))

# Check dependencies and load survival + rpart so data(lung) and htetree work
if (!requireNamespace("survival", quietly = TRUE)) {
  message("Package 'survival' not found. Install with: install.packages('survival')")
  quit(save = "no", status = 1)
}
suppressPackageStartupMessages({
  library("survival", character.only = TRUE)
  if (requireNamespace("rpart", quietly = TRUE)) library("rpart", character.only = TRUE)
})
if (!requireNamespace("htetree", quietly = TRUE)) {
  message("Package 'htetree' required for causalTree(). Install from GitHub or skip test.")
  message("Skipping causalTree fit; testing only helpers that work on rpart objects.")
  has_htetree <- FALSE
} else {
  has_htetree <- TRUE
}

set.seed(42)

# ---- Lung-like data: X (covariates), Y (outcome), D (censoring) ----
message("========== Lung-like data (X, Y, D) ==========")
lung_clean <- tryCatch({
  data("lung", package = "survival", envir = environment())
  na.omit(lung)
}, error = function(e) NULL)
if (is.null(lung_clean)) {
  # Fallback: simulated lung-like data (X, Y, D)
  n_sim <- 200
  X_sim <- matrix(rnorm(n_sim * 5), n_sim, 5)
  colnames(X_sim) <- c("age", "sex", "ph.ecog", "ph.karno", "pat.karno")
  X_sim[, "age"] <- 50 + 15 * scale(X_sim[, "age"])
  Y_sim <- 300 + 50 * X_sim[, 1] + rnorm(n_sim, 0, 80)
  D_sim <- rbinom(n_sim, 1, 0.7)
  lung_clean <- as.data.frame(X_sim)
  lung_clean$time <- Y_sim
  lung_clean$status <- ifelse(D_sim == 1, 2, 1)
  lung_clean <- lung_clean[, c(colnames(X_sim), "time", "status")]
  message("Using simulated lung-like data (n = ", n_sim, ")")
}

# X: covariates (use available columns)
cn <- setdiff(colnames(lung_clean), c("time", "status"))
if (length(cn) == 0) cn <- c("age", "sex", "ph.ecog", "ph.karno", "pat.karno")
X <- as.matrix(lung_clean[, cn, drop = FALSE])
colnames(X) <- cn
X <- X[, apply(X, 2, var, na.rm = TRUE) > 1e-10, drop = FALSE]

# Y: outcome (survival time in days)
Y <- if ("time" %in% colnames(lung_clean)) lung_clean$time else lung_clean[[ncol(lung_clean) - 1]]

# D: censoring indicator (1 = event, 0 = censored)
D <- if ("status" %in% colnames(lung_clean)) ifelse(lung_clean$status == 2, 1, 0) else rbinom(nrow(lung_clean), 1, 0.7)

# W: simulated binary treatment (e.g. drug vs placebo)
W <- rbinom(nrow(lung_clean), 1, 0.5 + 0.2 * (X[, "age"] > 60))

n <- nrow(lung_clean)
message("n = ", n, ", p = ", ncol(X))
message("Outcome Y (time): mean = ", round(mean(Y), 1), ", range = [", min(Y), ", ", max(Y), "]")
message("Censoring D: events = ", sum(D), ", censored = ", sum(D == 0))
message("Treatment W: proportion treated = ", round(mean(W), 3))

# Train/test split
train_prop <- 0.8
train_idx <- sample(1:n, size = round(train_prop * n))
X_train <- X[train_idx, , drop = FALSE]
Y_train <- Y[train_idx]
D_train <- D[train_idx]
W_train <- W[train_idx]

# Data frame for causal tree: outcome, treatment, covariates
df <- as.data.frame(X_train)
df$outcome <- Y_train
df$treatment <- W_train

message("")
message("---- Fitting causal tree (htetree) ----")

if (has_htetree) {
  covar_names <- setdiff(colnames(df), c("outcome", "treatment"))
  formula_ct <- as.formula(paste("outcome ~", paste(covar_names, collapse = " + ")))

  ct_model <- causalTree(
    formula = formula_ct,
    data = df,
    treatment = df$treatment,
    split.Rule = "CT",
    cv.option = "CT",
    split.Honest = TRUE,
    cv.Honest = TRUE,
    split.Bucket = FALSE,
    minsize = 10,
    cp = 0,
    maxdepth = 5,
    xval = 5
  )

  message("causalTree fitted: n splits = ", sum(ct_model$frame$var != "<leaf>"))

  # Prune using CV
  best_cp <- causalTree_CV(ct_model)
  message("Best cp from CV: ", round(best_cp, 4))
  ct_pruned <- causalTree_prune(ct_model, cp = max(best_cp, 0.005))
  message("Pruned tree: leaves = ", sum(ct_pruned$frame$var == "<leaf>"))

  # Predict (CATE) — use training data so model.frame finds formula variables
  pred_train <- causalTree_predict(ct_pruned, newdata = df)
  message("Predict (train): length = ", length(pred_train), ", mean CATE = ", round(mean(pred_train, na.rm = TRUE), 2))

  # ATE, ITE, HTE (pass df so predict can build model frame)
  ate <- causalTree_ATE(ct_pruned, newdata = df)
  message("ATE = ", round(ate, 2))
  ite_train <- causalTree_ITE(ct_pruned, newdata = df)
  message("ITE (train): mean = ", round(mean(ite_train, na.rm = TRUE), 2))
  hte_out <- causalTree_HTE(ct_pruned, data = df)
  message("HTE: ", nrow(hte_out$leaf_effects), " leaves")

  # Leaf rules
  leaf_rules <- extract_leaf_rules(ct_pruned, df, outcome_name = "outcome", treatment_name = "treatment")
  message("extract_leaf_rules: ", length(leaf_rules), " leaves")

  # Variable importance
  vip <- causalTree_VIP(ct_pruned)
  message("causalTree_VIP: ", length(vip), " variables")
  if (length(vip) > 0) message("  Top: ", paste(head(names(vip), 3), collapse = ", "))

  # Diagnostics
  diag <- Causal_Diagnostics(ct_pruned, df, outcome_name = "outcome", treatment_name = "treatment",
                             covar_names = covar_names)
  message("Causal_Diagnostics: leaf_summary nrow = ", nrow(diag$leaf_summary))
  message("  CATE summary: mean = ", round(diag$cate_summary$mean, 2), ", sd = ", round(diag$cate_summary$sd, 2))

  # Control and anova (smoke test)
  ctrl <- causalTree_control(minsplit = 20, cp = 0, xval = 5)
  message("causalTree_control: minsplit = ", ctrl$minsplit, ", xval = ", ctrl$xval)
  anova_init <- causalTree_anova(Y_train, offset = NULL, wt = rep(1, length(Y_train)))
  message("causalTree_anova: numresp = ", anova_init$numresp)

  # Optional: plot (only if interactive or env allows)
  if (interactive() && nrow(ct_pruned$frame) > 1) {
    try({
      causalTree_plot(ct_pruned, main = "Causal Tree (lung-like data)")
      message("Plot drawn (close device if needed).")
    }, silent = TRUE)
  }

  message("")
  message("========== causalTREE tests completed ==========")
} else {
  # Without htetree: test only helpers on a minimal rpart-like object
  message("Building minimal rpart-like object for helper tests (no htetree).")
  fake_frame <- data.frame(
    var = c("age", "<leaf>", "<leaf>"),
    n = c(200L, 80L, 120L),
    wt = c(200, 80, 120),
    dev = c(1e6, 4e5, 6e5),
    yval = c(0, -20, 30),
    complexity = c(0.01, 0.01, 0.01),
    ncompete = c(0L, 0L, 0L),
    nsurrogate = c(0L, 0L, 0L),
    row.names = c("1", "2", "3")
  )
  fake_where <- rep(c(2L, 3L), length.out = nrow(df))
  names(fake_where) <- rownames(df)
  fake_tree <- list(
    frame = fake_frame,
    where = fake_where,
    cptable = matrix(c(0.1, 0.01, 1, 2, 0.5, 1.2, 0.3), nrow = 5,
                     dimnames = list(c("CP", "nsplit", "rel error", "xerror", "xstd"), NULL)),
    splits = NULL,
    variable.importance = c(age = 100),
    terms = terms(outcome ~ age + sex, data = df),
    method = "anova"
  )
  class(fake_tree) <- "rpart"

  message("causalTree_CV: ", causalTree_CV(fake_tree))
  hte_fake <- causalTree_HTE(fake_tree)
  message("causalTree_HTE: ", nrow(hte_fake$leaf_effects), " leaves")
  leaf_rules_fake <- extract_leaf_rules(fake_tree, df, "outcome", "treatment")
  message("extract_leaf_rules: ", length(leaf_rules_fake), " leaves")
  message("causalTree_VIP: ", paste(names(causalTree_VIP(fake_tree)), collapse = ", "))
  diag_fake <- Causal_Diagnostics(fake_tree, df, "outcome", "treatment", covar_names = "age")
  message("Causal_Diagnostics: OK")
  # ATE/ITE need predict(); fake tree has no splits so skip here
  message("(ATE/ITE skipped without htetree - need full rpart object)")
  message("")
  message("========== causalTREE helper-only tests completed ==========")
}
