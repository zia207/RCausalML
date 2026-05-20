# Test R/meta_learners.R (S/T/X/R meta-learners, TMLE, DomainAdaptationLearner, estimate_ate, fit_predict)
# Run from package root: Rscript tests/test-meta_learners.R

pkg_root <- if (file.exists("R/meta_learners.R")) "." else if (file.exists("../R/meta_learners.R")) ".." else stop("Run from RCausalML package root")
if (file.exists(file.path(pkg_root, "DESCRIPTION")) && requireNamespace("devtools", quietly = TRUE)) {
  suppressPackageStartupMessages(devtools::load_all(pkg_root, quiet = TRUE))
} else if (requireNamespace("RCausalML", quietly = TRUE)) {
  library(RCausalML)
} else {
  source(file.path(pkg_root, "R/utils.R"))
  source(file.path(pkg_root, "R/propensity.R"))
  source(file.path(pkg_root, "R/meta_learners.R"))
}

set.seed(2026)
n <- 500L
p <- 4L
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("x", seq_len(p))
e <- plogis(0.4 * X[, 1L])
w <- rbinom(n, 1L, e)
y <- 0.3 * X[, 2L] + w * (0.8 + 0.2 * X[, 1L]) + rnorm(n)

message("========== test-meta_learners.R ==========")

message("---- SLearner (lm) ----")
sl <- SLearner(learner = "lm", control_name = 0)
sl <- fit(sl, X, w, y)
ps <- predict(sl, X)
stopifnot(length(ps) == n)
esa <- estimate_ate(sl, X, w, y, pretrain = TRUE)
stopifnot(is.numeric(esa$ate), length(esa$ate) == 1L)
message("  OK")

message("---- TLearner (lm) ----")
tl <- TLearner(learner = "lm", control_name = 0)
tl <- fit(tl, X, w, y)
pt <- predict(tl, X)
stopifnot(length(pt) == n)
message("  OK")

message("---- XLearner (lm) ----")
xl <- XLearner(learner = "lm", control_name = 0)
xl <- fit(xl, X, w, y)
px <- predict(xl, X)
stopifnot(length(px) == n)
message("  OK")

message("---- RLearner (lm, small n_fold) ----")
rl <- RLearner(learner = "lm", control_name = 0, n_fold = 3L)
rl <- fit(rl, X, w, y, verbose = FALSE)
pr <- predict(rl, X)
stopifnot(length(pr) == n)
message("  OK")

message("---- DomainAdaptationLearner (lm / glm propensity) ----")
dal <- DomainAdaptationLearner(learner = "lm", final_learner = "lm", propensity_learner = "glm")
dal <- fit(dal, X, w, y)
pd <- predict(dal, X)
stopifnot(length(pd) == n)
dal2 <- DomainAdaptationLearner(learner = "lm", propensity_learner = "glmnet")
dal2 <- fit(dal2, X, w, y)
pd2 <- predict(dal2, X[1:10, , drop = FALSE])
stopifnot(length(pd2) == 10L)
fp <- fit_predict(dal, X, w, y, return_ci = FALSE)
stopifnot(length(fp) == n)
ead <- estimate_ate(dal, X, w, y, pretrain = TRUE, return_ci = TRUE, bootstrap_ci = FALSE)
stopifnot(length(ead$ate) == 1L, is.numeric(ead$ate_lb))
message("  OK")

message("---- DomainAdaptationLearner multi-arm ----")
tr3 <- sample(0:2, n, replace = TRUE, prob = c(0.45, 0.35, 0.2))
y3 <- y + 0.15 * as.numeric(tr3 == 2L)
dm <- DomainAdaptationLearner(learner = "lm", final_learner = "lm")
dm <- fit(dm, X, tr3, y3)
pm <- predict(dm, X)
stopifnot(is.matrix(pm), ncol(pm) == 2L)
eam <- estimate_ate(dm, X, tr3, y3, pretrain = TRUE)
stopifnot(length(eam$ate) == 2L)
message("  OK")

message("---- fit_predict.TLearner return_ci=FALSE ----")
ftp <- fit_predict(tl, X, w, y, return_ci = FALSE)
stopifnot(length(ftp) == n)
message("  OK")

message("========== all meta_learners tests passed ==========")
