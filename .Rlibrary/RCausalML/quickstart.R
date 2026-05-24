# CausalML-R Quickstart (mirrors Python CausalML quickstart)
# https://causalml.readthedocs.io/en/latest/quickstart.html
#
# Meta-learners (S, T, X, R), fit(), estimate_ate(), predict(), and convenience
# wrappers (LRSRegressor, etc.) are defined in R/meta_learners.R.

library(RCausalML)

# ---- Synthetic data ----
set.seed(42)
d <- synthetic_data(mode = 1, n = 1000, p = 5, sigma = 1)
y <- d$y
X <- d$X
w <- d$w          # treatment (0/1)
e <- d$e          # propensity scores (for X-Learner, R-Learner, TMLE)

# ---- S-Learner (Linear Regression): LRSRegressor = SLearner(learner = "lm") ----
lr <- LRSRegressor()
lr <- fit(lr, X, w, y)
ate_lr <- estimate_ate(lr, X, w, y, return_ci = TRUE)
cat(sprintf("ATE (S-Learner / LRSRegressor): %.2f (%.2f, %.2f)\n",
    as.numeric(ate_lr$ate), ate_lr$ate_lb, ate_lr$ate_ub))

# ---- S-Learner with ranger (explicit SLearner) ----
sl <- SLearner(learner = "ranger")
sl <- fit(sl, X, w, y)
ate_sl <- estimate_ate(sl, X, w, y, return_ci = TRUE)
cat(sprintf("ATE (S-Learner ranger): %.2f (%.2f, %.2f)\n",
    as.numeric(ate_sl$ate), ate_sl$ate_lb, ate_sl$ate_ub))

# ---- T-Learner (separate models for control/treatment; e.g. ranger) ----
tl <- TLearner(learner = "ranger")
tl <- fit(tl, X, w, y)
ate_tl <- estimate_ate(tl, X, w, y, return_ci = TRUE)
cat(sprintf("ATE (T-Learner ranger): %.2f (%.2f, %.2f)\n",
    as.numeric(ate_tl$ate), ate_tl$ate_lb, ate_tl$ate_ub))

# ---- X-Learner (uses propensity when provided) ----
xl <- XLearner(learner = "ranger")
xl <- fit(xl, X, w, y, p = e)
ate_xl <- estimate_ate(xl, X, w, y, p = e, return_ci = TRUE)
cat(sprintf("ATE (X-Learner ranger): %.2f (%.2f, %.2f)\n",
    as.numeric(ate_xl$ate), ate_xl$ate_lb, ate_xl$ate_ub))

# ---- R-Learner (cross-fitting; propensity optional, else glmnet) ----
rl <- RLearner(learner = "ranger", n_fold = 5)
rl <- fit(rl, X, w, y, p = e)
ate_rl <- estimate_ate(rl, X, w, y, p = e, return_ci = TRUE)
cat(sprintf("ATE (R-Learner ranger): %.2f (%.2f, %.2f)\n",
    as.numeric(ate_rl$ate), ate_rl$ate_lb, ate_rl$ate_ub))

# ---- TMLE (Targeted Maximum Likelihood; from meta_learners.R) ----
tmle <- TMLELearner(learner = "ranger")
tmle <- fit(tmle, X, w, y, p = e)
ate_tmle <- estimate_ate(tmle, X, w, y, p = e, return_ci = TRUE)
cat(sprintf("ATE (TMLE ranger): %.2f (%.2f, %.2f)\n",
    as.numeric(ate_tmle$ate), ate_tmle$ate_lb, ate_tmle$ate_ub))

# ---- Propensity score (glmnet cross-fit) ----
ps <- propensity_glmnet(X, w, n_fold = 5)
cat("Propensity scores (first 5):", head(ps, 5), "\n")

# ---- CATE predictions (predict returns CATE per unit) ----
cate_s <- predict(lr, X, verbose = FALSE)
cate_t <- predict(tl, X)
cat("Mean CATE (S-Learner):", mean(cate_s), "  Mean CATE (T-Learner):", mean(cate_t), "\n")

# ---- Optional: one-step fit_predict for CATE (with optional CI) ----
# te_sl <- fit_predict(sl, X, w, y, return_ci = FALSE)
# cat("Mean CATE from fit_predict:", mean(te_sl), "\n")
