## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse  = TRUE,
  comment   = "#>",
  fig.width = 7,
  fig.height = 5,
  warning   = FALSE,
  message   = FALSE
)

## ----tlearner, eval = FALSE---------------------------------------------------
# library(RCausalML)
# 
# # Simulate a simple randomised trial
# set.seed(42)
# n   <- 500
# X   <- matrix(rnorm(n * 5), n, 5)
# W   <- rbinom(n, 1, 0.5)
# tau <- 2 + X[, 1]            # heterogeneous treatment effect
# Y   <- tau * W + rnorm(n)
# 
# # Fit the T-Learner
# tl  <- TLearner$new(learner = "ranger")
# tl$fit(X, W, Y)
# 
# # Predict CATE on new data
# X_test <- matrix(rnorm(100 * 5), 100, 5)
# cate   <- tl$predict(X_test)
# head(cate)

## ----dml, eval = FALSE--------------------------------------------------------
# library(RCausalML)
# 
# set.seed(42)
# n    <- 1000
# X    <- matrix(rnorm(n * 10), n, 10)
# W    <- as.numeric(X[, 1] > 0) + rnorm(n, sd = 0.1)
# Y    <- 2 * W + X[, 1] + rnorm(n)
# 
# dml  <- LinearDML$new(
#   model_y      = "ranger",
#   model_t      = "ranger",
#   n_folds      = 5
# )
# dml$fit(X, W, Y)
# 
# ate  <- dml$effect()
# cat("Estimated ATE:", round(ate, 3), "\n")

## ----uplift, eval = FALSE-----------------------------------------------------
# library(RCausalML)
# 
# # Simulate uplift dataset
# df <- make_uplift_classification(
#   n_samples  = 2000,
#   treatment_name = "treatment"
# )
# 
# # KL-based uplift random forest
# uf <- uplift_rf_kl(
#   formula   = ~ . - treatment - outcome,
#   data      = df,
#   treatment = df$treatment,
#   outcome   = df$outcome
# )
# 
# # Gain and Qini curves
# gain_df <- gain_curve(uf, df$treatment, df$outcome)
# plot_gain(gain_df)

## ----grf, eval = FALSE--------------------------------------------------------
# library(RCausalML)
# 
# set.seed(42)
# n  <- 800
# X  <- matrix(rnorm(n * 10), n, 10)
# W  <- rbinom(n, 1, 0.4)
# Y  <- (1 + X[, 1]) * W + rnorm(n)
# 
# cf  <- grf_causal_forest(X, Y, W)
# 
# # Average treatment effect
# grf_average_treatment_effect(cf)
# 
# # Variable importance
# grf_variable_importance(cf)

## ----dragonnet, eval = FALSE--------------------------------------------------
# library(RCausalML)
# 
# set.seed(42)
# n   <- 600
# X   <- matrix(rnorm(n * 8), n, 8)
# W   <- rbinom(n, 1, plogis(X[, 1]))
# Y   <- (0.5 + X[, 2]) * W + rnorm(n)
# 
# net <- dragonnet(X, W, Y,
#   hidden_size = 64,
#   epochs      = 100,
#   batch_size  = 64
# )
# 
# preds <- predict(net, X)
# head(preds)

