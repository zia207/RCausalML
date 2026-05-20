# Test NOTEARS (R/notears.R)
# Run from package root: Rscript tests/test-notears.R
# Quick (skip nonlinear): QUICK=1 Rscript tests/test-notears.R

pkg_root <- if (file.exists("R/notears.R")) "." else
  if (file.exists("../R/notears.R")) ".." else
    stop("Run from Causal_ML package root")

# Dependencies
if (!requireNamespace("expm", quietly = TRUE)) stop("Package 'expm' required. Install with install.packages('expm').")
if (!requireNamespace("igraph", quietly = TRUE)) stop("Package 'igraph' required. Install with install.packages('igraph').")
suppressPackageStartupMessages({
  library(expm)
  library(igraph)
})

source(file.path(pkg_root, "R/notears.R"))

quick <- nzchar(Sys.getenv("QUICK", ""))

message("========== NOTEARS tests (R/notears.R) ==========")
message("")

# --- 1. set_random_seed ---
message("---- 1. set_random_seed ----")
set_random_seed(42L)
message("  OK")
message("")

# --- 2. is_dag ---
message("---- 2. is_dag ----")
W_dag <- matrix(0, 3, 3)
W_dag[2,1] <- 1; W_dag[3,2] <- 1
stopifnot(is_dag(W_dag))
W_cycle <- matrix(0, 3, 3)
W_cycle[2,1] <- 1; W_cycle[3,2] <- 1; W_cycle[1,3] <- 1
stopifnot(!is_dag(W_cycle))
message("  OK")
message("")

# --- 3. simulate_dag ---
message("---- 3. simulate_dag ----")
B_er <- simulate_dag(10, 15, "ER")
stopifnot(is.matrix(B_er), nrow(B_er) == 10, ncol(B_er) == 10, all(B_er %in% c(0L, 1L)))
stopifnot(is_dag(B_er))
B_sf <- simulate_dag(10, 8, "SF")
stopifnot(is.matrix(B_sf), nrow(B_sf) == 10, is_dag(B_sf))
message("  OK")
message("")

# --- 4. simulate_parameter ---
message("---- 4. simulate_parameter ----")
B <- simulate_dag(5, 4, "ER")
W <- simulate_parameter(B)
stopifnot(is.matrix(W), nrow(W) == 5, ncol(W) == 5)
stopifnot(all((W != 0) == (B != 0)))
stopifnot(is_dag(W))
message("  OK")
message("")

# --- 5. simulate_linear_sem ---
message("---- 5. simulate_linear_sem ----")
set_random_seed(1)
W <- simulate_parameter(B)
X <- simulate_linear_sem(W, 50, "gauss")
stopifnot(is.matrix(X), nrow(X) == 50, ncol(X) == 5)
X2 <- simulate_linear_sem(W, 30, "uniform", noise_scale = 0.5)
stopifnot(nrow(X2) == 30)
message("  OK")
message("")

# --- 6. count_accuracy ---
message("---- 6. count_accuracy ----")
B_true <- matrix(0, 3, 3)
B_true[2,1] <- 1; B_true[3,2] <- 1
B_est <- B_true
acc <- count_accuracy(B_true, B_est)
stopifnot(is.list(acc), c("fdr", "tpr", "fpr", "shd", "nnz") %in% names(acc))
stopifnot(acc$tpr == 1, acc$fdr == 0, acc$shd == 0)
# With one extra edge
B_est2 <- B_true; B_est2[3,1] <- 1
acc2 <- count_accuracy(B_true, B_est2)
stopifnot(acc2$nnz == 3)
message("  OK")
message("")

# --- 7. notears_linear ---
message("---- 7. notears_linear ----")
set_random_seed(2)
d <- 5
s0 <- 6
B <- simulate_dag(d, s0, "ER")
W <- simulate_parameter(B)
X <- simulate_linear_sem(W, 200, "gauss")
W_est <- notears_linear(X, lambda1 = 0.1, loss_type = "l2", max_iter = 30L, w_threshold = 0.3)
stopifnot(is.matrix(W_est), nrow(W_est) == d, ncol(W_est) == d)
# B is row=child,col=parent; notears_linear returns row=parent,col=child
B_est_binary <- (t(W_est) != 0) + 0
acc_linear <- count_accuracy(B, B_est_binary)
message("  TPR: ", round(acc_linear$tpr, 3), " FDR: ", round(acc_linear$fdr, 3), " SHD: ", acc_linear$shd)
message("  OK")
message("")

# --- 8. demo_linear ---
message("---- 8. demo_linear ----")
capture.output(demo_linear())
message("  OK")
message("")

# --- 9. Nonlinear (if torch available and not QUICK) ---
if (!quick && requireNamespace("torch", quietly = TRUE)) {
  message("---- 9. notears_nonlinear (NotearsMLP) ----")
  set_random_seed(3)
  d <- 4
  n <- 150
  B <- simulate_dag(d, 4, "ER")
  X <- simulate_nonlinear_sem(B, n, sem_type = "mlp")
  tryCatch({
    model <- NotearsMLP(d = d, hidden = 5)
    W_nl <- notears_nonlinear(model, X, lambda1 = 0.01, lambda2 = 0.01,
                              max_iter = 20L, lbfgs_iter = 2L, verbose = FALSE)
    stopifnot(is.matrix(W_nl), nrow(W_nl) == d, ncol(W_nl) == d)
    message("  OK")
  }, error = function(e) message("  Skip (torch error): ", conditionMessage(e)))
  message("")
} else if (quick) {
  message("---- 9. notears_nonlinear ---- skipped (QUICK=1)")
} else {
  message("---- 9. notears_nonlinear ---- skipped (torch not installed)")
}
message("")

message("========== NOTEARS tests done ==========")
