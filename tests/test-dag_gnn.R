# Test DAG-GNN (R/causalDeepNet.R)
# Run from package root: Rscript tests/test-dag_gnn.R

pkg_root <- if (file.exists("R/causalDeepNet.R")) "." else
  if (file.exists("../R/causalDeepNet.R")) ".." else
    stop("Run from RCausalML package root")

if (!requireNamespace("torch", quietly = TRUE)) {
  message("SKIP: Package 'torch' not installed — skipping DAG-GNN tests.")
  message("Install with: install.packages('torch')")
  quit(status = 0, save = "no")
}

# dag_gnn functions are bundled in causalDeepNet.R; load from installed package
lib_paths <- c(file.path(pkg_root, ".Rlibrary"), .libPaths())
.libPaths(lib_paths)
suppressPackageStartupMessages(library(RCausalML))

message("========== DAG-GNN tests (R/causalDeepNet.R) ==========")
message("")

# --- 1. get_daggnn_device ---
message("---- 1. get_daggnn_device ----")
dev <- get_daggnn_device()
stopifnot(inherits(dev, "torch_device"))
message("  OK")
message("")

# --- 2. preprocess_adj, matrix_poly ---
message("---- 2. preprocess_adj, matrix_poly ----")
torch::torch_manual_seed(42L)
A <- torch::torch_randn(4L, 4L) * 0.1
padj <- preprocess_adj(A)
stopifnot(all(dim(padj) == c(4L, 4L)))
mp <- matrix_poly(A * A, 4L)
stopifnot(all(dim(mp) == c(4L, 4L)))
message("  OK")
message("")

# --- 3. DAGGNN forward ---
message("---- 3. DAGGNN forward ----")
model <- DAGGNN(n_nodes = 5L, hidden_dim = 16L)
X <- torch::torch_randn(c(20L, 5L))
out <- model(X)
stopifnot(identical(out$MX$shape, c(20L, 5L)))
stopifnot(identical(out$MZ$shape, c(20L, 5L)))
stopifnot(identical(out$A_eff$shape, c(5L, 5L)))
message("  OK")
message("")

# --- 4. elbo_loss, h_func ---
message("---- 4. elbo_loss, h_func ----")
elbo <- model$elbo_loss(X, out$MX, out$MZ)
h_val <- model$h_func()
stopifnot(length(as.numeric(elbo)) == 1)
stopifnot(length(as.numeric(h_val)) == 1)
message("  OK")
message("")

# --- 5. make_daggnn, daggnn_adj ---
message("---- 5. make_daggnn, daggnn_adj ----")
model2 <- make_daggnn(n_nodes = 4L)
A_hat <- daggnn_adj(model2, threshold = 0)
stopifnot(is.matrix(A_hat), nrow(A_hat) == 4, ncol(A_hat) == 4)
message("  OK")
message("")

message("All DAG-GNN tests passed.")
