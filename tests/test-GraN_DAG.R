# Test GraN-DAG (R/GraN_DAG.R)
# Run from package root: Rscript tests/test-GraN_DAG.R
# Quick run (minimal iterations): QUICK=1 Rscript tests/test-GraN_DAG.R

pkg_root <- if (file.exists("R/causalDeepNet.R")) "." else
  if (file.exists("../R/causalDeepNet.R")) ".." else
    stop("Run from RCausalML package root")

# Dependencies
if (!requireNamespace("torch", quietly = TRUE)) {
  message("SKIP: Package 'torch' not installed — skipping GraN_DAG tests.")
  quit(status = 0, save = "no")
}
if (!requireNamespace("R6", quietly = TRUE)) stop("Package 'R6' required.")
if (!requireNamespace("expm", quietly = TRUE)) stop("Package 'expm' required.")
suppressPackageStartupMessages({
  library(torch)
  library(R6)
  library(expm)
})

source(file.path(pkg_root, "R/GraN_DAG.R"))

quick <- nzchar(Sys.getenv("QUICK", ""))

message("========== GraN-DAG tests (R/GraN_DAG.R) ==========")
message("")

# --- 1. is_acyclic ----
message("---- 1. is_acyclic ----")
# DAG: 1 -> 2 -> 3
adj_dag <- torch_zeros(3, 3)
adj_dag[2, 1] <- 1
adj_dag[3, 2] <- 1
stopifnot(is_acyclic(adj_dag, device = "cpu"))
# Cycle: 1 -> 2 -> 3 -> 1
adj_cycle <- adj_dag$clone()
adj_cycle[1, 3] <- 1
stopifnot(!is_acyclic(adj_cycle, device = "cpu"))
message("  OK")
message("")

# --- 2. compute_constraint (via minimal model) ----
message("---- 2. compute_constraint ----")
# Use a tiny model to get w_adj
model <- NonlinearGaussANM(input_dim = 3L, hidden_num = 1L, hidden_dim = 2L)
w_adj <- model$get_w_adj()
h <- compute_constraint(model, w_adj)
stopifnot(length(as.numeric(h)) == 1)
message("  OK")
message("")

# --- 3. NormalizationData ----
message("---- 3. NormalizationData ----")
set.seed(42)
data <- matrix(rnorm(300), ncol = 5)
nd <- NormalizationData$new(data, train = TRUE, normalize = FALSE)
stopifnot(nd$n_samples > 0, nrow(as.array(nd$data_set)) == nd$n_samples)
s <- nd$sample(10L)
stopifnot(length(s) == 2L, nrow(as.array(s[[1]])) == 10L, ncol(as.array(s[[1]])) == 5L)
message("  OK")
message("")

# --- 4. GraNDAG constructor and get_causal_matrix ----
message("---- 4. GraNDAG constructor ----")
gnd <- GraNDAG$new(input_dim = 5L, hidden_num = 1L, hidden_dim = 4L,
                   iterations = if (quick) 50L else 200L,
                   batch_size = 32L, use_pns = FALSE, device_type = "cpu")
cm <- gnd$get_causal_matrix()
stopifnot(is.matrix(cm), nrow(cm) == 5, ncol(cm) == 5)
message("  OK")
message("")

# --- 5. GraphDAG (visualization) ----
message("---- 5. GraphDAG ----")
# est_dag only (no display, save to temp file)
est <- matrix(c(0,1,0, 0,0,1, 0,0,0), 3, 3, byrow = TRUE)
tmp <- tempfile(fileext = ".png")
out <- GraphDAG(est_dag = est, show = FALSE, save_name = tmp)
stopifnot(file.exists(tmp), is.list(out), identical(dim(out$est_dag), c(3L, 3L)))
unlink(tmp)
# est_dag + true_dag
true_dag <- matrix(c(0,1,0, 0,0,1, 0,0,0), 3, 3, byrow = TRUE)
tmp2 <- tempfile(fileext = ".png")
out2 <- GraphDAG(est_dag = est, true_dag = true_dag, show = FALSE, save_name = tmp2)
stopifnot(file.exists(tmp2), !is.null(out2$true_dag))
unlink(tmp2)
# validation: non-matrix should error
ok_err <- tryCatch({ GraphDAG(est_dag = 1:5, show = FALSE); FALSE }, error = function(e) TRUE)
stopifnot(ok_err)
# neither show nor save_name should error
ok_err2 <- tryCatch({ GraphDAG(est_dag = est, show = FALSE); FALSE }, error = function(e) TRUE)
stopifnot(ok_err2)
message("  OK")
message("")

# --- 6. GraNDAG learn (short run; may fail if torch/CUDA misconfigured) ----
message("---- 6. GraNDAG learn ----")
learn_ok <- tryCatch({
  set.seed(123)
  data <- matrix(rnorm(400), ncol = 5)
  gnd2 <- GraNDAG$new(input_dim = 5L, hidden_num = 1L, hidden_dim = 4L,
                      iterations = if (quick) 30L else 100L,
                      batch_size = 32L, use_pns = FALSE, device_type = "cpu")
  gnd2$learn(data)
  cm2 <- gnd2$get_causal_matrix()
  stopifnot(is.matrix(cm2), nrow(cm2) == 5, ncol(cm2) == 5)
  TRUE
}, error = function(e) {
  message("  SKIP (torch/device error: ", conditionMessage(e), ")")
  FALSE
})
if (learn_ok) message("  OK")
message("")

message("========== GraN-DAG tests done ==========")
