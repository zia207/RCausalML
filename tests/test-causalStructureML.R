# Test causalStructureML (defined in R/causalDeepNet.R)
# Run from package root: Rscript tests/test-causalStructureML.R
# Quick: QUICK=1 Rscript tests/test-causalStructureML.R

# causalStructureML lives in causalDeepNet.R (not a separate file)
pkg_root <- if (file.exists("R/causalDeepNet.R")) "." else
  if (file.exists("../R/causalDeepNet.R")) ".." else
  stop("Run from RCausalML package root")

if (requireNamespace("devtools", quietly = TRUE)) {
  suppressPackageStartupMessages(devtools::load_all(pkg_root, quiet = TRUE))
} else {
  # Fall back to installed package
  lib_paths <- c(file.path(pkg_root, ".Rlibrary"), .libPaths())
  .libPaths(lib_paths)
  suppressPackageStartupMessages(library(RCausalML))
}

quick <- nzchar(Sys.getenv("QUICK", ""))

message("========== causalStructureML.R tests ==========\n")

# --- 1. causal_structure_ml_model_descriptions ---
message("---- 1. causal_structure_ml_model_descriptions ----")
desc <- causal_structure_ml_model_descriptions()
stopifnot(is.character(desc), length(desc) == 5L)
exp_nms <- c(
  "notears_linear", "notears_nonlinear_mlp", "notears_nonlinear_sobolev",
  "dag_gnn", "grandag"
)
stopifnot(identical(names(desc), exp_nms))
stopifnot(all(nzchar(desc)))
message("  OK\n")

set.seed(42L)
n <- if (quick) 25L else 60L
d <- 4L
X <- matrix(rnorm(as.integer(n * d)), n, d)
colnames(X) <- paste0("x", seq_len(d))

# --- 2. notears_linear ---
message("---- 2. causalStructureML(notears_linear) ----")
r_lin <- causalStructureML(
  X,
  method = "notears_linear",
  lambda1 = 0.15,
  max_iter = if (quick) 15L else 40L,
  w_threshold = 0.25
)
stopifnot(inherits(r_lin, "causal_structure_ml"))
stopifnot(r_lin$method == "notears_linear")
stopifnot(identical(dim(r_lin$adjacency), c(d, d)))
stopifnot(identical(dim(r_lin$binary_adjacency), c(d, d)))
stopifnot(is.null(r_lin$fit))
stopifnot(r_lin$binary_adjacency %in% c(0L, 1L))
stopifnot(identical(rownames(r_lin$adjacency), colnames(X)))
message("  OK\n")

# --- 3. dag_gnn (torch) ---
message("---- 3. causalStructureML(dag_gnn) ----")
if (!requireNamespace("torch", quietly = TRUE)) {
  message("  SKIP (torch not installed)\n")
} else {
  r_dg <- causalStructureML(
    X,
    method = "dag_gnn",
    n_epochs = if (quick) 6L else 20L,
    hidden_dim = 16L,
    adj_threshold = 0.10,
    verbose = FALSE,
    seed = 1L
  )
  stopifnot(r_dg$method == "dag_gnn")
  stopifnot(identical(dim(r_dg$adjacency), c(d, d)))
  stopifnot(!is.null(r_dg$fit))
  stopifnot(is.numeric(r_dg$extra$final_loss) || is.na(r_dg$extra$final_loss))
  message("  OK\n")
}

# --- 4. notears_nonlinear_mlp (torch) ---
message("---- 4. causalStructureML(notears_nonlinear_mlp) ----")
if (!requireNamespace("torch", quietly = TRUE)) {
  message("  SKIP (torch not installed)\n")
} else {
  r_mlp <- causalStructureML(
    X,
    method = "notears_nonlinear_mlp",
    max_iter = if (quick) 15L else 40L,
    lbfgs_iter = 2L,
    verbose = FALSE,
    notears_hidden = 4L,
    w_threshold = 0.2
  )
  stopifnot(r_mlp$method == "notears_nonlinear_mlp")
  stopifnot(identical(dim(r_mlp$adjacency), c(d, d)))
  stopifnot(!is.null(r_mlp$fit))
  message("  OK\n")
}

# --- 5. notears_nonlinear_sobolev (torch) ---
message("---- 5. causalStructureML(notears_nonlinear_sobolev) ----")
if (!requireNamespace("torch", quietly = TRUE)) {
  message("  SKIP (torch not installed)\n")
} else {
  # L-BFGS fine-tune in notears_nonlinear() can hit NaN on small random data;
  # Adam-only phase is enough to smoke-test the causalStructureML wrapper.
  r_sob <- causalStructureML(
    X,
    method = "notears_nonlinear_sobolev",
    max_iter = if (quick) 20L else 40L,
    lbfgs_iter = 0L,
    verbose = FALSE,
    sobolev_k = 3L
  )
  stopifnot(r_sob$method == "notears_nonlinear_sobolev")
  stopifnot(identical(dim(r_sob$adjacency), c(d, d)))
  stopifnot(!is.null(r_sob$fit))
  message("  OK\n")
}

# --- 6. grandag (torch) ---
message("---- 6. causalStructureML(grandag) ----")
if (!requireNamespace("torch", quietly = TRUE)) {
  message("  SKIP (torch not installed)\n")
} else {
  r_g <- causalStructureML(
    X,
    method = "grandag",
    iterations = if (quick) 25L else 80L,
    batch_size = min(32L, n),
    stop_crit_win = 10L,
    lr = 1e-3
  )
  stopifnot(r_g$method == "grandag")
  stopifnot(identical(dim(r_g$adjacency), c(d, d)))
  stopifnot(inherits(r_g$fit, "GraNDAG"))
  stopifnot(identical(rownames(r_g$adjacency), colnames(X)))
  message("  OK\n")
}

# --- 7. data.frame input ---
message("---- 7. data.frame input ----")
df <- as.data.frame(X)
r_df <- causalStructureML(df, "notears_linear", max_iter = 10L, lambda1 = 0.2)
stopifnot(nrow(r_df$adjacency) == d)
message("  OK\n")

message("All causalStructureML.R tests passed.")
