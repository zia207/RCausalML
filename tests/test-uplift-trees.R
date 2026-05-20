# Test: uplift trees and forests (R/uplift_trees.R) with synthetic data
# Run from package root: Rscript tests/test-uplift-trees.R
# Or: source("tests/test-uplift-trees.R")

pkg_root <- if (file.exists("R/uplift_trees.R")) "." else
  if (file.exists("../R/uplift_trees.R")) ".." else stop("Run from Causal_ML package root")

# Source dependencies (same pattern as test-policy-learner.R)
source(file.path(pkg_root, "R/utils.R"))
source(file.path(pkg_root, "R/dataset.R"))
source(file.path(pkg_root, "R/uplift_tree_core.R"))
source(file.path(pkg_root, "R/uplift_trees.R"))
if (!exists("fit", mode = "function")) fit <- function(obj, ...) UseMethod("fit")

set.seed(42)

# ---- Synthetic data: binary outcome for uplift (make_uplift_classification) ----
out <- make_uplift_classification(
  treatment_name = c("control", "treatment1"),
  y_name = "conversion",
  n_samples = 1200,
  n_classification_features = 8,
  n_classification_informative = 4,
  n_uplift_increase_dict = list(treatment1 = 3),
  n_uplift_decrease_dict = list(treatment1 = 2),
  delta_uplift_increase_dict = list(treatment1 = 0.12),
  delta_uplift_decrease_dict = list(treatment1 = -0.08),
  random_seed = 42
)
df <- out$data
X <- as.matrix(df[, out$X_names])
w <- as.integer(df$treatment_group_key != "control")
y <- df$conversion

n <- nrow(X)
message("========== Synthetic uplift data ==========")
message("n = ", n, ", p = ", ncol(X))
message("Treatment rate: ", round(mean(w), 3))
message("Conversion (control): ", round(mean(y[w == 0]), 3), " | treatment: ", round(mean(y[w == 1]), 3))
message("")

# Train/test split
idx_test <- sample(n, size = round(0.25 * n))
X_train <- X[-idx_test, , drop = FALSE]
X_test <- X[idx_test, , drop = FALSE]
w_train <- w[-idx_test]
y_train <- y[-idx_test]

# ---- Single uplift trees (binary outcome) ----
message("---- Single trees: DDP, IDDP, IT, CIT ----")

tree_ddp <- uplift_tree_ddp(X_train, w_train, y_train, min_node_size = 30, max_depth = 4)
tau_ddp <- predict(tree_ddp, X_test)
message("uplift_tree_ddp: fitted OK; mean predicted CATE (test) = ", round(mean(tau_ddp), 4))

tree_iddp <- uplift_tree_iddp(X_train, w_train, y_train, min_node_size = 30, max_depth = 4)
tau_iddp <- predict(tree_iddp, X_test)
message("uplift_tree_iddp: fitted OK; mean predicted CATE (test) = ", round(mean(tau_iddp), 4))

tree_it <- interaction_tree(X_train, w_train, y_train, min_node_size = 30, max_depth = 4)
tau_it <- predict(tree_it, X_test)
message("interaction_tree: fitted OK; mean predicted CATE (test) = ", round(mean(tau_it), 4))

tree_cit <- causal_inference_tree(X_train, w_train, y_train, min_node_size = 30, max_depth = 4)
tau_cit <- predict(tree_cit, X_test)
message("causal_inference_tree: fitted OK; mean predicted CATE (test) = ", round(mean(tau_cit), 4))

# Optional: print one tree as string
message("")
message("---- Uplift tree (DDP) structure (first few lines) ----")
tree_str <- uplift_tree_string(tree_ddp)
cat(strsplit(tree_str, "\n")[[1]][1:5], sep = "\n")
message("...")
message("")

# ---- Uplift random forests (KL, ED, CTS) ----
message("---- Uplift random forests (n_trees=5 for speed) ----")

rf_kl <- uplift_rf_kl(X_train, w_train, y_train, n_trees = 5, min_node_size = 30,
                      max_depth = 4, random_state = 123)
pred_kl <- predict(rf_kl, X_test)
message("uplift_rf_kl: mean CATE (test) = ", round(mean(pred_kl), 4))
pred_kl_full <- predict(rf_kl, X_test, full_output = TRUE)
stopifnot(all(c("control", "treatment1", "delta_treatment1") %in% names(pred_kl_full)))
message("  full_output: control/treatment1/delta columns OK")

rf_ed <- uplift_rf_ed(X_train, w_train, y_train, n_trees = 5, min_node_size = 30,
                      max_depth = 4, random_state = 123)
pred_ed <- predict(rf_ed, X_test)
message("uplift_rf_ed: mean CATE (test) = ", round(mean(pred_ed), 4))

rf_cts <- uplift_rf_cts(X_train, w_train, y_train, n_trees = 5, min_node_size = 30,
                        max_depth = 4, random_state = 123)
pred_cts <- predict(rf_cts, X_test)
message("uplift_rf_cts: mean CATE (test) = ", round(mean(pred_cts), 4))
message("")

# ---- Multi-treatment (control + 2 treatments) ----
message("---- Multi-treatment uplift forest ----")
out_multi <- make_uplift_classification(
  treatment_name = c("control", "treatment1", "treatment2"),
  n_samples = 900,
  n_classification_features = 6,
  n_classification_informative = 3,
  n_uplift_increase_dict = list(treatment1 = 2, treatment2 = 2),
  n_uplift_decrease_dict = list(treatment1 = 1, treatment2 = 1),
  delta_uplift_increase_dict = list(treatment1 = 0.1, treatment2 = 0.15),
  delta_uplift_decrease_dict = list(treatment1 = -0.05, treatment2 = -0.1),
  random_seed = 42
)
df_m <- out_multi$data
X_m <- as.matrix(df_m[, out_multi$X_names])
w_m <- df_m$treatment_group_key
y_m <- df_m$conversion

rf_multi <- uplift_rf_multi(X_m, w_m, y_m, control_name = "control", n_trees = 3,
                            min_node_size = 40, max_depth = 3)
pred_multi <- predict(rf_multi, X_m[1:50, , drop = FALSE], full_output = TRUE)
message("uplift_rf_multi: fitted OK")
message("  full_output cols: ", paste(names(pred_multi), collapse = ", "))
message("  recommended_treatment sample: ", paste(head(pred_multi$recommended_treatment), collapse = " "))
message("")

message("========== test-uplift-trees.R done ==========")
