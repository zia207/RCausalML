# Test temporalCausaDiscovery (R/temporalCausaDiscovery.R)
# Run from package root: Rscript tests/test-temporalCausaDiscovery.R
# Requires: install.packages("torch")

pkg_root <- if (file.exists("R/temporalCausaDiscovery.R")) "." else
  if (file.exists("../R/temporalCausaDiscovery.R")) ".." else
  stop("Run from RCausalML package root")

if (!requireNamespace("torch", quietly = TRUE)) {
  message("SKIP: Package 'torch' not installed — skipping temporalCausaDiscovery tests.")
  message("Install with: install.packages('torch')")
  quit(status = 0, save = "no")
}
suppressPackageStartupMessages(library(torch))

source(file.path(pkg_root, "R/temporalCausaDiscovery.R"))

message("========== temporalCausaDiscovery.R tests ==========\n")

# --- 1. prepare_data shapes and lagged target column ---
message("---- 1. TCDF_prepare_data ----")
f1 <- tempfile(fileext = ".csv")
n <- 30L
write.csv(
  data.frame(x1 = rnorm(n), x2 = rnorm(n), y = rnorm(n)),
  f1,
  row.names = FALSE
)
prep <- TCDF_prepare_data(f1, "y")
stopifnot(length(prep$x$shape) == 2L, length(prep$y$shape) == 2L)
stopifnot(as.integer(prep$x$size(1)) == 3L, as.integer(prep$x$size(2)) == n)
stopifnot(as.integer(prep$y$size(2)) == n)
# First row of y is the target series; x should have lagged y in column "y"
df_chk <- read.csv(f1)
lag_y <- c(0, head(df_chk$y, -1L))
x_y_row <- as.numeric(prep$x[match("y", names(df_chk)), ]$cpu())
stopifnot(max(abs(x_y_row - lag_y)) < 1e-5)
message("  OK\n")

# --- 2. missing target column errors ---
message("---- 2. TCDF_prepare_data missing column ----")
ok <- FALSE
tryCatch(
  TCDF_prepare_data(f1, "nope"),
  error = function(e) ok <<- grepl("not found|Column", e$message, ignore.case = TRUE)
)
stopifnot(ok)
message("  OK\n")

# --- 3. TCDF_find_causes end-to-end (CPU, few epochs) ---
message("---- 3. TCDF_find_causes ----")
r <- TCDF_find_causes(
  target = "y",
  cuda = FALSE,
  epochs = 3L,
  kernel_size = 3L,
  layers = 2L,
  log_interval = 2L,
  lr = 0.01,
  optimizername = "Adam",
  seed = 42L,
  dilation_c = 2L,
  significance = 1.0,
  file = f1
)
stopifnot(is.list(r))
stopifnot(identical(names(r), c("validated", "causeswithdelay", "realloss", "scores")))
stopifnot(is.numeric(r$realloss), length(r$realloss) == 1L, is.finite(r$realloss))
stopifnot(is.numeric(r$scores), length(r$scores) == 3L)
stopifnot(is.data.frame(r$causeswithdelay))
stopifnot(identical(names(r$causeswithdelay), c("cause", "delay")))
stopifnot(is.integer(r$validated) || is.numeric(r$validated))
if (nrow(r$causeswithdelay) > 0L) {
  stopifnot(all(r$causeswithdelay$cause %in% r$validated))
}
message("  OK\n")

# --- 4. Optimizer switch ---
message("---- 4. TCDF_optimizer ----")
prep2 <- TCDF_prepare_data(f1, "y")
X <- prep2$x$unsqueeze(1)$contiguous()
Y <- prep2$y$unsqueeze(3)$contiguous()
dev <- TCDF_device(FALSE)
m <- TCDF_ADDSTCN(2L, 3L, 2L, kernel_size = 3L, cuda = FALSE, dilation_c = 2L)$to(device = dev)
X <- X$to(device = dev)
Y <- Y$to(device = dev)
for (nm in c("Adam", "RMSprop", "SGD")) {
  opt <- TCDF_optimizer(nm, m$parameters, lr = 0.01)
  o <- TCDF_train_one_epoch(1L, X, Y, m, opt, 99L, 99L)
  stopifnot(inherits(o$loss, "torch_tensor"))
}
message("  OK\n")

unlink(f1)

message("========== All temporalCausaDiscovery tests passed ==========")
