# Test: R/synthetic_data.R (synthetic_data function)
# Run from package root: Rscript tests/test-synthetic_data.R
# Or: source("tests/test-synthetic_data.R")

pkg_root <- if (file.exists("R/synthetic_data.R")) "." else
  if (file.exists("../R/synthetic_data.R")) ".." else stop("Run from RCausalML package root")

source(file.path(pkg_root, "R/synthetic_data.R"))

set.seed(42)

message("========== test-synthetic_data.R ==========")
message("")

# ---- 1. Basic call: mode=2, default n, p, sigma ----
message("---- 1. synthetic_data(mode=2) default args ----")
dat <- synthetic_data(mode = 2, n = 1000, p = 20, sigma = 5.5)
stopifnot(is.list(dat))
stopifnot(all(c("y", "X", "w", "tau", "b", "e") %in% names(dat)))
stopifnot(length(dat$y) == 1000)
stopifnot(nrow(dat$X) == 1000, ncol(dat$X) == 20)
stopifnot(length(dat$w) == 1000)
stopifnot(length(dat$tau) == 1000)
stopifnot(length(dat$b) == 1000)
stopifnot(length(dat$e) == 1000)
stopifnot(all(dat$w %in% c(0L, 1L)))
stopifnot(all(dat$e >= 0.05 & dat$e <= 0.95))
message("  OK: returns list with y, X, w, tau, b, e; correct dimensions")

# ---- 2. Outcome structure: Y = b + tau*W + noise ----
message("---- 2. Outcome structure (Y = b + tau*W + noise) ----")
resid <- dat$y - (dat$b + dat$tau * dat$w)
# Residuals should be roughly N(0, sigma^2)
stopifnot(abs(mean(resid)) < 1)
stopifnot(sd(resid) > 1, sd(resid) < 15)
message("  Residual mean: ", round(mean(resid), 4), ", sd: ", round(sd(resid), 4))
message("  OK: outcome structure consistent")

# ---- 3. Reproducibility (same seed -> same data) ----
message("---- 3. Reproducibility ----")
set.seed(99)
dat1 <- synthetic_data(mode = 2, n = 100, p = 20, sigma = 1)
set.seed(99)
dat2 <- synthetic_data(mode = 2, n = 100, p = 20, sigma = 1)
stopifnot(identical(dat1$X, dat2$X))
stopifnot(identical(dat1$tau, dat2$tau))
stopifnot(identical(dat1$b, dat2$b))
stopifnot(identical(dat1$e, dat2$e))
stopifnot(identical(dat1$w, dat2$w))
stopifnot(identical(dat1$y, dat2$y))
message("  OK: same seed gives identical data")

# ---- 4. mode != 2 raises error ----
message("---- 4. mode != 2 raises error ----")
err <- tryCatch(synthetic_data(mode = 1), error = function(e) e)
stopifnot(inherits(err, "error"))
message("  OK: mode=1 raises error as documented")

# ---- 5. Custom n, p (mode=2 requires p >= 11) ----
message("---- 5. Custom n, p ----")
dat_small <- synthetic_data(mode = 2, n = 50, p = 15, sigma = 2)
stopifnot(nrow(dat_small$X) == 50, ncol(dat_small$X) == 15)
stopifnot(length(dat_small$y) == 50)
message("  OK: custom n=50, p=15 works")

message("")
message("========== All tests passed. ==========")
