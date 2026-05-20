#' Causal forest with confounders
#'
#' Trains a causal forest that can be used to estimate
#' conditional average treatment effects tau(X). This function extends
#' \code{\link{causal_forest}} with an optional \code{confounders} argument
#' (similar to \code{bartc} in the bartCause package). When confounders are
#' provided, E[Y|.] and E[W|.] are estimated using both confounders and X,
#' so that residualization controls for confounding. The causal forest still
#' uses X for splits (tau(X)).
#'
#' When the treatment assignment W is binary and unconfounded given the
#' confounders and X, we have tau(X) = E[Y(1) - Y(0) | X = x].
#'
#' @param X The covariates used in the causal regression (for tau(X) and optionally for nuisance).
#' @param Y The outcome (must be a numeric vector with no NAs).
#' @param W The treatment assignment (must be a binary or real numeric vector with no NAs).
#' @param confounders Optional matrix or data frame of confounder variables (same number of rows as X).
#'   When provided, Y.hat and W.hat are estimated using both confounders and X (unless Y.hat/W.hat
#'   are supplied directly). This allows residualization to control for confounding. Default is NULL.
#' @param Y.hat Estimates of the expected responses E[Y | Xi], marginalizing
#'              over treatment. If Y.hat = NULL, these are estimated using
#'              a separate regression forest. When \code{confounders} is provided, the regression
#'              uses (confounders, X). Default is NULL.
#' @param W.hat Estimates of the treatment propensities E[W | Xi]. If W.hat = NULL,
#'              these are estimated using a separate regression forest. When \code{confounders}
#'              is provided, the regression uses (confounders, X). Default is NULL.
#' @param num.trees Number of trees grown in the forest. Note: Getting accurate
#'                  confidence intervals generally requires more trees than
#'                  getting accurate predictions. Default is 2000.
#' @param sample.weights Weights given to each sample in estimation.
#'                       If NULL, each observation receives the same weight.
#'                       Note: To avoid introducing confounding, weights should be
#'                       independent of the potential outcomes given X. Default is NULL.
#' @param clusters Vector of integers or factors specifying which cluster each observation corresponds to.
#'  Default is NULL (ignored).
#' @param equalize.cluster.weights If FALSE, each unit is given the same weight (so that bigger
#'  clusters get more weight). If TRUE, each cluster is given equal weight in the forest. In this case,
#'  during training, each tree uses the same number of observations from each drawn cluster: If the
#'  smallest cluster has K units, then when we sample a cluster during training, we only give a random
#'  K elements of the cluster to the tree-growing procedure. When estimating average treatment effects,
#'  each observation is given weight 1/cluster size, so that the total weight of each cluster is the
#'  same. Note that, if this argument is FALSE, sample weights may also be directly adjusted via the
#'  sample.weights argument. If this argument is TRUE, sample.weights must be set to NULL. Default is
#'  FALSE.
#' @param sample.fraction Fraction of the data used to build each tree.
#'                        Note: If honesty = TRUE, these subsamples will
#'                        further be cut by a factor of honesty.fraction. Default is 0.5.
#' @param mtry Number of variables tried for each split. Default is
#'             \eqn{\sqrt p + 20} where p is the number of variables.
#' @param min.node.size A target for the minimum number of observations in each tree leaf. Note that nodes
#'                      with size smaller than min.node.size can occur, as in the original randomForest package.
#'                      Default is 5.
#' @param honesty Whether to use honest splitting (i.e., sub-sample splitting). Default is TRUE.
#'  For a detailed description of honesty, honesty.fraction, honesty.prune.leaves, and recommendations for
#'  parameter tuning, see the grf algorithm reference.
#' @param honesty.fraction The fraction of data that will be used for determining splits if honesty = TRUE. Corresponds
#'                         to set J1 in the notation of the paper. Default is 0.5 (i.e. half of the data is used for
#'                         determining splits).
#' @param honesty.prune.leaves If TRUE, prunes the estimation sample tree such that no leaves
#'  are empty. If FALSE, keep the same tree as determined in the splits sample (if an empty leave is encountered, that
#'  tree is skipped and does not contribute to the estimate). Setting this to FALSE may improve performance on
#'  small/marginally powered data, but requires more trees (note: tuning does not adjust the number of trees).
#'  Only applies if honesty is enabled. Default is TRUE.
#' @param alpha A tuning parameter that controls the maximum imbalance of a split. Default is 0.05.
#' @param imbalance.penalty A tuning parameter that controls how harshly imbalanced splits are penalized. Default is 0.
#' @param stabilize.splits Whether or not the treatment should be taken into account when
#'                         determining the imbalance of a split. Default is TRUE.
#' @param ci.group.size The forest will grow ci.group.size trees on each subsample.
#'                      In order to provide confidence intervals, ci.group.size must
#'                      be at least 2. Default is 2.
#' @param tune.parameters A vector of parameter names to tune.
#'  If "all": all tunable parameters are tuned by cross-validation. The following parameters are
#'  tunable: ("sample.fraction", "mtry", "min.node.size", "honesty.fraction",
#'   "honesty.prune.leaves", "alpha", "imbalance.penalty"). If honesty is FALSE the honesty.* parameters are not tuned.
#'  Default is "none" (no parameters are tuned).
#' @param tune.num.trees The number of trees in each 'mini forest' used to fit the tuning model. Default is 200.
#' @param tune.num.reps The number of forests used to fit the tuning model. Default is 50.
#' @param tune.num.draws The number of random parameter values considered when using the model
#'                          to select the optimal parameters. Default is 1000.
#' @param compute.oob.predictions Whether OOB predictions on training set should be precomputed. Default is TRUE.
#' @param num.threads Number of threads used in training. By default, the number of threads is set
#'                    to the maximum hardware concurrency.
#' @param seed The seed of the C++ random number generator.
#'
#' @return A trained causal forest object (same structure as \code{\link{causal_forest}}, with an additional
#'  \code{confounders} element when confounders were supplied). Use \code{\link{predict.causal_forest}}
#'  for predictions.
#'
#' @references Athey, Susan, Julie Tibshirani, and Stefan Wager. "Generalized Random Forests".
#'  Annals of Statistics, 47(2), 2019.
#' @references  Wager, Stefan, and Susan Athey. "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests".
#'  Journal of the American Statistical Association, 113(523), 2018.
#' @references Nie, Xinkun, and Stefan Wager. "Quasi-Oracle Estimation of Heterogeneous Treatment Effects".
#'  Biometrika, 108(2), 2021.
#'
#' @examples
#' \donttest{
#' # Same as causal_forest when confounders = NULL
#' n <- 500
#' p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' W <- rbinom(n, 1, 0.5)
#' Y <- pmax(X[, 1], 0) * W + X[, 2] + pmin(X[, 3], 0) + rnorm(n)
#' c.forest <- causal_forest_confounder(X, Y, W)
#'
#' # With confounders: use separate confounder matrix for nuisance models
#' Z <- matrix(rnorm(n * 3), n, 3)  # 3 confounders
#' c.forest <- causal_forest_confounder(X, Y, W, confounders = Z)
#' c.pred <- predict(c.forest, X)
#' }
#'
#' @seealso
#' \code{\link{RCausalML-package}}
#' @export
causal_forest_confounder <- function(X, Y, W,
                                    confounders = NULL,
                                    Y.hat = NULL,
                                    W.hat = NULL,
                                    num.trees = 2000,
                                    sample.weights = NULL,
                                    clusters = NULL,
                                    equalize.cluster.weights = FALSE,
                                    sample.fraction = 0.5,
                                    mtry = min(ceiling(sqrt(ncol(X)) + 20), ncol(X)),
                                    min.node.size = 5,
                                    honesty = TRUE,
                                    honesty.fraction = 0.5,
                                    honesty.prune.leaves = TRUE,
                                    alpha = 0.05,
                                    imbalance.penalty = 0,
                                    stabilize.splits = TRUE,
                                    ci.group.size = 2,
                                    tune.parameters = "none",
                                    tune.num.trees = 200,
                                    tune.num.reps = 50,
                                    tune.num.draws = 1000,
                                    compute.oob.predictions = TRUE,
                                    num.threads = NULL,
                                    seed = runif(1, 0, .Machine$integer.max)) {
  has.missing.values <- validate_X(X, allow.na = TRUE)
  validate_sample_weights(sample.weights, X)
  Y <- validate_observations(Y, X)
  W <- validate_observations(W, X)
  clusters <- validate_clusters(clusters, X)
  samples.per.cluster <- validate_equalize_cluster_weights(equalize.cluster.weights, clusters, sample.weights)
  num.threads <- validate_num_threads(num.threads)

  # Validate confounders if provided
  X.orthog <- X
  if (!is.null(confounders)) {
    if (!inherits(confounders, c("matrix", "data.frame"))) {
      stop("confounders must be a matrix or data.frame.")
    }
    confounders <- as.matrix(confounders)
    if (nrow(confounders) != nrow(X)) {
      stop("confounders must have the same number of rows as X.")
    }
    if (!is.numeric(confounders)) {
      stop("confounders must be numeric.")
    }
    if (anyNA(confounders)) {
      stop("confounders must not contain NA.")
    }
    X.orthog <- cbind(confounders, X)
  }

  all.tunable.params <- c("sample.fraction", "mtry", "min.node.size", "honesty.fraction",
                          "honesty.prune.leaves", "alpha", "imbalance.penalty")
  default.parameters <- list(sample.fraction = 0.5,
                             mtry = min(ceiling(sqrt(ncol(X)) + 20), ncol(X)),
                             min.node.size = 5,
                             honesty.fraction = 0.5,
                             honesty.prune.leaves = TRUE,
                             alpha = 0.05,
                             imbalance.penalty = 0)

  args.orthog <- list(X = X.orthog,
                      num.trees = max(50, num.trees / 4),
                      sample.weights = sample.weights,
                      clusters = clusters,
                      equalize.cluster.weights = equalize.cluster.weights,
                      sample.fraction = sample.fraction,
                      mtry = min(mtry, ncol(X.orthog)),
                      min.node.size = 5,
                      honesty = TRUE,
                      honesty.fraction = 0.5,
                      honesty.prune.leaves = honesty.prune.leaves,
                      alpha = alpha,
                      imbalance.penalty = imbalance.penalty,
                      ci.group.size = 1,
                      tune.parameters = tune.parameters,
                      num.threads = num.threads,
                      seed = seed)

  if (is.null(Y.hat)) {
    forest.Y <- do.call(regression_forest, c(Y = list(Y), args.orthog))
    Y.hat <- predict(forest.Y)$predictions
  } else if (length(Y.hat) == 1) {
    Y.hat <- rep(Y.hat, nrow(X))
  } else if (length(Y.hat) != nrow(X)) {
    stop("Y.hat has incorrect length.")
  }

  if (is.null(W.hat)) {
    forest.W <- do.call(regression_forest, c(Y = list(W), args.orthog))
    W.hat <- predict(forest.W)$predictions
  } else if (length(W.hat) == 1) {
    W.hat <- rep(W.hat, nrow(X))
  } else if (length(W.hat) != nrow(X)) {
    stop("W.hat has incorrect length.")
  }

  Y.centered <- Y - Y.hat
  W.centered <- W - W.hat
  data <- create_train_matrices(X, outcome = Y.centered, treatment = W.centered,
                                sample.weights = sample.weights)
  args <- list(num.trees = num.trees,
               clusters = clusters,
               samples.per.cluster = samples.per.cluster,
               sample.fraction = sample.fraction,
               mtry = mtry,
               min.node.size = min.node.size,
               honesty = honesty,
               honesty.fraction = honesty.fraction,
               honesty.prune.leaves = honesty.prune.leaves,
               alpha = alpha,
               imbalance.penalty = imbalance.penalty,
               stabilize.splits = stabilize.splits,
               ci.group.size = ci.group.size,
               compute.oob.predictions = compute.oob.predictions,
               num.threads = num.threads,
               seed = seed,
               reduced.form.weight = 0,
               legacy.seed = get_legacy_seed(),
               verbose = get_verbose())

  tuning.output <- NULL
  if (!identical(tune.parameters, "none")) {
    if (identical(tune.parameters, "all")) {
      tune.parameters <- all.tunable.params
    } else {
      tune.parameters <- unique(match.arg(tune.parameters, all.tunable.params, several.ok = TRUE))
    }
    if (!honesty) {
      tune.parameters <- tune.parameters[!grepl("honesty", tune.parameters)]
    }
    tune.parameters.defaults <- default.parameters[tune.parameters]
    tuning.output <- tune_forest(data = data,
                                 nrow.X = nrow(X),
                                 ncol.X = ncol(X),
                                 args = args,
                                 tune.parameters = tune.parameters,
                                 tune.parameters.defaults = tune.parameters.defaults,
                                 tune.num.trees = tune.num.trees,
                                 tune.num.reps = tune.num.reps,
                                 tune.num.draws = tune.num.draws,
                                 train = causal_train)

    args <- utils::modifyList(args, as.list(tuning.output[["params"]]))
  }

  forest <- do.call.rcpp(causal_train, c(data, args))
  class(forest) <- c("causal_forest", "grf")
  forest[["seed"]] <- seed
  forest[["num.threads"]] <- num.threads
  forest[["ci.group.size"]] <- ci.group.size
  forest[["X.orig"]] <- X
  forest[["Y.orig"]] <- Y
  forest[["W.orig"]] <- W
  forest[["Y.hat"]] <- Y.hat
  forest[["W.hat"]] <- W.hat
  forest[["confounders"]] <- confounders
  forest[["clusters"]] <- clusters
  forest[["equalize.cluster.weights"]] <- equalize.cluster.weights
  forest[["sample.weights"]] <- sample.weights
  forest[["tunable.params"]] <- args[all.tunable.params]
  forest[["tuning.output"]] <- tuning.output
  forest[["has.missing.values"]] <- has.missing.values

  forest
}
