# =============================================================================
# causaldata integration: Load textbook datasets for use with RCausalML
# =============================================================================
# Datasets from the causaldata R package (Huntington-Klein & Barrett) are
# standardized to (X, w, y) or (X, w, y, Z) for use with meta-learners, DML,
# causal forests, and IV methods. Always cite causaldata and the original
# data sources when using these datasets; see ?load_causaldata and citation("causaldata").
# =============================================================================

.check_causaldata <- function() {
  if (!requireNamespace("causaldata", quietly = TRUE))
    stop("Package 'causaldata' is required. Install with: install.packages(\"causaldata\")")
}

# Load a dataset from causaldata: use data() first (works when not in namespace)
.get_causaldata_dataset <- function(name) {
  env <- new.env()
  tryCatch(
    data(list = name, package = "causaldata", envir = env),
    error = function(e) NULL
  )
  if (exists(name, envir = env, inherits = FALSE))
    return(get(name, envir = env))
  # Some .rda files expose object under a different name
  objs <- ls(env, all.names = TRUE)
  if (length(objs) >= 1L)
    return(get(objs[1L], envir = env))
  # Fallback: namespace (e.g. lazy-loaded data)
  ns <- asNamespace("causaldata")
  if (exists(name, envir = ns, inherits = FALSE))
    return(get(name, envir = ns))
  stop("Dataset '", name, "' not found in package 'causaldata'. ",
       "Try: data(package = \"causaldata\") to list datasets, or update: install.packages(\"causaldata\").")
}

#' List causaldata datasets supported for RCausalML
#'
#' Returns a table of dataset names and their role (treatment, outcome, IV)
#' for use with \code{\link{load_causaldata}}. All datasets require the
#' \pkg{causaldata} package.
#'
#' @details
#' Use this function first to discover supported dataset names, then pass a
#' selected name to \code{\link{load_causaldata}}.
#'
#' @return A data frame with columns: \code{name}, \code{description},
#'   \code{type} (binary_treatment, iv), \code{treatment}, \code{outcome},
#'   \code{covariates}, \code{textbook}.
#' @references
#' Huntington-Klein, N., & Barrett, M. (2024). causaldata: Example data sets
#' for causal inference textbooks. R package.
#' \url{https://github.com/NickCH-K/causaldata}.
#'
#' Huntington-Klein, N. (2021). \emph{The Effect: An Introduction to Research
#' Design and Causality}. \url{https://theeffectbook.net}.
#'
#' Cunningham, S. (2021). \emph{Causal Inference: The Mixtape}. Yale University
#' Press. \url{https://mixtape.scunning.com}.
#'
#' Hernan, M. A., & Robins, J. M. (2020). \emph{Causal Inference: What If}.
#' Chapman & Hall/CRC. \url{https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book}.
#' @examples
#' \dontrun{
#' library(RCausalML)
#' ds <- list_causaldata_datasets()
#' head(ds)
#' }
#' @seealso \code{\link{load_causaldata}}, \code{citation("causaldata")}
#' @export
list_causaldata_datasets <- function() {
  .check_causaldata()
  out <- data.frame(
    name = c(
      "nsw_mixtape",
      "cps_mixtape",
      "abortion",
      "close_college",
      "social_insure",
      "black_politicians",
      "thornton_hiv",
      "nhefs_complete"
    ),
    description = c(
      "NSW job training experiment (Lalonde); experimental",
      "CPS comparison to NSW; observational",
      "Abortion repeal and gonorrhea (state-year); DiD",
      "College proximity and wages (Card); IV",
      "Social networks and insurance take-up; experiment",
      "Black politicians and turnout; RDD",
      "HIV testing incentive experiment",
      "NHEFS smoking and weight; observational"
    ),
    type = c(
      "binary_treatment",
      "binary_treatment",
      "binary_treatment",
      "iv",
      "binary_treatment",
      "binary_treatment",
      "binary_treatment",
      "binary_treatment"
    ),
    treatment = c(
      "treat", "treat", "repeal", "educ", "default", "treat_out", "any", "qsmk"
    ),
    outcome = c(
      "re78", "re78", "lnr", "lwage", "takeup_survey", "responded", "got", "wt82_71"
    ),
    covariates = c(
      "age, educ, black, hisp, marr, nodegree, re74, re75",
      "age, educ, black, hisp, marr, nodegree, re74, re75",
      "age, race, year, income, ur, poverty, crack, alcohol, ...",
      "exper, black, south, married, smsa",
      "age, agpop, ricearea_2010, disaster_prob, male, intensive, ...",
      "totalpop, medianhhincom, blackpercent, leg_black, ...",
      "age, distvct, villnum, tinc",
      "sex, age, race, school, smokeintensity, smokeyrs, ..."
    ),
    textbook = c(
      "Mixtape", "Mixtape", "Mixtape", "Mixtape", "The Effect",
      "Mixtape", "The Effect", "What If"
    ),
    stringsAsFactors = FALSE
  )
  out
}

#' Load a causaldata dataset in RCausalML-ready form
#'
#' Loads a dataset from the \pkg{causaldata} package and returns covariates
#' \code{X}, treatment \code{w}, outcome \code{y}, and optionally instrument
#' \code{Z}, plus the full data frame and citation information. Use these
#' with \code{\link{SLearner}}, \code{\link{DRLearner}}, \code{\link{causal_forest}},
#' \code{\link{OrthoIVLearner}}, etc. Always cite the causaldata package and
#' the original data source when publishing or reporting results.
#'
#' @details
#' This helper standardizes selected textbook datasets into a common shape used
#' throughout RCausalML:
#' \code{X} (covariates), \code{w} (treatment), \code{y} (outcome), and
#' optionally \code{Z} (instrument for IV examples).
#'
#' @param name Character. One of \code{nsw_mixtape}, \code{cps_mixtape},
#'   \code{abortion}, \code{close_college}, \code{social_insure},
#'   \code{black_politicians}, \code{thornton_hiv}, \code{nhefs_complete}.
#'   See \code{\link{list_causaldata_datasets}}.
#' @param subset_abortion If \code{name == "abortion"}, subset to 15--19 year-old
#'   gonorrhea (\code{bf15 == 1}) and drop NA in key variables. Default \code{TRUE}.
#' @return A list with components:
#'   \item{X}{Model matrix of covariates (numeric).}
#'   \item{w}{Treatment vector (0/1 or numeric for \code{close_college}).}
#'   \item{y}{Outcome vector.}
#'   \item{Z}{Instrument (only for \code{close_college}).}
#'   \item{data}{Original \code{data.frame} from causaldata.}
#'   \item{citation}{List with \code{package}, \code{data_source}, \code{textbook}
#'     for use in reports; run \code{citation("causaldata")} for BibTeX.}
#' @references
#' Huntington-Klein, N., & Barrett, M. (2024). causaldata: Example data sets
#' for causal inference textbooks. R package.
#' \url{https://github.com/NickCH-K/causaldata}.
#'
#' Cunningham, S. (2021). \emph{Causal Inference: The Mixtape}. Yale University
#' Press. \url{https://mixtape.scunning.com}.
#'
#' Huntington-Klein, N. (2021). \emph{The Effect: An Introduction to Research
#' Design and Causality}. \url{https://theeffectbook.net}.
#'
#' Hernan, M. A., & Robins, J. M. (2020). \emph{Causal Inference: What If}.
#' Chapman & Hall/CRC.
#' @seealso \code{\link{list_causaldata_datasets}}, \code{citation("causaldata")}
#' @examples
#' \dontrun{
#' library(RCausalML)
#' # List supported datasets
#' list_causaldata_datasets()
#'
#' # NSW experimental data (Mixtape)
#' d <- load_causaldata("nsw_mixtape")
#' sl <- SLearner(learner = "ranger")
#' sl <- fit(sl, d$X, d$w, d$y)
#' estimate_ate(sl, d$X, d$w, d$y)
#' # Cite the data
#' citation("causaldata")
#' print(d$citation)
#'
#' # IV: college proximity and wages (Card)
#' d_iv <- load_causaldata("close_college")
#' # Use d_iv$X, d_iv$w (educ), d_iv$y (lwage), d_iv$Z (nearc4) with OrthoIVLearner
#' }
#' @export
load_causaldata <- function(name,
                            subset_abortion = TRUE) {
  .check_causaldata()
  name <- match.arg(name, c(
    "nsw_mixtape", "cps_mixtape", "abortion", "close_college",
    "social_insure", "black_politicians", "thornton_hiv", "nhefs_complete"
  ))

  citation_base <- list(
    package = "Huntington-Klein, N., & Barrett, M. (2024). causaldata: Example data sets for causal inference textbooks. R package. https://github.com/NickCH-K/causaldata.",
    data_source = NULL,
    textbook = NULL
  )

  if (name == "nsw_mixtape") {
    data <- .get_causaldata_dataset("nsw_mixtape")
    cov_cols <- c("age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75")
    X <- as.matrix(data[, cov_cols])
    rownames(X) <- NULL
    w <- as.numeric(data$treat)
    y <- as.numeric(data$re78)
    citation_base$data_source <- "Lalonde, R. (1986). Evaluating the econometric evaluations of training programs with experimental data. American Economic Review, 76(4), 604-620."
    citation_base$textbook <- "Cunningham, S. (2021). Causal Inference: The Mixtape. Yale University Press."
    return(list(X = X, w = w, y = y, data = data, citation = citation_base))
  }

  if (name == "cps_mixtape") {
    data <- .get_causaldata_dataset("cps_mixtape")
    cov_cols <- c("age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75")
    X <- as.matrix(data[, cov_cols])
    rownames(X) <- NULL
    w <- as.numeric(data$treat)
    y <- as.numeric(data$re78)
    citation_base$data_source <- "Dehejia, R. H., & Wahba, S. (1999). Causal effects in nonexperimental studies: Reevaluating the evaluation of training programs. JASA, 94(448), 1053-1062."
    citation_base$textbook <- "Cunningham, S. (2021). Causal Inference: The Mixtape. Yale University Press."
    return(list(X = X, w = w, y = y, data = data, citation = citation_base))
  }

  if (name == "abortion") {
    data <- .get_causaldata_dataset("abortion")
    if (isTRUE(subset_abortion) && "bf15" %in% names(data)) {
      data <- data[data$bf15 == 1L, , drop = FALSE]
    }
    cov_cols <- c("age", "race", "year", "income", "ur", "poverty", "crack", "alcohol")
    cov_cols <- intersect(cov_cols, names(data))
    if (length(cov_cols) == 0) cov_cols <- c("year", "income", "ur", "poverty")
    cov_cols <- intersect(cov_cols, names(data))
    keep <- complete.cases(data[, c("repeal", "lnr", cov_cols), drop = FALSE])
    data <- data[keep, , drop = FALSE]
    X <- as.matrix(data[, cov_cols, drop = FALSE])
    rownames(X) <- NULL
    w <- as.numeric(data$repeal)
    y <- as.numeric(data$lnr)
    citation_base$data_source <- "Cunningham, S., & Cornwell, C. (2013). The long-run effect of abortion on sexually transmitted infections. American Law and Economics Review, 15(1), 381-407."
    citation_base$textbook <- "Cunningham, S. (2021). Causal Inference: The Mixtape. Yale University Press."
    return(list(X = X, w = w, y = y, data = data, citation = citation_base))
  }

  if (name == "close_college") {
    data <- .get_causaldata_dataset("close_college")
    cov_cols <- c("exper", "black", "south", "married", "smsa")
    X <- as.matrix(data[, cov_cols])
    rownames(X) <- NULL
    w <- as.numeric(data$educ)
    y <- as.numeric(data$lwage)
    Z <- as.numeric(data$nearc4)
    citation_base$data_source <- "Card, D. (1995). Aspects of labour economics: Essays in honour of John Vanderkamp. University of Toronto Press."
    citation_base$textbook <- "Cunningham, S. (2021). Causal Inference: The Mixtape. Yale University Press."
    return(list(X = X, w = w, y = y, Z = Z, data = data, citation = citation_base))
  }

  if (name == "social_insure") {
    data <- .get_causaldata_dataset("social_insure")
    cov_cols <- c("age", "agpop", "ricearea_2010", "disaster_prob", "male", "intensive", "risk_averse", "literacy", "pre_takeup_rate")
    cov_cols <- intersect(cov_cols, names(data))
    keep <- complete.cases(data[, c("default", "takeup_survey", cov_cols), drop = FALSE])
    data <- data[keep, , drop = FALSE]
    X <- as.matrix(data[, cov_cols, drop = FALSE])
    rownames(X) <- NULL
    w <- as.numeric(data$default)
    y <- as.numeric(data$takeup_survey)
    citation_base$data_source <- "Cai, J., De Janvry, A., & Sadoulet, E. (2015). Social networks and the decision to insure. American Economic Journal: Applied Economics, 7(2), 81-108."
    citation_base$textbook <- "Huntington-Klein, N. (2021). The Effect: An Introduction to Research Design and Causality."
    return(list(X = X, w = w, y = y, data = data, citation = citation_base))
  }

  if (name == "black_politicians") {
    data <- .get_causaldata_dataset("black_politicians")
    cov_cols <- c("totalpop", "medianhhincom", "black_medianhh", "white_medianhh", "blackpercent", "urbanpercent", "leg_black", "leg_senator", "leg_democrat", "south")
    cov_cols <- intersect(cov_cols, names(data))
    keep <- complete.cases(data[, c("treat_out", "responded", cov_cols), drop = FALSE])
    data <- data[keep, , drop = FALSE]
    X <- as.matrix(data[, cov_cols, drop = FALSE])
    rownames(X) <- NULL
    w <- as.numeric(data$treat_out)
    y <- as.numeric(data$responded)
    citation_base$data_source <- "Broockman, D. E. (2013). Black politicians are more intrinsically motivated to advance blacks' interests. American Journal of Political Science, 57(3), 521-536."
    citation_base$textbook <- "Huntington-Klein, N. (2021). The Effect: An Introduction to Research Design and Causality."
    return(list(X = X, w = w, y = y, data = data, citation = citation_base))
  }

  if (name == "thornton_hiv") {
    data <- .get_causaldata_dataset("thornton_hiv")
    cov_cols <- c("age", "distvct", "villnum", "tinc")
    cov_cols <- intersect(cov_cols, names(data))
    if (length(cov_cols) == 0) cov_cols <- "age"
    keep <- complete.cases(data[, c("any", "got", cov_cols), drop = FALSE])
    data <- data[keep, , drop = FALSE]
    X <- as.matrix(data[, cov_cols, drop = FALSE])
    rownames(X) <- NULL
    w <- as.numeric(data$any)
    y <- as.numeric(data$got)
    citation_base$data_source <- "Thornton, R. L. (2008). The demand for, and impact of, learning HIV status. American Economic Review, 98(5), 1829-1863."
    citation_base$textbook <- "Cunningham, S. (2021). Causal Inference: The Mixtape. Yale University Press."
    return(list(X = X, w = w, y = y, data = data, citation = citation_base))
  }

  if (name == "nhefs_complete") {
    data <- .get_causaldata_dataset("nhefs_complete")
    cov_cols <- c("sex", "age", "race", "school", "smokeintensity", "smokeyrs")
    cov_cols <- intersect(cov_cols, names(data))
    if (length(cov_cols) == 0) cov_cols <- c("sex", "age", "race")
    keep <- complete.cases(data[, c("qsmk", "wt82_71", cov_cols), drop = FALSE])
    data <- data[keep, , drop = FALSE]
    X <- as.matrix(data[, cov_cols, drop = FALSE])
    rownames(X) <- NULL
    w <- as.numeric(data$qsmk)
    y <- as.numeric(data$wt82_71)
    citation_base$data_source <- "Hernan, M. A., & Robins, J. M. (2020). Causal Inference: What If. Chapman & Hall/CRC."
    citation_base$textbook <- "Hernan, M. A., & Robins, J. M. (2020). Causal Inference: What If."
    return(list(X = X, w = w, y = y, data = data, citation = citation_base))
  }

  stop("Unsupported dataset: ", name)
}
