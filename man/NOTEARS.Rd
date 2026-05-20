% NOTE: Topic overview; individual function docs are in R/notears.R
\name{NOTEARS}
\alias{NOTEARS}
\title{NOTEARS: Causal DAG structure learning via continuous optimization}
\description{
The \pkg{RCausalML} package implements NOTEARS (NO TEARS) for causal directed acyclic graph (DAG) structure learning from observational data. NOTEARS treats the discrete acyclicity constraint as a continuous constraint \eqn{h(W) = 0} using the trace of the matrix exponential (\eqn{h(W) = \mathrm{tr}(e^{W \odot W}) - d}), so the problem is solved with continuous optimization (e.g. augmented Lagrangian with L-BFGS-B for linear, Adam + L-BFGS for nonlinear).

\strong{Linear NOTEARS} (pure R): learns a weighted adjacency matrix \eqn{W} by minimizing a loss (e.g. least squares, logistic, Poisson) plus L1 penalty, subject to \eqn{h(W) = 0}. Uses \code{expm} and L-BFGS-B.

\strong{Nonlinear NOTEARS} (R + \pkg{torch}): uses MLP or Sobolev basis models to capture nonlinear relationships; acyclicity is enforced via a differentiable \eqn{h(W)}. Requires the \pkg{torch} package.

\strong{Main functions:}
\itemize{
  \item \code{\link{notears_linear}} — Linear NOTEARS: L2/logistic/Poisson loss + L1, augmented Lagrangian.
  \item \code{\link{notears_nonlinear}} — Nonlinear NOTEARS with MLP or Sobolev model (Adam then L-BFGS).
  \item \code{NotearsMLP} — Torch module: MLP for nonlinear SEM (when \pkg{torch} is installed).
  \item \code{NotearsSobolev} — Torch module: Sobolev basis for nonlinear SEM (when \pkg{torch} is installed).
}

\strong{Utilities and simulation:}
\itemize{
  \item \code{\link{is_dag}} — Check if a weighted adjacency matrix is a DAG.
  \item \code{\link{simulate_dag}} — Simulate random DAG (Erdős–Rényi or scale-free).
  \item \code{\link{simulate_parameter}} — Draw edge weights for a DAG.
  \item \code{\link{simulate_linear_sem}} — Generate data from a linear SEM (Gaussian, exponential, uniform noise).
  \item \code{\link{simulate_nonlinear_sem}} — Generate data from a nonlinear SEM (MLP or MIM).
  \item \code{\link{count_accuracy}} — Compare estimated graph to ground truth (FDR, TPR, FPR, SHD, nnz).
  \item \code{\link{set_random_seed}} — Set R and torch seeds for reproducibility.
}

\strong{Demos:}
\itemize{
  \item \code{\link{demo_linear}} — Linear NOTEARS on synthetic data.
  \item \code{\link{demo_nonlinear_mlp}} — Nonlinear NOTEARS (MLP) on synthetic data.
  \item \code{\link{demo_nonlinear_sobolev}} — Nonlinear NOTEARS (Sobolev) on random data.
}
}
\seealso{
\code{\link{notears_linear}}, \code{\link{notears_nonlinear}}, \code{\link{simulate_linear_sem}}, \code{\link{simulate_dag}}, \code{\link{count_accuracy}}.
}
\references{
Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. \emph{NeurIPS}. \url{https://arxiv.org/abs/1803.01422}.
}
