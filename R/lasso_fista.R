#' @useDynLib rcupp, .registration = TRUE
#' @import Rcpp
#' @importFrom Rcpp evalCpp
#' @noRd
NULL

#' Fit a Gaussian lasso by FISTA (low-level entry point)
#'
#' Calls the compiled solver in \pkg{rcupp}. When the package was built without
#' a CUDA Toolkit this uses the CPU reference shim; otherwise it runs on the
#' GPU. Device memory (when applicable) stays resident for the whole loop.
#'
#' @param X Numeric \code{n x p} design matrix (column-major on the C++ side).
#' @param y Numeric response of length \code{n}.
#' @param lambda Non-negative lasso penalty (a single value).
#' @param max_iter Maximum number of FISTA iterations.
#' @param tol Relative convergence tolerance.
#'
#' @return A list with \code{beta}, \code{iterations}, \code{objective},
#'   \code{converged}, \code{cuda} and \code{backend}.
#' @export
lasso_fista <- function(X, y, lambda, max_iter = 1000L, tol = 1e-6) {
  X <- as.matrix(X)
  if (storage.mode(X) != "double") storage.mode(X) <- "double"
  y <- as.numeric(y)

  res <- lasso_fista_cpp(X, y, as.double(lambda),
                         as.integer(max_iter), as.double(tol))
  names(res$beta) <- colnames(X)
  res$backend <- if (isTRUE(res$cuda)) "cuda" else "cpu-shim"
  res
}

#' Is a CUDA backend linked into rcupp?
#'
#' @return \code{TRUE} if the package was compiled against a CUDA Toolkit and a
#'   device is available, \code{FALSE} for the CPU reference shim build.
#' @export
cuda_available <- function() {
  isTRUE(cuda_available_cpp())
}
