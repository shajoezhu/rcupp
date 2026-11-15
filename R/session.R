#' Persistent lasso sessions (device-X / cuBLAS-handle reuse)
#'
#' A session keeps the design matrix \code{x} and response \code{y} resident
#' (on the GPU device in a CUDA build), together with the cuBLAS handle, the
#' FISTA work buffers and the cached Lipschitz constant, across MANY fits. Use
#' it for repeated-fit workloads (lambda paths, cross-validation) where the
#' design is unchanged: upload once with [session_set_data()], then call
#' [session_fit()] once per \code{lambda}. This removes the repeated per-call
#' setup (upload, \code{cublasCreate}, \code{cudaMalloc}, power iteration) that
#' dominates such workloads.
#'
#' Cold-start fits (\code{warm_start = FALSE}, the default) are bit-identical to
#' a one-shot [lasso_fista()] call for the same data and \code{lambda}.
#' Warm-start fits seed from the previous solution and converge to the same
#' optimum (not bit-identical), so they should be validated with a relaxed
#' tolerance.
#'
#' Sessions own GPU resources. The external pointer returned by
#' [lasso_session()] has a finalizer that frees them at garbage-collection time;
#' [session_destroy()] frees them immediately.
#'
#' @param session An object returned by [lasso_session()].
#' @param x Numeric \code{n x p} design matrix (no missing values).
#' @param y Numeric response of length \code{n}.
#' @param lambda A single non-negative lasso penalty.
#' @param warm_start Logical; seed the fit from the previous solution reached
#'   by this session. Default \code{FALSE} (cold start).
#' @param max_iter Maximum FISTA iterations for this fit.
#' @param tol Relative convergence tolerance.
#'
#' @return [lasso_session()] returns an external pointer of class
#'   \code{"rcupp_session"}. [session_set_data()] and [session_destroy()]
#'   return \code{invisible(session)} / \code{invisible(NULL)} respectively.
#'   [session_fit()] returns a list like [lasso_fista()]: \code{beta},
#'   \code{iterations}, \code{objective}, \code{converged}, \code{cuda} and
#'   \code{backend}.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(60 * 5), 60, 5)
#' y <- drop(X %*% c(2, -1, 0, 0, 0)) + rnorm(60)
#' s <- lasso_session()
#' session_set_data(s, X, y)
#' fit_a <- session_fit(s, lambda = 0.1)
#' fit_b <- session_fit(s, lambda = 0.05)
#' session_destroy(s)
#'
#' @name lasso_session
NULL

#' @rdname lasso_session
#' @export
lasso_session <- function() {
  session_create_cpp()
}

#' @rdname lasso_session
#' @export
session_set_data <- function(session, x, y) {
  x <- as.matrix(x)
  if (!is.numeric(x) || anyNA(x)) {
    stop("'x' must be a numeric matrix without missing values", call. = FALSE)
  }
  if (storage.mode(x) != "double") storage.mode(x) <- "double"
  y <- as.numeric(y)
  if (length(y) != nrow(x) || anyNA(y)) {
    stop("'y' must be numeric, of length nrow(x), without NA", call. = FALSE)
  }
  session_set_data_cpp(session, x, y)
  invisible(session)
}

#' @rdname lasso_session
#' @export
session_fit <- function(session, lambda, warm_start = FALSE,
                        max_iter = 1000L, tol = 1e-6) {
  if (!is.numeric(lambda) || length(lambda) != 1L || is.na(lambda) ||
        lambda < 0) {
    stop("'lambda' must be a single non-negative number", call. = FALSE)
  }
  if (!is.logical(warm_start) || length(warm_start) != 1L ||
        is.na(warm_start)) {
    stop("'warm_start' must be TRUE or FALSE", call. = FALSE)
  }
  res <- session_fit_cpp(session, as.double(lambda), as.integer(warm_start),
                         as.integer(max_iter), as.double(tol))
  res$backend <- if (isTRUE(res$cuda)) "cuda" else "cpu-shim"
  res
}

#' @rdname lasso_session
#' @export
session_destroy <- function(session) {
  session_destroy_cpp(session)
  invisible(NULL)
}
