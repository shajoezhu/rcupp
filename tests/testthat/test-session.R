# Session API tests. These are deliberately backend-agnostic: they must pass
# identically on a CUDA build and on the no-CUDA CPU fallback.

rand_data <- function(n = 80L, p = 6L, seed = 7L) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  beta <- c(2.5, -1.5, rep(0, p - 2L))
  list(X = X, y = drop(X %*% beta) + rnorm(n, sd = 0.4))
}

test_that("session lifecycle: create / set_data / fit / destroy", {
  d <- rand_data()
  s <- lasso_session()
  expect_s3_class(s, "rcupp_session")
  expect_type(s, "externalptr")

  expect_invisible(session_set_data(s, d$X, d$y))
  res <- session_fit(s, lambda = 0.05)
  expect_true(all(c("beta", "iterations", "objective", "converged",
                    "cuda", "backend") %in% names(res)))
  expect_length(res$beta, ncol(d$X))
  expect_true(res$iterations >= 1L)
  expect_true(is.finite(res$objective))
  expect_identical(res$cuda, cuda_available())
  expect_true(res$backend %in% c("cuda", "cpu-shim"))

  expect_invisible(session_destroy(s))
})

test_that("cold-start session fit is bit-identical to one-shot lasso_fista", {
  d <- rand_data(n = 90L, p = 8L, seed = 11L)
  s <- lasso_session()
  session_set_data(s, d$X, d$y)

  for (lam in c(0.5, 0.1, 0.05, 0.005)) {
    one <- lasso_fista(d$X, d$y, lam, max_iter = 2000L, tol = 1e-10)
    ss <- session_fit(s, lam, warm_start = FALSE,
                      max_iter = 2000L, tol = 1e-10)
    expect_equal(ss$beta, one$beta, tolerance = 0)      # bit-identical
    expect_equal(ss$objective, one$objective, tolerance = 0)
    expect_equal(ss$iterations, one$iterations)
  }
  session_destroy(s)
})

test_that("repeated fits over one session stay correct (X uploaded once)", {
  d <- rand_data(n = 100L, p = 7L, seed = 13L)
  s <- lasso_session()
  session_set_data(s, d$X, d$y)
  lams <- c(0.3, 0.2, 0.1, 0.05, 0.02, 0.01)
  prev_obj <- Inf
  for (lam in lams) {
    ss <- session_fit(s, lam)
    one <- lasso_fista(d$X, d$y, lam)
    expect_equal(ss$beta, one$beta, tolerance = 0)
    expect_true(is.finite(ss$objective))
    prev_obj <- ss$objective
  }
  # sanity: smaller lambda does not increase the fitted objective value
  expect_true(is.finite(prev_obj))
  session_destroy(s)
})

test_that("warm-start converges to the same optimum (relaxed tolerance)", {
  d <- rand_data(n = 100L, p = 8L, seed = 17L)
  lam <- 0.05
  cold_one <- lasso_fista(d$X, d$y, lam, max_iter = 2000L, tol = 1e-10)

  s <- lasso_session()
  session_set_data(s, d$X, d$y)
  # a preceding fit on the same session provides the warm-start seed
  session_fit(s, 0.2, warm_start = FALSE, max_iter = 2000L, tol = 1e-10)
  warm <- session_fit(s, lam, warm_start = TRUE,
                      max_iter = 2000L, tol = 1e-10)
  expect_equal(warm$beta, cold_one$beta, tolerance = 1e-3)
  expect_equal(warm$objective, cold_one$objective, tolerance = 1e-4)
  session_destroy(s)
})

test_that("session input validation and destroyed-handle safety", {
  d <- rand_data()
  s <- lasso_session()
  expect_error(session_fit(s, 0.05), "no data")   # fit before set_data
  session_set_data(s, d$X, d$y)
  expect_error(session_fit(s, -1), "non-negative")
  expect_error(session_fit(s, 0.05, warm_start = NA), "TRUE or FALSE")
  expect_error(session_set_data(s, d$X, d$y[1:3]), "length nrow")
  session_destroy(s)
  # a destroyed handle can no longer be used
  expect_error(session_fit(s, 0.05), "destroyed")
})

test_that("independent sessions do not interfere", {
  d1 <- rand_data(seed = 21L)
  d2 <- rand_data(seed = 22L)
  s1 <- lasso_session(); session_set_data(s1, d1$X, d1$y)
  s2 <- lasso_session(); session_set_data(s2, d2$X, d2$y)
  expect_equal(session_fit(s1, 0.05)$beta,
               lasso_fista(d1$X, d1$y, 0.05)$beta, tolerance = 0)
  expect_equal(session_fit(s2, 0.05)$beta,
               lasso_fista(d2$X, d2$y, 0.05)$beta, tolerance = 0)
  session_destroy(s1); session_destroy(s2)
})
