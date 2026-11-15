test_that("lasso_fista returns the documented result structure", {
  d <- orthogonal_lasso_data(c(3, -2, 0, 0), lambda = 0.5)
  res <- lasso_fista(d$X, d$y, d$lambda, max_iter = 2000L, tol = 1e-10)

  expect_true(all(c("beta", "iterations", "objective", "converged",
                    "cuda", "backend") %in% names(res)))
  expect_type(res$beta, "double")
  expect_length(res$beta, ncol(d$X))
  expect_true(res$iterations >= 1L)
  expect_type(res$converged, "logical")
  expect_true(is.finite(res$objective))
  expect_type(res$cuda, "logical")
  expect_true(res$backend %in% c("cuda", "cpu-shim"))
})

test_that("an orthogonal design recovers the analytic lasso solution", {
  beta_true <- c(3, -2, 0, 0)
  lambda <- 0.5
  d <- orthogonal_lasso_data(beta_true, lambda)
  res <- lasso_fista(d$X, d$y, lambda, max_iter = 2000L, tol = 1e-10)

  expected <- soft_threshold(beta_true, lambda)
  expect_equal(res$beta, expected, tolerance = 1e-6)
  expect_equal(res$objective,
               objective_value(d$X, d$y, res$beta, lambda),
               tolerance = 1e-8)
  expect_equal(drop(d$X %*% res$beta),
               drop(d$X %*% expected), tolerance = 1e-6)
})

test_that("a large lambda shrinks every coefficient to zero", {
  d <- orthogonal_lasso_data(c(3, -2, 0, 0), lambda = 1e6)
  res <- lasso_fista(d$X, d$y, 1e6)

  expect_true(all(res$beta == 0))
  expect_equal(res$objective, objective_value(d$X, d$y, res$beta, 1e6))
})

test_that("a zero design matrix gives zero coefficients", {
  set.seed(1)
  X <- matrix(0, 30, 4)
  y <- rnorm(30)
  res <- lasso_fista(X, y, lambda = 0.05)

  expect_true(all(res$beta == 0))
  expect_true(is.finite(res$objective))
})

test_that("coefficients inherit the column names of X", {
  d <- orthogonal_lasso_data(c(3, -2, 0, 0), lambda = 0.5)
  colnames(d$X) <- c("a", "b", "c", "d")
  res <- lasso_fista(d$X, d$y, 0.5)
  expect_named(res$beta, c("a", "b", "c", "d"))
})

test_that("x is coerced to a double matrix", {
  d <- orthogonal_lasso_data(c(3, -2, 0, 0), lambda = 0.5)
  ref <- lasso_fista(d$X, d$y, 0.5)

  X_int <- matrix(as.integer(d$X), nrow(d$X), ncol(d$X))
  res_int <- lasso_fista(X_int, d$y, 0.5)
  expect_equal(as.numeric(res_int$beta), as.numeric(ref$beta),
               tolerance = 1e-8)

  # as.matrix() also accepts a data frame
  res_df <- lasso_fista(as.data.frame(d$X), d$y, 0.5)
  expect_equal(as.numeric(res_df$beta), as.numeric(ref$beta),
               tolerance = 1e-8)
})

test_that("the objective decreases as lambda shrinks", {
  d <- orthogonal_lasso_data(c(3, -2, 0, 0), lambda = 0.5)
  hi <- lasso_fista(d$X, d$y, 5)
  lo <- lasso_fista(d$X, d$y, 0.01)
  expect_lte(lo$objective, hi$objective)
})

test_that("max_iter caps the number of FISTA iterations", {
  d <- orthogonal_lasso_data(c(3, -2, 0, 0), lambda = 0.5)
  res <- lasso_fista(d$X, d$y, 0.5, max_iter = 1L, tol = 1e-14)
  expect_equal(res$iterations, 1L)
})

test_that("the backend flag agrees with cuda_available()", {
  d <- orthogonal_lasso_data(c(3, -2, 0, 0), lambda = 0.5)
  res <- lasso_fista(d$X, d$y, 0.5)

  expect_identical(res$cuda, cuda_available())
  expected_backend <- if (isTRUE(res$cuda)) "cuda" else "cpu-shim"
  expect_identical(res$backend, expected_backend)
})
