# Independent analytic reference for the objective solved by rcupp:
#   F(b) = 1/(2n) ||y - X b||^2 + lambda ||b||_1
# When X'X = n I, the minimiser is the elementwise soft-threshold of
# c = X'y / n at level lambda, i.e. sign(c) * max(|c| - lambda, 0).

orthogonal_lasso_data <- function(beta, lambda) {
  n <- length(beta)
  X <- sqrt(n) * diag(n)
  list(X = X, y = drop(X %*% beta), lambda = lambda)
}

soft_threshold <- function(x, level) {
  sign(x) * pmax(abs(x) - level, 0)
}

objective_value <- function(X, y, beta, lambda) {
  sum((y - drop(X %*% beta))^2) / (2 * nrow(X)) +
    lambda * sum(abs(beta))
}
