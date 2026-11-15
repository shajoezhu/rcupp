#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include "rcupp_internal.h"

/*
 * NO-CUDA build only. Implements the shared ABI with Armadillo/BLAS.
 *
 * This is the SUPPORTED CPU fallback: it provides the same FISTA solver as the
 * CUDA build (rcupp_api_cpu.cpp provides the same public C API), so the core
 * packages and all downstream packages install, run and pass R CMD check on
 * machines without a toolkit -- just without GPU acceleration. It deliberately
 * has NO [[Rcpp::export]] attributes (those live in rcupp_exports.cpp only);
 * the link step selects either this object or the CUDA objects, never both.
 *
 * The numerical contract is identical, line for line, to
 *   src/lasso_fista_cuda.cu   and   ../cudaGlmnet/R/lasso_fista_cpu.R
 * so GPU/CPU-fallback/oracle results can be compared directly.
 */

namespace {

/* Largest eigenvalue of X'X by deterministic power iteration.
 * Start vector v[j] = sin(j+1), j = 0..p-1 -- identical in all three impls. */
double largest_eig_xtx(const arma::mat& X, int p) {
  arma::vec v(p);
  for (int j = 0; j < p; ++j) v[j] = std::sin(static_cast<double>(j + 1));
  const double nv = arma::norm(v, 2);
  if (nv > 0.0) v /= nv;

  double lf = 0.0;
  for (int k = 0; k < 100; ++k) {
    const arma::vec u = X.t() * (X * v);       // X'X v
    const double rq = arma::dot(v, u);         // Rayleigh quotient, v unit norm
    const double nu = arma::norm(u, 2);
    lf = rq;
    if (nu > 0.0) {
      v = u / nu;
    } else {
      break;
    }
  }
  return lf > 0.0 ? lf : 0.0;
}

}  // namespace

extern "C" int rcupp_cuda_available(void) {
  return 0;
}

/* No CUDA resources in the CPU fallback: nothing to tear down. */
extern "C" void rcupp_pool_shutdown(void) {
}

extern "C" int rcupp_lasso_fista(const double* X, const double* y,
                                 int n, int p, double lambda,
                                 int max_iter, double tol,
                                 double* beta_out, int* iters_out,
                                 double* obj_out, int* converged_out) {
  if (X == NULL || y == NULL || beta_out == NULL || n <= 0 || p <= 0 ||
      lambda < 0.0 || max_iter <= 0 || !(tol > 0.0)) {
    return 1;
  }

  // Auxiliary (no-copy) views of R's column-major storage.
  arma::mat Xm(const_cast<double*>(X), n, p, false, true);
  arma::vec yv(const_cast<double*>(y), n, false, true);

  const double Lf = largest_eig_xtx(Xm, p);
  const double L = std::max(Lf / static_cast<double>(n), 1e-12);
  const double step = 1.0 / (L * static_cast<double>(n));  // = 1/Lf
  const double thresh = lambda / L;

  arma::vec beta(p); beta.zeros();
  arma::vec z(p);     z.zeros();
  double tk = 1.0;
  int iters = 0;
  int converged = 0;

  for (int it = 1; it <= max_iter; ++it) {
    const arma::vec r = yv - Xm * z;                 // n
    const arma::vec u = z + step * (Xm.t() * r);     // p
    arma::vec beta_new(p);
    for (int j = 0; j < p; ++j) {                    // soft-threshold
      const double a = std::fabs(u[j]) - thresh;
      beta_new[j] = a > 0.0 ? std::copysign(a, u[j]) : 0.0;
    }
    const double tnew = (1.0 + std::sqrt(1.0 + 4.0 * tk * tk)) / 2.0;
    const double mom = (tk - 1.0) / tnew;
    const arma::vec z_new = beta_new + mom * (beta_new - beta);

    const double diff = arma::norm(beta_new - beta, 2);
    const double denom = std::max(1.0, arma::norm(beta_new, 2));

    beta = beta_new;
    z = z_new;
    tk = tnew;
    iters = it;
    if (diff / denom < tol) {
      converged = 1;
      break;
    }
  }

  for (int j = 0; j < p; ++j) beta_out[j] = beta[j];

  const arma::vec resid = yv - Xm * beta;
  const double sse = arma::dot(resid, resid);
  const double l1 = arma::sum(arma::abs(beta));
  *obj_out = sse / (2.0 * static_cast<double>(n)) + lambda * l1;
  *iters_out = iters;
  *converged_out = converged;
  return 0;
}
