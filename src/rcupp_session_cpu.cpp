#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "rcupp_session.h"

/*
 * NO-CUDA (CPU fallback) implementation of the P1 SESSION C ABI
 * (rcupp_session.h). Same ABI and R-level behaviour as rcupp_session_cuda.cu,
 * so downstream code and tests are backend-agnostic. X/y are kept as host
 * copies; L is computed once in set_data(). Cold-start fit is BIT-IDENTICAL to
 * the one-shot cpu_stub.cpp solver (same Armadillo expressions, same objective
 * formula); warm-start reuses the previous solution.
 */

namespace {

/* Same deterministic power iteration as cpu_stub.cpp (clamps lf to >= 0). */
double largest_eig_xtx(const arma::mat& X, int p) {
  arma::vec v(p);
  for (int j = 0; j < p; ++j) v[j] = std::sin(static_cast<double>(j + 1));
  const double nv = arma::norm(v, 2);
  if (nv > 0.0) v /= nv;

  double lf = 0.0;
  for (int k = 0; k < 100; ++k) {
    const arma::vec u = X.t() * (X * v);       // X'X v
    const double rq = arma::dot(v, u);         // Rayleigh quotient
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

struct Session {
  bool has_data;
  bool has_solution;
  int n;
  int p;
  double L;
  double step;
  std::vector<double> h_X;       // column-major, n x p
  std::vector<double> h_y;
  std::vector<double> last_beta; // warm-start seed (previous solution)

  Session()
      : has_data(false), has_solution(false), n(0), p(0),
        L(0.0), step(0.0) {}
};

}  // namespace

extern "C" void* rcupp_session_create(void) {
  return static_cast<void*>(new Session());
}

extern "C" int rcupp_session_set_data(void* h, const double* X, const double* y,
                                      int n, int p) {
  if (h == NULL || X == NULL || y == NULL || n <= 0 || p <= 0) return 1;
  Session* s = static_cast<Session*>(h);

  const size_t szX = static_cast<size_t>(n) * static_cast<size_t>(p);
  s->h_X.assign(X, X + szX);
  s->h_y.assign(y, y + n);
  s->n = n;
  s->p = p;

  arma::mat Xm(s->h_X.data(), n, p, false, true);
  const double Lf = largest_eig_xtx(Xm, p);
  s->L = std::max(Lf / static_cast<double>(n), 1e-12);
  s->step = 1.0 / (s->L * static_cast<double>(n));

  s->has_data = true;
  s->has_solution = false;
  return 0;
}

extern "C" int rcupp_session_fit(void* h, double lambda, int warm_start,
                                 int max_iter, double tol,
                                 double* beta_out, int* iters_out,
                                 double* obj_out, int* converged_out) {
  if (h == NULL || beta_out == NULL || lambda < 0.0 || max_iter <= 0 ||
      !(tol > 0.0)) {
    return 1;
  }
  Session* s = static_cast<Session*>(h);
  if (!s->has_data) return 1;

  const int n = s->n, p = s->p;
  const double step = s->step;
  const double thresh = lambda / s->L;

  arma::mat Xm(s->h_X.data(), n, p, false, true);
  arma::vec yv(s->h_y.data(), n, false, true);

  arma::vec beta(p);
  arma::vec z(p);
  int min_iter = 1;
  if (warm_start && s->has_solution) {
    for (int j = 0; j < p; ++j) beta[j] = s->last_beta[j];
    z = beta;                 /* reset momentum: z = beta, tk = 1 */
    min_iter = 2;             /* don't converge on an already-good start */
  } else {
    beta.zeros();
    z.zeros();
  }

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
    if (it >= min_iter && diff / denom < tol) {
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

  s->last_beta.assign(p, 0.0);
  for (int j = 0; j < p; ++j) s->last_beta[j] = beta[j];
  s->has_solution = true;
  return 0;
}

extern "C" void rcupp_session_destroy(void* h) {
  delete static_cast<Session*>(h);
}
