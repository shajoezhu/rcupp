#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "rcupp_session.h"
#include "gpu_ops.cuh"

/*
 * CUDA implementation of the P1 SESSION C ABI (rcupp_session.h).
 *
 * A session owns its OWN cuBLAS handle and its OWN 14 device work buffers --
 * it is deliberately independent from the process-global one-shot pool in
 * lasso_fista_cuda.cu and from the F0 one-shot public operators. X/y are
 * uploaded ONCE in set_data() and stay resident on the device for the whole
 * life of the session, so fitting many lambdas (a glmnet lambda path, CV
 * folds, ...) no longer re-uploads X, recreates the handle, reallocates
 * buffers or recomputes L on every fit.
 *
 * Numerical contract (identical to rcupp_lasso_fista):
 *   F(beta) = 1/(2n) ||y - X beta||^2 + lambda ||beta||_1
 *   X is n x p, COLUMN-MAJOR (R order).
 * Cold-start fit (warm_start = 0) is BIT-IDENTICAL to a one-shot
 * rcupp_lasso_fista() call for the same X/y/lambda: L is computed by the SAME
 * deterministic power iteration, the FISTA loop is the SAME op_* sequence in
 * the SAME buffer layout, and the objective uses the SAME host-side formula
 * over a host mirror of X/y. Warm-start fit reuses the previous solution as
 * its starting point (converges to the same optimum, not bit-identical).
 *
 * NO new __global__ kernel is introduced: everything is expressed with the
 * existing fixed op_* primitives (op_soft_threshold included).
 */

namespace {

struct Session {
  cublasHandle_t handle;
  bool handle_ok;
  bool has_data;       /* set_data() succeeded; fit() is allowed. */
  bool has_solution;   /* at least one fit done -> d_beta is a warm-start seed. */
  int n;
  int p;
  double L;            /* cached Lipschitz constant = lambda_max(X'X)/n. */
  double step;         /* = 1 / (L * n). */
  std::vector<double> h_X;  /* host mirror for the objective (bit-identical). */
  std::vector<double> h_y;
  DeviceBuffer X, y, beta, z, beta_new, z_new, u, r, Xz, grad, tmp, v, Au, w;

  Session()
      : handle(), handle_ok(false), has_data(false), has_solution(false),
        n(0), p(0), L(0.0), step(0.0),
        X(0), y(0), beta(0), z(0), beta_new(0), z_new(0), u(0), r(0),
        Xz(0), grad(0), tmp(0), v(0), Au(0), w(0) {}
};

bool buffers_ok(const Session& s) {
  return s.X.valid() && s.y.valid() && s.beta.valid() && s.z.valid() &&
         s.beta_new.valid() && s.z_new.valid() && s.u.valid() &&
         s.r.valid() && s.Xz.valid() && s.grad.valid() && s.tmp.valid() &&
         s.v.valid() && s.Au.valid() && s.w.valid();
}

/* Deterministic power iteration for lambda_max(X'X), identical to
 * lasso_fista_cuda.cu and cpu_stub.cpp. Start vector v[j] = sin(j+1). */
double largest_eig_xtx(Session& s) {
  const int n = s.n, p = s.p;
  std::vector<double> v0(p);
  double nv = 0.0;
  for (int j = 0; j < p; ++j) {
    v0[j] = std::sin(static_cast<double>(j + 1));
    nv += v0[j] * v0[j];
  }
  nv = std::sqrt(nv);
  if (nv > 0.0) {
    for (int j = 0; j < p; ++j) v0[j] /= nv;
  }
  cudaMemcpy(s.v.get(), v0.data(), p * sizeof(double), cudaMemcpyHostToDevice);

  double Lf = 0.0;
  for (int k = 0; k < 100; ++k) {
    op_gemv(s.handle, s.X.get(), n, p, s.v.get(), s.w.get());       /* w = X v  */
    op_gemv_t(s.handle, s.X.get(), n, p, s.w.get(), s.Au.get());   /* u = X'X v */
    const double rq = op_dot(s.handle, p, s.v.get(), s.Au.get());   /* Rayleigh  */
    const double nu = op_nrm2(s.handle, p, s.Au.get());
    Lf = rq;
    if (nu > 0.0) {
      op_copy_scale(s.handle, p, s.Au.get(), 1.0 / nu, s.v.get());
    } else {
      break;
    }
  }
  return Lf;
}

}  // namespace

extern "C" void* rcupp_session_create(void) {
  Session* s = new Session();
  if (cublasCreate(&s->handle) != CUBLAS_STATUS_SUCCESS) {
    delete s;
    return NULL;
  }
  s->handle_ok = true;
  return static_cast<void*>(s);
}

extern "C" int rcupp_session_set_data(void* h, const double* X, const double* y,
                                      int n, int p) {
  if (h == NULL || X == NULL || y == NULL || n <= 0 || p <= 0) return 1;
  Session* s = static_cast<Session*>(h);
  if (!s->handle_ok) return 1;

  const size_t szX = static_cast<size_t>(n) * static_cast<size_t>(p);
  /* (Re)allocate every work buffer for this (n, p). reset() frees the old
   * block first, so a session can be re-bound to new data. */
  s->X.reset(szX);
  s->y.reset(n);
  s->beta.reset(p);
  s->z.reset(p);
  s->beta_new.reset(p);
  s->z_new.reset(p);
  s->u.reset(p);
  s->r.reset(n);
  s->Xz.reset(n);
  s->grad.reset(p);
  s->tmp.reset(p);
  s->v.reset(p);
  s->Au.reset(p);
  s->w.reset(n);
  if (!buffers_ok(*s)) return 1;

  /* Host mirror for the objective (must be bit-identical to one-shot). */
  s->h_X.assign(X, X + szX);
  s->h_y.assign(y, y + n);

  /* H -> D exactly once, for the whole session. */
  cudaMemcpy(s->X.get(), X, szX * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(s->y.get(), y, n * sizeof(double), cudaMemcpyHostToDevice);

  s->n = n;
  s->p = p;
  const double Lf = largest_eig_xtx(*s);
  s->L = std::max(Lf / static_cast<double>(n), 1e-12);
  s->step = 1.0 / (s->L * static_cast<double>(n));

  s->has_data = true;
  s->has_solution = false;  /* new data invalidates any warm-start seed. */
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
  if (!s->has_data || !s->handle_ok) return 1;

  const int n = s->n, p = s->p;
  const double step = s->step;
  const double thresh = lambda / s->L;

  /* Cold start: beta=z=0, tk=1 -> bit-identical to one-shot. Warm start: seed
   * from the previous solution (z=beta, tk=1), and require >= 2 iterations so
   * the convergence test cannot fire prematurely on an already-good start. */
  int min_iter = 1;
  if (warm_start && s->has_solution) {
    op_copy_scale(s->handle, p, s->beta.get(), 1.0, s->z.get());
    min_iter = 2;
  } else {
    cudaMemset(s->beta.get(), 0, p * sizeof(double));
    cudaMemset(s->z.get(), 0, p * sizeof(double));
  }

  double tk = 1.0;
  int iters = 0;
  int converged = 0;

  for (int it = 1; it <= max_iter; ++it) {
    op_gemv(s->handle, s->X.get(), n, p, s->z.get(), s->Xz.get());       /* X z      */
    op_diff(s->handle, n, s->y.get(), s->Xz.get(), s->r.get());          /* y - X z  */
    op_gemv_t(s->handle, s->X.get(), n, p, s->r.get(), s->grad.get());   /* X'(y-Xz) */
    op_copy_scale(s->handle, p, s->z.get(), 1.0, s->u.get());             /* u = z    */
    op_axpy(s->handle, p, step, s->grad.get(), s->u.get());               /* +=step*g */
    op_soft_threshold(s->u.get(), thresh, s->beta_new.get(), p);

    const double tnew = (1.0 + std::sqrt(1.0 + 4.0 * tk * tk)) / 2.0;
    const double mom = (tk - 1.0) / tnew;
    op_diff(s->handle, p, s->beta_new.get(), s->beta.get(), s->tmp.get()); /* b+-b */
    op_copy_scale(s->handle, p, s->beta_new.get(), 1.0, s->z_new.get());
    op_axpy(s->handle, p, mom, s->tmp.get(), s->z_new.get());

    const double diff = op_nrm2(s->handle, p, s->tmp.get());
    const double denom = std::max(1.0, op_nrm2(s->handle, p, s->beta_new.get()));

    op_copy_scale(s->handle, p, s->beta_new.get(), 1.0, s->beta.get());    /* b <- b+ */
    op_copy_scale(s->handle, p, s->z_new.get(), 1.0, s->z.get());          /* z <- z+ */
    tk = tnew;
    iters = it;
    if (it >= min_iter && diff / denom < tol) {
      converged = 1;
      break;
    }
  }

  /* D -> H exactly once, then the objective on the host from the mirror --
   * identical formula to rcupp_lasso_fista / cpu_stub. */
  cudaMemcpy(beta_out, s->beta.get(), p * sizeof(double), cudaMemcpyDeviceToHost);

  double sse = 0.0;
  double l1 = 0.0;
  for (int i = 0; i < n; ++i) {
    double xi = 0.0;
    for (int j = 0; j < p; ++j) {
      xi += s->h_X[static_cast<size_t>(j) * n + i] * beta_out[j];
    }
    const double e = s->h_y[i] - xi;
    sse += e * e;
  }
  for (int j = 0; j < p; ++j) l1 += std::fabs(beta_out[j]);
  *obj_out = sse / (2.0 * static_cast<double>(n)) + lambda * l1;
  *iters_out = iters;
  *converged_out = converged;

  s->has_solution = true;  /* d_beta now holds the warm-start seed. */
  return 0;
}

extern "C" void rcupp_session_destroy(void* h) {
  if (h == NULL) return;
  Session* s = static_cast<Session*>(h);
  if (s->handle_ok) cublasDestroy(s->handle);
  delete s;  /* DeviceBuffer destructors cudaFree each buffer. */
}
