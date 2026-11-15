#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "rcupp_internal.h"
#include "gpu_ops.cuh"

/*
 * GPU FISTA solver for the Gaussian lasso -- implements the shared C ABI.
 *
 * Numerical contract is IDENTICAL, line for line, to:
 *   ../cpu_stub.cpp                      (CPU reference shim)
 *   ../../cudaGlmnet/R/lasso_fista_cpu.R (pure-R numerical oracle)
 * so results can be compared directly.
 *
 * X, y, beta, z and all work buffers stay on the device for the whole loop:
 * one H->D copy at the start, one D->H copy at the end. That residency is the
 * entire reason for this low-level package.
 *
 * P1 (transparent one-shot pool): the cuBLAS handle and the 14 device work
 * buffers are kept in a PROCESS-GLOBAL pool and reused ACROSS calls, so a
 * stream of rcupp_lasso_fista() calls no longer pays cublasCreate/Destroy plus
 * 14 cudaMalloc/cudaFree each time. Every call STILL re-uploads X/y, zeroes
 * beta/z and reruns the deterministic power iteration for L, and every reused
 * buffer is fully overwritten before it is read, so the numerics are
 * BIT-IDENTICAL to the original per-call-allocation implementation. This is
 * deliberately independent from the explicit P1 session API (see
 * rcupp_session_cuda.cu), which instead keeps X/y resident across fits.
 */

namespace {

/*
 * OneShotPool owns CUDA resources that must be released at a CONTROLLED time.
 * It is held through a deliberately LEAKED naked pointer (never a static
 * object with a destructor): at process exit the CUDA context may already be
 * gone, and a static destructor calling cudaFree/cublasDestroy would crash.
 * rcupp_pool_shutdown() (invoked from .onUnload) frees everything before the
 * DLL is unloaded; if it is never called, the pointer simply leaks at exit,
 * which is harmless (the OS reclaims memory, no CUDA calls in any destructor).
 */
struct OneShotPool {
  cublasHandle_t handle;
  bool handle_ok;
  int cur_n;
  int cur_p;
  DeviceBuffer X, y, beta, z, beta_new, z_new, u, r, Xz, grad, tmp, v, Au, w;

  OneShotPool()
      : handle(), handle_ok(false), cur_n(0), cur_p(0),
        X(0), y(0), beta(0), z(0), beta_new(0), z_new(0), u(0), r(0),
        Xz(0), grad(0), tmp(0), v(0), Au(0), w(0) {}

  bool buffers_ok() const {
    return X.valid() && y.valid() && beta.valid() && z.valid() &&
           beta_new.valid() && z_new.valid() && u.valid() && r.valid() &&
           Xz.valid() && grad.valid() && tmp.valid() && v.valid() &&
           Au.valid() && w.valid();
  }

  /* Lazily create the handle, and (re)allocate all 14 buffers only when the
   * (n, p) shape changes. cudaMalloc failures leave valid() == false. */
  bool ensure(int n, int p) {
    if (!handle_ok) {
      if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
      handle_ok = true;
    }
    if (n != cur_n || p != cur_p) {
      const size_t szX = static_cast<size_t>(n) * static_cast<size_t>(p);
      X.reset(szX);
      y.reset(n);
      beta.reset(p);
      z.reset(p);
      beta_new.reset(p);
      z_new.reset(p);
      u.reset(p);
      r.reset(n);
      Xz.reset(n);
      grad.reset(p);
      tmp.reset(p);
      v.reset(p);
      Au.reset(p);
      w.reset(n);
      if (!buffers_ok()) return false;
      cur_n = n;
      cur_p = p;
    }
    return buffers_ok();
  }
};

/* Deliberately leaked; freed only by rcupp_pool_shutdown() at unload. */
OneShotPool* g_pool = NULL;

OneShotPool& pool() {
  if (!g_pool) g_pool = new OneShotPool();
  return *g_pool;
}

}  // namespace

extern "C" void rcupp_pool_shutdown(void) {
  if (g_pool) {
    if (g_pool->handle_ok) cublasDestroy(g_pool->handle);
    delete g_pool;  /* DeviceBuffer destructors cudaFree each buffer. */
    g_pool = NULL;
  }
}

extern "C" int rcupp_cuda_available(void) {
  int count = 0;
  const cudaError_t e = cudaGetDeviceCount(&count);
  return (e == cudaSuccess && count > 0) ? 1 : 0;
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

  OneShotPool& P = pool();
  if (!P.ensure(n, p)) return 1;
  cublasHandle_t h = P.handle;

  /* Bind short local names to the pooled buffers. Everything below is the
   * ORIGINAL FISTA code verbatim -- only buffer OWNERSHIP changed from local
   * DeviceBuffers to references into the process-global pool. */
  DeviceBuffer& d_X = P.X;
  DeviceBuffer& d_y = P.y;
  DeviceBuffer& d_beta = P.beta;
  DeviceBuffer& d_z = P.z;
  DeviceBuffer& d_beta_new = P.beta_new;
  DeviceBuffer& d_z_new = P.z_new;
  DeviceBuffer& d_u = P.u;
  DeviceBuffer& d_r = P.r;
  DeviceBuffer& d_Xz = P.Xz;
  DeviceBuffer& d_grad = P.grad;
  DeviceBuffer& d_tmp = P.tmp;
  DeviceBuffer& d_v = P.v;
  DeviceBuffer& d_Au = P.Au;
  DeviceBuffer& d_w = P.w;

  /* Redundant with ensure(), kept as a cheap defensive check (no shared-handle
   * teardown here: a failure must NOT cublasDestroy the pooled handle). */
  const bool ok = d_X.valid() && d_y.valid() && d_beta.valid() &&
                  d_z.valid() && d_beta_new.valid() && d_z_new.valid() &&
                  d_u.valid() && d_r.valid() && d_Xz.valid() &&
                  d_grad.valid() && d_tmp.valid() && d_v.valid() &&
                  d_Au.valid() && d_w.valid();
  if (!ok) return 1;

  const size_t szX = static_cast<size_t>(n) * static_cast<size_t>(p);

  /* H -> D exactly once. */
  cudaMemcpy(d_X.get(), X, szX * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y.get(), y, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(d_beta.get(), 0, p * sizeof(double));
  cudaMemset(d_z.get(), 0, p * sizeof(double));

  /* L = lambda_max(X'X)/n by deterministic power iteration.
   * Start vector v[j] = sin(j+1), j = 0..p-1 -- identical in all three impls. */
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
  cudaMemcpy(d_v.get(), v0.data(), p * sizeof(double), cudaMemcpyHostToDevice);

  double Lf = 0.0;
  for (int k = 0; k < 100; ++k) {
    op_gemv(h, d_X.get(), n, p, d_v.get(), d_w.get());       /* w = X v        */
    op_gemv_t(h, d_X.get(), n, p, d_w.get(), d_Au.get());   /* u = X'X v      */
    const double rq = op_dot(h, p, d_v.get(), d_Au.get());   /* Rayleigh       */
    const double nu = op_nrm2(h, p, d_Au.get());
    Lf = rq;
    if (nu > 0.0) {
      op_copy_scale(h, p, d_Au.get(), 1.0 / nu, d_v.get());
    } else {
      break;
    }
  }

  const double L = std::max(Lf / static_cast<double>(n), 1e-12);
  const double step = 1.0 / (L * static_cast<double>(n));  /* = 1/Lf         */
  const double thresh = lambda / L;

  double tk = 1.0;
  int iters = 0;
  int converged = 0;

  for (int it = 1; it <= max_iter; ++it) {
    op_gemv(h, d_X.get(), n, p, d_z.get(), d_Xz.get());          /* X z        */
    op_diff(h, n, d_y.get(), d_Xz.get(), d_r.get());              /* y - X z    */
    op_gemv_t(h, d_X.get(), n, p, d_r.get(), d_grad.get());      /* X'(y-Xz)   */
    op_copy_scale(h, p, d_z.get(), 1.0, d_u.get());               /* u = z      */
    op_axpy(h, p, step, d_grad.get(), d_u.get());                  /* +=step*grad*/
    op_soft_threshold(d_u.get(), thresh, d_beta_new.get(), p);

    const double tnew = (1.0 + std::sqrt(1.0 + 4.0 * tk * tk)) / 2.0;
    const double mom = (tk - 1.0) / tnew;
    op_diff(h, p, d_beta_new.get(), d_beta.get(), d_tmp.get());   /* b+ - b     */
    op_copy_scale(h, p, d_beta_new.get(), 1.0, d_z_new.get());
    op_axpy(h, p, mom, d_tmp.get(), d_z_new.get());

    const double diff = op_nrm2(h, p, d_tmp.get());
    const double denom = std::max(1.0, op_nrm2(h, p, d_beta_new.get()));

    op_copy_scale(h, p, d_beta_new.get(), 1.0, d_beta.get());      /* b <- b+    */
    op_copy_scale(h, p, d_z_new.get(), 1.0, d_z.get());            /* z <- z+    */
    tk = tnew;
    iters = it;
    if (diff / denom < tol) {
      converged = 1;
      break;
    }
  }

  /* D -> H exactly once. NOTE: no cublasDestroy(h) here -- the handle belongs
   * to the pool and is destroyed by rcupp_pool_shutdown() at unload. */
  cudaMemcpy(beta_out, d_beta.get(), p * sizeof(double), cudaMemcpyDeviceToHost);

  /* Objective on the host (O(np), once) -- same formula in all three impls. */
  double sse = 0.0;
  double l1 = 0.0;
  for (int i = 0; i < n; ++i) {
    double xi = 0.0;
    for (int j = 0; j < p; ++j) {
      xi += X[static_cast<size_t>(j) * n + i] * beta_out[j];
    }
    const double e = y[i] - xi;
    sse += e * e;
  }
  for (int j = 0; j < p; ++j) l1 += std::fabs(beta_out[j]);
  *obj_out = sse / (2.0 * static_cast<double>(n)) + lambda * l1;
  *iters_out = iters;
  *converged_out = converged;

  return 0;
}
