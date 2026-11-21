#ifndef RCUPP_PARTS_IMPL_H
#define RCUPP_PARTS_IMPL_H

/*
 * Level-2 composite parts of the Phase 0 device-context C ABI, implemented
 * ONCE and included by BOTH backends (rcupp_ctx_cuda.cu / rcupp_ctx_cpu.cpp).
 *
 * These compose ONLY the level-1 resident operators declared in rcupp_ctx.h
 * (gemv / gemv_t / axpy / scal / copy / dot / soft_threshold). Because those
 * operators are already backend-agnostic, this shared host code runs
 * unchanged on the CUDA backend and the CPU fallback: every buffer stays
 * resident, only the small scalar reductions (dot) cross to the host.
 *
 * Each backend TU includes this header and exposes thin extern "C" entry
 * points that call these helpers. Scratch buffers are ctx-owned and obtained
 * via rcupp_buf_alloc()/rcupp_buf_free(), so the public signatures stay small.
 * NOTE: the CUDA backend overrides rcupp_d_prox_grad_step with a SINGLE fused
 * kernel (ctx_prox_grad_fused_kernel in rcupp_ctx_cuda.cu) that folds a whole
 * iteration into one launch; the implementation below remains the CPU fallback
 * and the arithmetic reference. Other hot loops that need momentum / custom
 * schedules still compose level-1 ops directly (cudaGlmnet/src/enet_path.cpp,
 * and cudaFin P3-fin whose inner step now calls the fused part).
 *
 * This is CUDA-free, Rcpp-free, STL-light plain C++: safe for nvcc AND g++.
 */

#include <cmath>
#include "rcupp_ctx.h"

/* Solve (X^T X + shift*I) x = X^T y by conjugate gradients.
 *
 * Let A(v) = X^T (X v) + shift*v. Standard CG on the SPD system A x = b with
 * b = X^T y, all in the p-dimensional coefficient space:
 *   r = b - A(x); dir = r; rsold = <r,r>
 *   loop: Ap = A(dir); a = rsold / <dir,Ap>; x += a dir; r -= a Ap
 *         if ||r|| <= tol -> converged; beta = <r,r>/rsold;
 *         dir = r + beta dir; rsold = <r,r>
 * shift > 0 makes A positive definite regardless of X's rank (X^T X is PSD).
 * "converged" is reported through the out-arg, not the return code. */
static inline int rcupp_cg_solve_impl(void* c, void* X, int n, int p,
                                      void* y, double shift, double tol,
                                      int max_iter, void* x, int* iters,
                                      double* residual, int* converged) {
  if (iters) *iters = 0;
  if (residual) *residual = 0.0;
  if (converged) *converged = 0;
  if (c == NULL || X == NULL || y == NULL || x == NULL || n <= 0 || p <= 0 ||
      max_iter < 1 || !(tol > 0.0) || !(shift >= 0.0)) return 1;

  /* scratch: b(p), tmpn(n), Ap(p), r(p), dir(p) -- written before every read */
  void* b    = rcupp_buf_alloc(c, p);
  void* tmpn = rcupp_buf_alloc(c, n);
  void* Ap   = rcupp_buf_alloc(c, p);
  void* r    = rcupp_buf_alloc(c, p);
  void* dir  = rcupp_buf_alloc(c, p);
  if (b == NULL || tmpn == NULL || Ap == NULL || r == NULL || dir == NULL) {
    rcupp_buf_free(c, b);
    rcupp_buf_free(c, tmpn);
    rcupp_buf_free(c, Ap);
    rcupp_buf_free(c, r);
    rcupp_buf_free(c, dir);
    return 1;
  }

  rcupp_d_gemv_t(c, X, n, p, y, b);              /* b = X^T y            */
  rcupp_d_gemv(c, X, n, p, x, tmpn);             /* tmpn = X x           */
  rcupp_d_gemv_t(c, X, n, p, tmpn, Ap);          /* Ap = X^T (X x)       */
  rcupp_d_axpy(c, p, shift, x, Ap);              /* Ap += shift x        */
  rcupp_d_copy(c, p, b, r);
  rcupp_d_axpy(c, p, -1.0, Ap, r);               /* r = b - A(x)         */
  rcupp_d_copy(c, p, r, dir);                    /* dir = r              */
  double rsold = rcupp_d_dot(c, p, r, r);
  double rsnew = rsold;
  int it = 0;
  int conv = 0;
  if (std::sqrt(rsold) <= tol) {
    conv = 1;                                    /* initial guess already solves */
  } else {
    for (it = 1; it <= max_iter; ++it) {
      rcupp_d_gemv(c, X, n, p, dir, tmpn);       /* tmpn = X dir         */
      rcupp_d_gemv_t(c, X, n, p, tmpn, Ap);      /* Ap = X^T (X dir)     */
      rcupp_d_axpy(c, p, shift, dir, Ap);        /* Ap += shift dir      */
      const double dap = rcupp_d_dot(c, p, dir, Ap);
      if (!(dap > 0.0)) break;                   /* breakdown (not SPD)  */
      const double a = rsold / dap;
      rcupp_d_axpy(c, p, a, dir, x);             /* x += a dir           */
      rcupp_d_axpy(c, p, -a, Ap, r);             /* r -= a Ap            */
      rsnew = rcupp_d_dot(c, p, r, r);
      if (std::sqrt(rsnew) <= tol) { conv = 1; break; }
      const double beta = rsnew / rsold;
      rcupp_d_scal(c, p, beta, dir);             /* dir *= beta          */
      rcupp_d_axpy(c, p, 1.0, r, dir);           /* dir = r + beta dir   */
      rsold = rsnew;
    }
  }

  if (iters) *iters = it > max_iter ? max_iter : it;
  if (residual) *residual = std::sqrt(rsnew);
  if (converged) *converged = conv;

  rcupp_buf_free(c, dir);
  rcupp_buf_free(c, r);
  rcupp_buf_free(c, Ap);
  rcupp_buf_free(c, tmpn);
  rcupp_buf_free(c, b);
  return 0;
}

/* ONE proximal-gradient (ISTA) step (CPU fallback + arithmetic reference):
 *   out = soft_threshold( z + step * X^T (y - X z), threshold ) * inv_scale
 * This reproduces exactly the inner step of cudaGlmnet/src/enet_path.cpp
 * (lines 113-122) but folds it into a single C-ABI call, with device-resident
 * scratch. On CUDA, rcupp_ctx_cuda.cu overrides this with a single fused
 * kernel (same arithmetic, one launch, no per-call scratch alloc); the CPU
 * backend still runs this level-1 composition, bit-identical. The lambda path,
 * FISTA momentum and convergence stay DOWNSTREAM. z/out aliasing is safe. */
static inline int rcupp_prox_grad_step_impl(void* c, void* X, int n, int p,
                                            void* y, void* z, double step,
                                            double threshold,
                                            double inv_scale, void* out) {
  if (c == NULL || X == NULL || y == NULL || z == NULL || out == NULL ||
      n <= 0 || p <= 0) return 1;

  /* scratch: Xb(n), r(n), g(p), u(p) -- written before every read */
  void* Xb = rcupp_buf_alloc(c, n);
  void* r  = rcupp_buf_alloc(c, n);
  void* g  = rcupp_buf_alloc(c, p);
  void* u  = rcupp_buf_alloc(c, p);
  if (Xb == NULL || r == NULL || g == NULL || u == NULL) {
    rcupp_buf_free(c, Xb);
    rcupp_buf_free(c, r);
    rcupp_buf_free(c, g);
    rcupp_buf_free(c, u);
    return 1;
  }

  rcupp_d_gemv(c, X, n, p, z, Xb);               /* Xb = X z            */
  rcupp_d_copy(c, n, y, r);
  rcupp_d_axpy(c, n, -1.0, Xb, r);               /* r = y - X z         */
  rcupp_d_scal(c, n, step, r);                   /* r *= step           */
  rcupp_d_gemv_t(c, X, n, p, r, g);              /* g = X^T r           */
  rcupp_d_copy(c, p, z, u);
  rcupp_d_axpy(c, p, 1.0, g, u);                 /* u = z + g           */
  rcupp_d_soft_threshold(c, u, threshold, out, p);
  rcupp_d_scal(c, p, inv_scale, out);            /* out *= inv_scale     */

  rcupp_buf_free(c, u);
  rcupp_buf_free(c, g);
  rcupp_buf_free(c, r);
  rcupp_buf_free(c, Xb);
  return 0;
}

#endif /* RCUPP_PARTS_IMPL_H */
