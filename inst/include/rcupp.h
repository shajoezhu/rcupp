#ifndef RCUPP_H
#define RCUPP_H

/*
 * Public C API of rcupp -- the "CUDA-for-Rcpp" framework entry point.
 *
 * A downstream package uses it with:
 *   DESCRIPTION:  LinkingTo: rcupp, Rcpp
 *                 Imports: rcupp
 *   src/foo.cpp:  #include <rcupp.h>
 *
 * `LinkingTo: rcupp` adds -I<rcupp>/include automatically (no src/Makevars
 * needed); the wrappers below resolve the real implementations at runtime via
 * R_GetCCallable(). rcupp registers them at DLL load (R_init_rcupp, see
 * rcupp_exports.cpp). There is NO link-time dependency on rcupp's shared
 * library -- just make sure the rcupp namespace is loaded before the first
 * call (Imports: rcupp guarantees this).
 *
 * Design:
 *   - Plain C/C++ only: pulls in <R_ext/Rdynload.h>, leaks NO CUDA, Rcpp or
 *     STL types across the package boundary.
 *   - All pointers are HOST double*; X is n x p, column-major (R order).
 *   - The rcupp_* operators are one-shot (each call copies H->D->H). For
 *     persistent device residency / a reusable cuBLAS handle across many fits,
 *     use the P1 SESSION C ABI below (rcupp_session_*).
 *   - `static inline` + a cached function pointer: safe to include from many
 *     translation units, no duplicate symbols, one lazy lookup per TU.
 *
 * See docs/linking-to-rcupp.md for the reusable downstream build recipe.
 */

#include <R_ext/Rdynload.h>

/* 1 when a CUDA backend is linked into rcupp, else 0 (CPU fallback). */
static inline int rcupp_cuda_available(void) {
  typedef int (*fn_t)(void);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_cuda_available");
  return fun();
}

/* out[n] = X(n x p, column-major) * v(p) */
static inline void rcupp_gemv(const double* X, int n, int p,
                               const double* v, double* out) {
  typedef void (*fn_t)(const double*, int, int, const double*, double*);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_gemv");
  fun(X, n, p, v, out);
}

/* out[p] = t(X)(n x p, column-major) * r(n) */
static inline void rcupp_gemv_t(const double* X, int n, int p,
                                 const double* r, double* out) {
  typedef void (*fn_t)(const double*, int, int, const double*, double*);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_gemv_t");
  fun(X, n, p, r, out);
}

/* y[i] += alpha * x[i]  (length n, in place) */
static inline void rcupp_axpy(int n, double alpha, const double* x, double* y) {
  typedef void (*fn_t)(int, double, const double*, double*);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_axpy");
  fun(n, alpha, x, y);
}

/* x[i] *= alpha  (length n, in place) */
static inline void rcupp_scal(int n, double alpha, double* x) {
  typedef void (*fn_t)(int, double, double*);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_scal");
  fun(n, alpha, x);
}

/* sum(a[i] * b[i]) */
static inline double rcupp_dot(int n, const double* a, const double* b) {
  typedef double (*fn_t)(int, const double*, const double*);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_dot");
  return fun(n, a, b);
}

/* sqrt(sum(x[i]^2)) */
static inline double rcupp_nrm2(int n, const double* x) {
  typedef double (*fn_t)(int, const double*);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_nrm2");
  return fun(n, x);
}

/* out[i] = sign(u[i]) * max(|u[i]| - t, 0)  (soft threshold / lasso prox) */
static inline void rcupp_soft_threshold(const double* u, double t,
                                         double* out, int n) {
  typedef void (*fn_t)(const double*, double, double*, int);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_soft_threshold");
  fun(u, t, out, n);
}

/* ---------------------------------------------------------------------------
 * P1 SESSION C ABI: persistent device-X / cuBLAS-handle reuse across fits.
 *
 *   void* h = rcupp_session_create();
 *   rcupp_session_set_data(h, X, y, n, p);   (upload X/y + cache L once)
 *   double beta[p]; int iters, conv; double obj;
 *   for each lambda:
 *     rcupp_session_fit(h, lambda, 0, 1000, 1e-6,
 *                       beta, &iters, &obj, &conv);  (warm_start = 0 = cold)
 *   rcupp_session_destroy(h);
 *
 * warm_start = 1 seeds the fit from the previous solution (converges to the
 * same optimum, NOT bit-identical); warm_start = 0 is a cold start. All
 * pointers are HOST double*, X is n x p column-major (R order). The handle is
 * opaque; treat it as owned by you until rcupp_session_destroy. Backend-
 * agnostic: the CPU fallback implements the same ABI.
 * ------------------------------------------------------------------------- */
static inline void* rcupp_session_create(void) {
  typedef void* (*fn_t)(void);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_session_create");
  return fun();
}

static inline int rcupp_session_set_data(void* h, const double* X,
                                         const double* y, int n, int p) {
  typedef int (*fn_t)(void*, const double*, const double*, int, int);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_session_set_data");
  return fun(h, X, y, n, p);
}

static inline int rcupp_session_fit(void* h, double lambda, int warm_start,
                                    int max_iter, double tol,
                                    double* beta_out, int* iters_out,
                                    double* obj_out, int* converged_out) {
  typedef int (*fn_t)(void*, double, int, int, double,
                      double*, int*, double*, int*);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_session_fit");
  return fun(h, lambda, warm_start, max_iter, tol,
             beta_out, iters_out, obj_out, converged_out);
}

static inline void rcupp_session_destroy(void* h) {
  typedef void (*fn_t)(void*);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_session_destroy");
  fun(h);
}

#endif /* RCUPP_H */
