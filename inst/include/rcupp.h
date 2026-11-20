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

/* ===========================================================================
 * Phase 0: generic device-context C ABI (ABI version 1)
 *
 * An algorithm-agnostic, RESIDENT device layer. A downstream package writes
 * its own solver loop in C++ (Rcpp glue) and composes these operators; rcupp
 * never learns what lasso / ridge / PCA is. All symbols are resolved lazily
 * via R_GetCCallable() and registered by rcupp at DLL load.
 *
 * rcupp_ctx owns the CUDA/cuBLAS handle + a buffer registry. Every rcupp_buf
 * belongs to a ctx and is freed at rcupp_ctx_destroy(). All rcupp_d_* ops run
 * on resident buffers with NO host<->device round trip. X is n x p,
 * column-major (R order), matching the F0 operators exactly.
 *
 * This sits ALONGSIDE the one-shot F0 operators and the lasso-specific P1
 * session above; those remain unchanged. Use this layer when you want to keep
 * data resident across many operator calls (e.g. an ISTA / power-iteration
 * inner loop) instead of copying H->D->H on every call.
 * ======================================================================== */
#ifndef RCUPP_CTX_TYPEDEFS
#define RCUPP_CTX_TYPEDEFS
typedef void* rcupp_ctx;
typedef void* rcupp_buf;
#endif

/* Which triangle a symmetric-rank-k / Cholesky operator reads or writes.
 * Values match cublasFillMode_t (LOWER=0, UPPER=1) on the CUDA backend. */
#ifndef RCUPP_FILL_LOWER
#define RCUPP_FILL_LOWER 0
#define RCUPP_FILL_UPPER 1
#endif

/* ---- ABI version ---- */
static inline int rcupp_ctx_abi(void) {
  typedef int (*fn_t)(void);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_ctx_abi");
  return fun();
}

/* ---- context lifecycle ---- */
static inline rcupp_ctx rcupp_ctx_create(void) {
  typedef rcupp_ctx (*fn_t)(void);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_ctx_create");
  return fun();
}
static inline void rcupp_ctx_destroy(rcupp_ctx c) {
  typedef void (*fn_t)(rcupp_ctx);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_ctx_destroy");
  fun(c);
}
static inline void rcupp_ctx_sync(rcupp_ctx c) {
  typedef void (*fn_t)(rcupp_ctx);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_ctx_sync");
  fun(c);
}

/* ---- buffer lifecycle (owned by ctx) ---- */
static inline rcupp_buf rcupp_buf_alloc(rcupp_ctx c, int n) {
  typedef rcupp_buf (*fn_t)(rcupp_ctx, int);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_buf_alloc");
  return fun(c, n);
}
static inline void rcupp_buf_free(rcupp_ctx c, rcupp_buf b) {
  typedef void (*fn_t)(rcupp_ctx, rcupp_buf);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_buf_free");
  fun(c, b);
}
static inline int rcupp_buf_upload(rcupp_ctx c, rcupp_buf b,
                                   const double* host, int n) {
  typedef int (*fn_t)(rcupp_ctx, rcupp_buf, const double*, int);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_buf_upload");
  return fun(c, b, host, n);
}
static inline int rcupp_buf_download(rcupp_ctx c, rcupp_buf b,
                                     double* host, int n) {
  typedef int (*fn_t)(rcupp_ctx, rcupp_buf, double*, int);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_buf_download");
  return fun(c, b, host, n);
}
static inline int rcupp_buf_fill(rcupp_ctx c, rcupp_buf b,
                                 double value, int n) {
  typedef int (*fn_t)(rcupp_ctx, rcupp_buf, double, int);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_buf_fill");
  return fun(c, b, value, n);
}

/* ---- resident device operators ---- */
static inline int rcupp_d_gemv(rcupp_ctx c, rcupp_buf X, int n, int p,
                               rcupp_buf v, rcupp_buf out) {
  typedef int (*fn_t)(rcupp_ctx, rcupp_buf, int, int, rcupp_buf, rcupp_buf);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_gemv");
  return fun(c, X, n, p, v, out);
}
static inline int rcupp_d_gemv_t(rcupp_ctx c, rcupp_buf X, int n, int p,
                                 rcupp_buf r, rcupp_buf out) {
  typedef int (*fn_t)(rcupp_ctx, rcupp_buf, int, int, rcupp_buf, rcupp_buf);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_gemv_t");
  return fun(c, X, n, p, r, out);
}
static inline int rcupp_d_axpy(rcupp_ctx c, int n, double alpha,
                               rcupp_buf x, rcupp_buf y) {
  typedef int (*fn_t)(rcupp_ctx, int, double, rcupp_buf, rcupp_buf);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_axpy");
  return fun(c, n, alpha, x, y);
}
static inline int rcupp_d_scal(rcupp_ctx c, int n, double alpha,
                               rcupp_buf x) {
  typedef int (*fn_t)(rcupp_ctx, int, double, rcupp_buf);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_scal");
  return fun(c, n, alpha, x);
}
static inline int rcupp_d_copy(rcupp_ctx c, int n,
                               rcupp_buf src, rcupp_buf dst) {
  typedef int (*fn_t)(rcupp_ctx, int, rcupp_buf, rcupp_buf);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_copy");
  return fun(c, n, src, dst);
}
static inline int rcupp_d_ger(rcupp_ctx c, int n, int p, double alpha,
                              rcupp_buf x, rcupp_buf y, rcupp_buf A) {
  typedef int (*fn_t)(rcupp_ctx, int, int, double,
                      rcupp_buf, rcupp_buf, rcupp_buf);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_ger");
  return fun(c, n, p, alpha, x, y, A);
}
static inline int rcupp_d_soft_threshold(rcupp_ctx c, rcupp_buf u,
                                         double t, rcupp_buf out, int n) {
  typedef int (*fn_t)(rcupp_ctx, rcupp_buf, double, rcupp_buf, int);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_soft_threshold");
  return fun(c, u, t, out, n);
}

/* ---- scalar reductions (host-returning; 8-byte sync point) ---- */
static inline double rcupp_d_dot(rcupp_ctx c, int n,
                                 rcupp_buf a, rcupp_buf b) {
  typedef double (*fn_t)(rcupp_ctx, int, rcupp_buf, rcupp_buf);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_dot");
  return fun(c, n, a, b);
}
static inline double rcupp_d_nrm2(rcupp_ctx c, int n, rcupp_buf x) {
  typedef double (*fn_t)(rcupp_ctx, int, rcupp_buf);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_nrm2");
  return fun(c, n, x);
}

/* ---------------------------------------------------------------------------
 * Phase 0 B-stage: four additional generic parts (ABI stays at 1).
 *
 * These are ADDITIVE, algorithm-agnostic building blocks on the SAME resident
 * buffers: a device-side row gather and a symmetric-rank-k / Cholesky solve
 * trio. They let a downstream package keep ONE full matrix resident and refit
 * many row subsets without re-uploading (e.g. reference-panel / CV / rolling
 * windows). rcupp still knows nothing about ridge, lasso or CV -- it only
 * supplies parts; the solver loop lives downstream (Rcpp + rcupp).
 *
 *   rcupp_d_gather_rows : dst[m x p] = X[idx[0..m), ]  (device-side row index)
 *   rcupp_d_syrk        : C[p x p] = alpha * X^T X + beta * C  (one triangle)
 *   rcupp_d_potrf       : Cholesky factor of an SPD A (LAPACK-style info)
 *   rcupp_d_potrs       : solve A X = B from that factor
 *
 * uplo is RCUPP_FILL_LOWER / RCUPP_FILL_UPPER and MUST match across syrk,
 * potrf and potrs. The int* info out-arg follows LAPACK semantics (0 = ok,
 * >0 = not positive definite); operator return is 0 on success, non-0 on an
 * invalid argument / backend failure. ABI version is UNCHANGED (1): existing
 * downstream code and the rcupp_ctx_abi() guard need no update.
 * ------------------------------------------------------------------------- */

/* dst[m x p] = X[idx[0..m), ], column-major (X[i + j*n], dst[i + j*m]).
 * idx is a HOST array of m zero-based row indices; each must be in [0, n). */
static inline int rcupp_d_gather_rows(rcupp_ctx c, rcupp_buf X, int n, int p,
                                      const int* idx, int m, rcupp_buf dst) {
  typedef int (*fn_t)(rcupp_ctx, rcupp_buf, int, int,
                      const int*, int, rcupp_buf);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_gather_rows");
  return fun(c, X, n, p, idx, m, dst);
}

/* C[p x p] = alpha * X^T X + beta * C, X is n x p; only uplo triangle of C is
 * written. alpha here is the BLAS scalar, NOT the elastic-net mixing alpha. */
static inline int rcupp_d_syrk(rcupp_ctx c, rcupp_buf X, int n, int p,
                               int uplo, double alpha, double beta,
                               rcupp_buf Cbuf) {
  typedef int (*fn_t)(rcupp_ctx, rcupp_buf, int, int,
                      int, double, double, rcupp_buf);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_syrk");
  return fun(c, X, n, p, uplo, alpha, beta, Cbuf);
}

/* Cholesky: A[n x n] (lda) -> factor in the uplo triangle; *info LAPACK. */
static inline int rcupp_d_potrf(rcupp_ctx c, int uplo, int n,
                                rcupp_buf A, int lda, int* info) {
  typedef int (*fn_t)(rcupp_ctx, int, int, rcupp_buf, int, int*);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_potrf");
  return fun(c, uplo, n, A, lda, info);
}

/* Solve A X = B from the potrf factor; A[n x n] (lda), B[n x nrhs] (ldb). */
static inline int rcupp_d_potrs(rcupp_ctx c, int uplo, int n, int nrhs,
                                rcupp_buf A, int lda,
                                rcupp_buf B, int ldb, int* info) {
  typedef int (*fn_t)(rcupp_ctx, int, int, int,
                      rcupp_buf, int, rcupp_buf, int, int*);
  static fn_t fun = (fn_t) R_GetCCallable("rcupp", "rcupp_d_potrs");
  return fun(c, uplo, n, nrhs, A, lda, B, ldb, info);
}

#endif /* RCUPP_H */
