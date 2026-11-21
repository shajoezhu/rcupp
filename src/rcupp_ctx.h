#ifndef RCUPP_CTX_H
#define RCUPP_CTX_H

/*
 * Private declaration of rcupp's PHASE 0 generic device-context C ABI.
 *
 * This is the algorithm-agnostic, RESIDENT device layer: rcupp owns device
 * memory + operators (the "parts"), and a downstream package composes them
 * into its own solver loop in C++ (Rcpp glue). rcupp never learns what a
 * lasso / ridge / PCA is. It sits ALONGSIDE (does not replace) the F0
 * one-shot operators (rcupp_api.h) and the lasso-specific P1 session
 * (rcupp_session.h).
 *
 * PLAIN C ABI, opaque void* handles, host double* only, no CUDA/Rcpp/STL
 * tokens cross the package boundary. It can be included by the g++-compiled
 * host wrappers AND the nvcc-compiled .cu implementation. Exactly one
 * implementation is linked:
 *   - CUDA build : rcupp_ctx_cuda.cu
 *   - no-CUDA    : rcupp_ctx_cpu.cpp
 * configure selects which object via @EXTRA_OBJECTS@ (see Makevars.in).
 *
 * The downstream-facing wrappers live in ../inst/include/rcupp.h and resolve
 * these symbols at runtime via R_GetCCallable() (registered in
 * rcupp_exports.cpp). Downstream packages include THAT header, not this one.
 *
 * All buffers belong to a context; rcupp_ctx_destroy() frees every buffer
 * still owned. X is n x p, COLUMN-MAJOR (R order), matching the F0 operators.
 * ABI version 1: rcupp_ctx_abi() returns 1. The B-stage parts
 * (gather_rows / syrk / potrf / potrs) and the reserved/level-2 parts
 * (gemm / gesvd / cg_solve / prox_grad_step) are ADDITIVE and do not bump the
 * ABI -- rcupp_ctx_abi() stays 1 and downstream guards need no update.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version: downstream can guard with Imports: rcupp (>= 0.1.0) or this
 * runtime query. Returns 1 in both backends. */
int   rcupp_ctx_abi(void);

/* ---- context lifecycle ---- */
void* rcupp_ctx_create(void);                 /* NULL on failure */
void  rcupp_ctx_destroy(void* c);             /* NULL-safe; frees all bufs */
void  rcupp_ctx_sync(void* c);                /* wait for queued work */

/* ---- buffer lifecycle (owned by ctx, freed at destroy) ---- */
void* rcupp_buf_alloc(void* c, int n);                         /* n doubles, NULL on failure */
void  rcupp_buf_free(void* c, void* b);                        /* optional early free */
int   rcupp_buf_upload(void* c, void* b,
                       const double* host, int n);             /* H->D */
int   rcupp_buf_download(void* c, void* b,
                         double* host, int n);                 /* D->H, implies sync */
int   rcupp_buf_fill(void* c, void* b,
                     double value, int n);                     /* set every element */

/* ---- resident device operators: no H<->D round trip ----
 * X is n x p column-major (R order). Semantics match the F0 operators.
 * Returns 0 on success, non-0 on invalid args / shape / buffer mismatch. */
int   rcupp_d_gemv(void* c, void* X, int n, int p,
                   void* v, void* out);                       /* out[n] = X v      */
int   rcupp_d_gemv_t(void* c, void* X, int n, int p,
                     void* r, void* out);                     /* out[p] = X^T r    */
int   rcupp_d_axpy(void* c, int n, double alpha,
                   void* x, void* y);                         /* y += alpha x      */
int   rcupp_d_scal(void* c, int n, double alpha,
                   void* x);                                  /* x *= alpha        */
int   rcupp_d_copy(void* c, int n,
                   void* src, void* dst);                     /* dst = src         */
int   rcupp_d_ger(void* c, int n, int p, double alpha,
                  void* x, void* y, void* A);                 /* A += alpha x y^T  */
int   rcupp_d_soft_threshold(void* c, void* u, double t,
                             void* out, int n);               /* lasso prox        */

/* Scalar reductions: return host double, a sync point (8 bytes, not X).
 * On invalid args they return 0.0 (no separate error channel). */
double rcupp_d_dot(void* c, int n, void* a, void* b);
double rcupp_d_nrm2(void* c, int n, void* x);

/* ---- B-stage additive parts (ABI stays 1) ----
 * uplo: RCUPP_FILL_LOWER=0 / RCUPP_FILL_UPPER=1 (mirrors cublasFillMode_t).
 * info: LAPACK-style host int out-arg (0 = ok, >0 = not positive definite).
 * Returns 0 on success, non-0 on invalid args / shape / backend failure. */
int   rcupp_d_gather_rows(void* c, void* X, int n, int p,
                          const int* idx, int m, void* dst);        /* dst[m x p] = X[idx,] */
int   rcupp_d_syrk(void* c, void* X, int n, int p,
                   int uplo, double alpha, double beta, void* C);   /* C[p x p] = a X^T X + b C */
int   rcupp_d_potrf(void* c, int uplo, int n,
                    void* A, int lda, int* info);                  /* Cholesky factor */
int   rcupp_d_potrs(void* c, int uplo, int n, int nrhs,
                    void* A, int lda, void* B, int ldb, int* info);/* solve A X = B */

/* ---- reserved / level-2 parts (ABI stays 1) ----
 * These complete the original Phase-B catalog. gemm/gesvd are level-1
 * BLAS/LAPACK-style parts; cg_solve/prox_grad_step are level-2 composites
 * built ONLY from the level-1 resident operators (see rcupp_parts_impl.h),
 * so both backends share one implementation and stay algorithm-agnostic:
 * rcupp still learns nothing about lasso / ridge / CV / lambda paths.
 *
 * op (gemm): RCUPP_OP_N=0 (no transpose) / RCUPP_OP_T=1 (transpose).
 * gemm:   C[m x n] = alpha * op(A) * op(B) + beta * C, column-major.
 * gesvd:  thin SVD of a TALL resident A[m x n] (m >= n, lda >= m). A is
 *         read-only (factored on an internal copy). k = n. Outputs S[k]
 *         (descending), U[m x k] (ldu >= m), VT[k x n] (ldvt >= k).
 *         *info is LAPACK-style (0 = ok, >0 = no convergence). m < n is
 *         REJECTED (cuSOLVER's legacy gesvd is tall-only); use the Gram
 *         (syrk) + Cholesky parts downstream for wide matrices.
 * cg_solve: solve (X^T X + shift*I) x = X^T y by conjugate gradients on
 *         resident buffers (a generic damped-normal-equations primitive,
 *         NOT a model solver). X is n x p, y n, x p (in: guess, out: solve).
 *         shift >= 0 (shift > 0 => SPD), tol > 0, max_iter >= 1. Out-args:
 *         *iters, *residual (= ||r||2 at stop), *converged (1/0).
 * prox_grad_step: ONE proximal-gradient step on resident buffers:
 *         out = prox(z + step * X^T (y - X z)), where prox(u) is
 *         soft_threshold(u, threshold) scaled by inv_scale. X n x p, y n,
 *         z p (current iterate), out p (next iterate). A level-2 part: CUDA
 *         fuses the whole step into ONE kernel launch (CPU fallback composes
 *         level-1 ops); z/out may alias. The lambda path / momentum /
 *         convergence stay DOWNSTREAM.
 * Returns 0 on success, non-0 on invalid args / shape / backend failure. */
int   rcupp_d_gemm(void* c, int transa, int transb, int m, int n, int k,
                   double alpha, void* A, int lda, void* B, int ldb,
                   double beta, void* C, int ldc);
int   rcupp_d_gesvd(void* c, void* A, int m, int n, int lda, void* S,
                    void* U, int ldu, void* VT, int ldvt, int* info);
int   rcupp_d_cg_solve(void* c, void* X, int n, int p, void* y, double shift,
                       double tol, int max_iter, void* x, int* iters,
                       double* residual, int* converged);
int   rcupp_d_prox_grad_step(void* c, void* X, int n, int p, void* y, void* z,
                             double step, double threshold, double inv_scale,
                             void* out);

#ifdef __cplusplus
}
#endif

#endif /* RCUPP_CTX_H */
