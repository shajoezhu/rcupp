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
 * ABI version 1: rcupp_ctx_abi() returns 1.
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

#ifdef __cplusplus
}
#endif

#endif /* RCUPP_CTX_H */
