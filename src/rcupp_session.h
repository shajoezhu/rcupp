#ifndef RCUPP_SESSION_H
#define RCUPP_SESSION_H

/*
 * Private declaration of rcupp's P1 SESSION C ABI.
 *
 * A session keeps the design matrix X / response y resident (on the device in
 * the CUDA build), plus the cuBLAS handle, the FISTA work buffers and the
 * cached Lipschitz constant L, across MANY fits. That removes the repeated
 * per-call setup (upload, cublasCreate, cudaMalloc, power iteration) that
 * dominates lambda paths, cross-validation and other repeated-fit workloads.
 *
 * This is a PLAIN C ABI: opaque void* handle, host double* data, no CUDA/Rcpp
 * tokens. It can be included by the g++-compiled host wrappers AND the
 * nvcc-compiled .cu implementation. Exactly one implementation is linked:
 *   - CUDA build : rcupp_session_cuda.cu
 *   - no-CUDA    : rcupp_session_cpu.cpp
 * configure selects which object goes into OBJECTS (see Makevars.in).
 *
 * The downstream-facing wrappers live in ../inst/include/rcupp.h and resolve
 * these symbols at runtime via R_GetCCallable() (registered in
 * rcupp_exports.cpp). Downstream packages include THAT header, not this one.
 *
 * Numerical contract (identical to rcupp_lasso_fista):
 *   F(beta) = 1/(2n) ||y - X beta||^2 + lambda ||beta||_1
 *   X is n x p, COLUMN-MAJOR (R order).
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Create an empty session. Returns NULL on failure. */
void* rcupp_session_create(void);

/*
 * Bind (or rebind) the data for this session. Copies X/y into the session's
 * own storage (host mirror + device copy in the CUDA build) and precomputes
 * the Lipschitz constant L = lambda_max(X'X)/n once, by the same deterministic
 * power iteration used by the one-shot solver. Call once, then fit() many
 * lambdas against the SAME X/y. Returns 0 on success, 1 on invalid arguments
 * or allocation failure.
 */
int rcupp_session_set_data(void* h, const double* X, const double* y,
                           int n, int p);

/*
 * Solve one lasso for `lambda` reusing the session's resident state.
 *   warm_start = 0 : cold start (beta=z=0, tk=1) -> BIT-IDENTICAL to the
 *                    one-shot rcupp_lasso_fista() for the same data/lambda.
 *   warm_start = 1 : start from the previous fit's solution (z=beta, tk=1,
 *                    min 2 iterations); converges to the same optimum but is
 *                    NOT bit-identical. Use a relaxed validation tolerance.
 * Fills beta_out[p], *iters_out, *obj_out, *converged_out. Returns 0 on
 * success, 1 on invalid arguments (e.g. set_data() was never called).
 */
int rcupp_session_fit(void* h, double lambda, int warm_start,
                      int max_iter, double tol,
                      double* beta_out, int* iters_out,
                      double* obj_out, int* converged_out);

/* Destroy a session and free all of its resources. Safe on NULL. */
void rcupp_session_destroy(void* h);

#ifdef __cplusplus
}
#endif

#endif /* RCUPP_SESSION_H */
