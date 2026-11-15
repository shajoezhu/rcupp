#ifndef RCUPP_INTERNAL_H
#define RCUPP_INTERNAL_H

/*
 * The single shared ABI between the R/Rcpp layer and the compiled solver.
 * Pure C/C++ on purpose (no CUDA tokens) so it can be included by both the
 * g++-compiled host files and the nvcc-compiled .cu files.
 *
 * Exactly one implementation is linked into the package shared object:
 *   - CUDA build : gpu_ops.o + lasso_fista_cuda.o
 *   - no-CUDA    : cpu_stub.o
 * configure selects which via OBJECTS; there is no #ifdef dispatch here.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when a usable CUDA device/backend was linked in, else 0 (CPU reference shim). */
int rcupp_cuda_available(void);

/*
 * Gaussian lasso via FISTA.
 *   F(beta) = 1/(2n) ||y - X beta||^2 + lambda ||beta||_1
 * X is n x p, COLUMN-MAJOR (R order; matches cuBLAS with no transpose).
 * Returns 0 on success, 1 on invalid arguments.
 * Fills beta_out[p], *iters_out, *obj_out and *converged_out.
 */
int rcupp_lasso_fista(const double* X, const double* y,
                      int n, int p, double lambda,
                      int max_iter, double tol,
                      double* beta_out, int* iters_out,
                      double* obj_out, int* converged_out);

/*
 * Tear down the process-global one-shot pool (persistent cuBLAS handle +
 * reused device buffers) owned by the CUDA implementation of
 * rcupp_lasso_fista. Called from .onUnload so CUDA resources are released at
 * a controlled time BEFORE the DLL is unloaded / the CUDA context is torn
 * down, avoiding destruction-order crashes at process exit. No-op in the
 * no-CUDA build. Safe to call more than once.
 */
void rcupp_pool_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* RCUPP_INTERNAL_H */
