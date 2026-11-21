#ifndef RCUPP_API_H
#define RCUPP_API_H

/*
 * Internal declaration of rcupp's PUBLIC fixed-operator C ABI.
 *
 * Exactly one implementation is linked into the package shared object:
 *   - CUDA build : rcupp_api_cuda.cu  (+ rcupp_cuda_available from lasso_fista_cuda.o)
 *   - no-CUDA    : rcupp_api_cpu.cpp  (+ rcupp_cuda_available from cpu_stub.o)
 * The symbols are registered for downstream use in rcupp_exports.cpp via
 * R_RegisterCCallable() (the [[Rcpp::init]] hook).
 *
 * The downstream-facing header is ../inst/include/rcupp.h -- it wraps these
 * with R_GetCCallable(). Downstream code must include THAT header, not this
 * one; nothing here leaks into another package.
 *
 * All pointers are HOST double*; X is n x p, column-major (R order).
 * rcupp_cuda_available() is also declared in rcupp_internal.h; repeating the
 * identical extern "C" declaration here is legal and keeps the public ABI set
 * in one place.
 */

#ifdef __cplusplus
extern "C" {
#endif

int    rcupp_cuda_available(void);

void   rcupp_gemv(const double* X, int n, int p,
                  const double* v, double* out);
void   rcupp_gemv_t(const double* X, int n, int p,
                    const double* r, double* out);
void   rcupp_axpy(int n, double alpha, const double* x, double* y);
void   rcupp_scal(int n, double alpha, double* x);
double rcupp_dot(int n, const double* a, const double* b);
double rcupp_nrm2(int n, const double* x);
void   rcupp_soft_threshold(const double* u, double t, double* out, int n);

/* ---- Diagnostic / profiling primitives (ABI stays 1) ---------------------
 * Backend-agnostic, non-invasive: they never touch the operator hot path.
 * All byte/ms values cross the boundary as double (exact to 2^53).
 *   rcupp_device_mem_info: free and total device bytes; 0 on success, 1 when
 *     CUDA is unavailable/failed (both outputs set to -1.0).
 *   rcupp_device_tic:      arm a default-stream CUDA event; 0 on success, 1
 *     when CUDA is unavailable/failed.
 *   rcupp_device_toc:      device-side elapsed ms since tic (>= 0), or -1.0
 *     sentinel when CUDA is unavailable / no armed tic / error.
 *   rcupp_diag_shutdown:   release the process-global event pair; idempotent,
 *     internal lifecycle helper (deliberately NOT registered for downstream).
 * The CPU fallback implements all four with well-defined sentinels / no-ops so
 * downstream C++ callers and R behave identically on either backend.
 * ------------------------------------------------------------------------- */
int    rcupp_device_mem_info(double* free_bytes, double* total_bytes);
int    rcupp_device_tic(void);
double rcupp_device_toc(void);
void   rcupp_diag_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* RCUPP_API_H */
