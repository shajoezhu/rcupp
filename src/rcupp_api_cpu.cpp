#include <cmath>
#include "rcupp_api.h"

/*
 * CPU-fallback implementation of rcupp's PUBLIC fixed-operator C ABI.
 *
 * Identical signatures to rcupp_api_cuda.cu; plain column-major loops, no CUDA
 * and deliberately no RcppArmadillo dependency so it links cleanly in the
 * no-CUDA build. rcupp_cuda_available() is defined in cpu_stub.cpp, not here.
 *
 * This object is compiled by g++ ONLY in the no-CUDA build (configure adds it
 * to @EXTRA_OBJECTS@); the CUDA build links rcupp_api_cuda.o instead. Both
 * register the same symbols, so downstream packages are backend-agnostic.
 */

extern "C" void rcupp_gemv(const double* X, int n, int p,
                           const double* v, double* out) {
  /* X is n x p column-major: X[i + j*n]. */
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < p; ++j) s += X[i + j * n] * v[j];
    out[i] = s;
  }
}

extern "C" void rcupp_gemv_t(const double* X, int n, int p,
                             const double* r, double* out) {
  for (int j = 0; j < p; ++j) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += X[i + j * n] * r[i];
    out[j] = s;
  }
}

extern "C" void rcupp_axpy(int n, double alpha, const double* x, double* y) {
  for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

extern "C" void rcupp_scal(int n, double alpha, double* x) {
  for (int i = 0; i < n; ++i) x[i] *= alpha;
}

extern "C" double rcupp_dot(int n, const double* a, const double* b) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

extern "C" double rcupp_nrm2(int n, const double* x) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += x[i] * x[i];
  return std::sqrt(s);
}

extern "C" void rcupp_soft_threshold(const double* u, double t,
                                     double* out, int n) {
  for (int i = 0; i < n; ++i) {
    const double a = std::fabs(u[i]) - t;
    out[i] = a > 0.0 ? std::copysign(a, u[i]) : 0.0;
  }
}

/* ---- Diagnostic / profiling primitives (CPU fallback) --------------------
 * Identical signatures to rcupp_api_cuda.cu. There is no device here, so every
 * query returns the documented "CUDA unavailable" sentinel: -1.0 for byte/ms
 * values and 1 for the int status. This keeps downstream C++ callers and the
 * R layer backend-agnostic -- they convert the sentinels to NA/FALSE, never
 * crash. rcupp_diag_shutdown() is a no-op (no CUDA resources to release).
 * ------------------------------------------------------------------------- */
extern "C" int rcupp_device_mem_info(double* free_bytes, double* total_bytes) {
  if (free_bytes) *free_bytes = -1.0;
  if (total_bytes) *total_bytes = -1.0;
  return 1;
}

extern "C" int rcupp_device_tic(void) {
  return 1;
}

extern "C" double rcupp_device_toc(void) {
  return -1.0;
}

extern "C" void rcupp_diag_shutdown(void) {
}
