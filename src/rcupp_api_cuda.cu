#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
#include "gpu_ops.cuh"
#include "rcupp_api.h"

/*
 * CUDA-backed implementation of rcupp's PUBLIC fixed-operator C ABI.
 *
 * Each call is one-shot: H->D copy, an op_* device primitive, D->H copy.
 * Persistent device residency and a reusable cuBLAS handle are a future
 * session API (P1); this file deliberately keeps the API stateless.
 *
 * op_soft_threshold() is a HOST launcher defined in gpu_ops.cu. Calling it
 * across translation units is fine -- only the inner __global__ kernel is
 * private to that file -- so no -rdc=true / device-link step is needed. The
 * blocking D->H cudaMemcpy after a kernel launch also serves as the sync.
 *
 * This object is compiled by nvcc ONLY in the CUDA build (configure adds it to
 * @EXTRA_OBJECTS@); the CPU build links rcupp_api_cpu.o instead.
 */

namespace {

/* One cuBLAS handle per wrapper call; destroyed on scope exit. */
class ScopedHandle {
 public:
  ScopedHandle() : ok_(false) {
    ok_ = (cublasCreate(&handle_) == CUBLAS_STATUS_SUCCESS);
  }
  ~ScopedHandle() {
    if (ok_) cublasDestroy(handle_);
  }
  ScopedHandle(const ScopedHandle&) = delete;
  ScopedHandle& operator=(const ScopedHandle&) = delete;
  bool ok() const { return ok_; }
  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_;
  bool ok_;
};

/*
 * Diagnostic event pair for the default-stream device timer. Held through a
 * deliberately LEAKED naked pointer (same rationale as OneShotPool in
 * lasso_fista_cuda.cu: a static destructor calling cudaEventDestroy could run
 * after the CUDA context is gone and crash). Created lazily on the first
 * rcupp_device_tic(); released only by rcupp_diag_shutdown() at unload. If that
 * is never called the pointer simply leaks at exit -- harmless, no CUDA calls
 * in any destructor. There is ONE process-global pair, so the timer is NOT
 * reentrant/nestable (R is single-threaded; only call-stack nesting matters).
 */
struct DiagState {
  cudaEvent_t start;
  cudaEvent_t stop;
  bool created;
  bool armed;

  DiagState() : start(NULL), stop(NULL), created(false), armed(false) {}
};

/* Deliberately leaked; freed only by rcupp_diag_shutdown() at unload. */
DiagState* g_diag = NULL;

DiagState& diag() {
  if (!g_diag) g_diag = new DiagState();
  return *g_diag;
}

}  // namespace

extern "C" void rcupp_gemv(const double* X, int n, int p,
                           const double* v, double* out) {
  if (n <= 0 || p <= 0) return;
  ScopedHandle sh;
  if (!sh.ok()) return;
  DeviceBuffer d_X(static_cast<std::size_t>(n) * p), d_v(p), d_out(n);
  if (!d_X.valid() || !d_v.valid() || !d_out.valid()) return;
  cudaMemcpy(d_X.get(), X, sizeof(double) * n * p, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v.get(), v, sizeof(double) * p, cudaMemcpyHostToDevice);
  op_gemv(sh.get(), d_X.get(), n, p, d_v.get(), d_out.get());
  cudaMemcpy(out, d_out.get(), sizeof(double) * n, cudaMemcpyDeviceToHost);
}

extern "C" void rcupp_gemv_t(const double* X, int n, int p,
                             const double* r, double* out) {
  if (n <= 0 || p <= 0) return;
  ScopedHandle sh;
  if (!sh.ok()) return;
  DeviceBuffer d_X(static_cast<std::size_t>(n) * p), d_r(n), d_out(p);
  if (!d_X.valid() || !d_r.valid() || !d_out.valid()) return;
  cudaMemcpy(d_X.get(), X, sizeof(double) * n * p, cudaMemcpyHostToDevice);
  cudaMemcpy(d_r.get(), r, sizeof(double) * n, cudaMemcpyHostToDevice);
  op_gemv_t(sh.get(), d_X.get(), n, p, d_r.get(), d_out.get());
  cudaMemcpy(out, d_out.get(), sizeof(double) * p, cudaMemcpyDeviceToHost);
}

extern "C" void rcupp_axpy(int n, double alpha, const double* x, double* y) {
  if (n <= 0) return;
  ScopedHandle sh;
  if (!sh.ok()) return;
  DeviceBuffer d_x(n), d_y(n);
  if (!d_x.valid() || !d_y.valid()) return;
  cudaMemcpy(d_x.get(), x, sizeof(double) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y.get(), y, sizeof(double) * n, cudaMemcpyHostToDevice);
  op_axpy(sh.get(), n, alpha, d_x.get(), d_y.get());
  cudaMemcpy(y, d_y.get(), sizeof(double) * n, cudaMemcpyDeviceToHost);
}

extern "C" void rcupp_scal(int n, double alpha, double* x) {
  if (n <= 0) return;
  ScopedHandle sh;
  if (!sh.ok()) return;
  DeviceBuffer d_x(n), d_y(n);
  if (!d_x.valid() || !d_y.valid()) return;
  cudaMemcpy(d_x.get(), x, sizeof(double) * n, cudaMemcpyHostToDevice);
  /* op_copy_scale(dst = alpha * src) -- separate dst keeps it unambiguous. */
  op_copy_scale(sh.get(), n, d_x.get(), alpha, d_y.get());
  cudaMemcpy(x, d_y.get(), sizeof(double) * n, cudaMemcpyDeviceToHost);
}

extern "C" double rcupp_dot(int n, const double* a, const double* b) {
  if (n <= 0) return 0.0;
  ScopedHandle sh;
  if (!sh.ok()) return 0.0;
  DeviceBuffer d_a(n), d_b(n);
  if (!d_a.valid() || !d_b.valid()) return 0.0;
  cudaMemcpy(d_a.get(), a, sizeof(double) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b.get(), b, sizeof(double) * n, cudaMemcpyHostToDevice);
  /* cuBLAS v2 reductions write the scalar through a host pointer (blocking). */
  return op_dot(sh.get(), n, d_a.get(), d_b.get());
}

extern "C" double rcupp_nrm2(int n, const double* x) {
  if (n <= 0) return 0.0;
  ScopedHandle sh;
  if (!sh.ok()) return 0.0;
  DeviceBuffer d_x(n);
  if (!d_x.valid()) return 0.0;
  cudaMemcpy(d_x.get(), x, sizeof(double) * n, cudaMemcpyHostToDevice);
  return op_nrm2(sh.get(), n, d_x.get());
}

extern "C" void rcupp_soft_threshold(const double* u, double t,
                                     double* out, int n) {
  if (n <= 0) return;
  DeviceBuffer d_u(n), d_out(n);
  if (!d_u.valid() || !d_out.valid()) return;
  cudaMemcpy(d_u.get(), u, sizeof(double) * n, cudaMemcpyHostToDevice);
  op_soft_threshold(d_u.get(), t, d_out.get(), n);
  cudaMemcpy(out, d_out.get(), sizeof(double) * n, cudaMemcpyDeviceToHost);
}

/* ---- Diagnostic / profiling primitives --------------------------------- */

extern "C" int rcupp_device_mem_info(double* free_bytes, double* total_bytes) {
  if (free_bytes) *free_bytes = -1.0;
  if (total_bytes) *total_bytes = -1.0;
  if (rcupp_cuda_available() != 1) return 1;
  std::size_t f = 0, t = 0;
  if (cudaMemGetInfo(&f, &t) != cudaSuccess) return 1;
  if (free_bytes) *free_bytes = static_cast<double>(f);
  if (total_bytes) *total_bytes = static_cast<double>(t);
  return 0;
}

extern "C" int rcupp_device_tic(void) {
  if (rcupp_cuda_available() != 1) return 1;
  DiagState& d = diag();
  if (!d.created) {
    if (cudaEventCreate(&d.start) != cudaSuccess) return 1;
    if (cudaEventCreate(&d.stop) != cudaSuccess) {
      cudaEventDestroy(d.start);
      d.start = NULL;
      return 1;
    }
    d.created = true;
  }
  /* Record on the legacy default stream (0); all rcupp operators use it. */
  if (cudaEventRecord(d.start, 0) != cudaSuccess) {
    d.armed = false;
    return 1;
  }
  d.armed = true;
  return 0;
}

extern "C" double rcupp_device_toc(void) {
  DiagState* d = g_diag;  /* may be NULL if tic was never armed */
  if (d == NULL || !d->created || !d->armed) return -1.0;
  d->armed = false;
  if (cudaEventRecord(d->stop, 0) != cudaSuccess) return -1.0;
  if (cudaEventSynchronize(d->stop) != cudaSuccess) return -1.0;
  float ms = 0.0f;
  if (cudaEventElapsedTime(&ms, d->start, d->stop) != cudaSuccess) return -1.0;
  return static_cast<double>(ms);
}

extern "C" void rcupp_diag_shutdown(void) {
  if (g_diag) {
    if (g_diag->created) {
      cudaEventDestroy(g_diag->start);
      cudaEventDestroy(g_diag->stop);
    }
    delete g_diag;
    g_diag = NULL;
  }
}
