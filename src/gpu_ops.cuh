#ifndef RCUPP_GPU_OPS_CUH
#define RCUPP_GPU_OPS_CUH

/*
 * The FIXED GPU operator set exposed to the solver. This is a small, closed
 * library of primitives -- NOT a generic "compile CUDA from R" framework.
 *
 * All pointers are device pointers. X is n x p, column-major (R order), so it
 * maps straight onto cuBLAS without any transpose.
 *
 * compileAttributes() does not scan .cuh/.cu, and these files carry no
 * [[Rcpp::export]] attributes by design.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>

/* out(n) = X * v,  v(p) */
void op_gemv(cublasHandle_t h, const double* X, int n, int p,
             const double* v, double* out);

/* out(p) = X^T * r,  r(n) */
void op_gemv_t(cublasHandle_t h, const double* X, int n, int p,
               const double* r, double* out);

/* y += alpha * x  (length n) */
void op_axpy(cublasHandle_t h, int n, double alpha,
             const double* x, double* y);

/* dst = alpha * src  (length n); alpha = 1 doubles as a device-to-device copy */
void op_copy_scale(cublasHandle_t h, int n, const double* src,
                   double alpha, double* dst);

/* out = a - b  (length n) */
void op_diff(cublasHandle_t h, int n, const double* a,
             const double* b, double* out);

double op_nrm2(cublasHandle_t h, int n, const double* x);
double op_dot(cublasHandle_t h, int n, const double* a, const double* b);

/* The single custom kernel: out = sign(u) * max(|u| - t, 0), elementwise. */
void op_soft_threshold(const double* u, double t, double* out, int n);

/* RAII owner of one device double buffer. */
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : ptr_(NULL) {
    if (count > 0) {
      cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(double));
    }
  }
  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  double* get() const { return ptr_; }
  bool valid() const { return ptr_ != NULL; }

  /* Reallocate to `count` doubles, freeing the old block first. Used by the
   * P1 one-shot pool / session to resize work buffers when (n, p) changes,
   * instead of destroying and rebuilding the whole owner. If cudaMalloc fails,
   * ptr_ becomes NULL and valid() reports false. */
  void reset(size_t count) {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = NULL;
    }
    if (count > 0) {
      cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(double));
    }
  }

 private:
  double* ptr_;
};

#endif /* RCUPP_GPU_OPS_CUH */
