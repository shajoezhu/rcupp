#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include "gpu_ops.cuh"

/*
 * CUDA implementation of the fixed operator set.
 *
 * The soft-threshold kernel and its launcher live in the SAME translation unit
 * on purpose: no device code is referenced across .cu files, so nvcc needs
 * neither -rdc=true nor a separate device-link step. The final shared object is
 * linked by R's C++ linker with -lcudart -lcublas.
 */

namespace {

__global__ void soft_threshold_kernel(const double* u, double t,
                                      double* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const double a = fabs(u[i]) - t;
    out[i] = a > 0.0 ? copysign(a, u[i]) : 0.0;
  }
}

}  // namespace

void op_gemv(cublasHandle_t h, const double* X, int n, int p,
             const double* v, double* out) {
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasDgemv(h, CUBLAS_OP_N, n, p, &alpha, X, n, v, 1, &beta, out, 1);
}

void op_gemv_t(cublasHandle_t h, const double* X, int n, int p,
               const double* r, double* out) {
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasDgemv(h, CUBLAS_OP_T, n, p, &alpha, X, n, r, 1, &beta, out, 1);
}

void op_axpy(cublasHandle_t h, int n, double alpha,
             const double* x, double* y) {
  cublasDaxpy(h, n, &alpha, x, 1, y, 1);
}

void op_copy_scale(cublasHandle_t h, int n, const double* src,
                   double alpha, double* dst) {
  cublasDcopy(h, n, src, 1, dst, 1);
  cublasDscal(h, n, &alpha, dst, 1);
}

void op_diff(cublasHandle_t h, int n, const double* a,
             const double* b, double* out) {
  const double neg_one = -1.0;
  cublasDcopy(h, n, a, 1, out, 1);
  cublasDaxpy(h, n, &neg_one, b, 1, out, 1);
}

/* cuBLAS v2 reductions return cublasStatus_t and write the scalar through a
 * pointer (host pointer under the default CUBLAS_POINTER_MODE_HOST). */
double op_nrm2(cublasHandle_t h, int n, const double* x) {
  double result = 0.0;
  cublasDnrm2(h, n, x, 1, &result);
  return result;
}

double op_dot(cublasHandle_t h, int n, const double* a, const double* b) {
  double result = 0.0;
  cublasDdot(h, n, a, 1, b, 1, &result);
  return result;
}

void op_soft_threshold(const double* u, double t, double* out, int n) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  soft_threshold_kernel<<<grid, block>>>(u, t, out, n);
}
