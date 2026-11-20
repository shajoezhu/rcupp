#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstddef>
#include <vector>
#include "gpu_ops.cuh"
#include "rcupp_ctx.h"

/*
 * CUDA implementation of the PHASE 0 generic device-context C ABI
 * (rcupp_ctx.h). Algorithm-agnostic, resident: a context owns one cuBLAS
 * handle and a registry of device buffers; every rcupp_d_* operator runs on
 * those buffers with NO host<->device round trip. Downstream composes them
 * into its own solver loop (Rcpp + rcupp).
 *
 * This sits ALONGSIDE the one-shot F0 operators (rcupp_api_cuda.cu) and the
 * lasso-specific P1 session (rcupp_session_cuda.cu); it does not modify them.
 * Exactly one of rcupp_ctx_cuda.o / rcupp_ctx_cpu.o is linked (configure).
 *
 * Device code: the __global__ kernels here (ctx_fill_kernel,
 * ctx_gather_rows_kernel) are DEFINED and LAUNCHED in this translation unit,
 * so no device symbol crosses .cu files and nvcc needs neither -rdc=true nor
 * a device-link step -- the same rule gpu_ops.cu documents. All other
 * operators call the existing op_* host launchers in gpu_ops.cu, cuBLAS
 * (syrk) or cuSOLVER (potrf/potrs) directly.
 *
 * Lifetime: contexts/buffers are created and destroyed explicitly by the
 * downstream C++ caller (RAII). Nothing is held in a process-global static
 * with a CUDA-touching destructor, so there is no unload-order hazard.
 */

namespace {

/* fill: no cuBLAS primitive for "set every element to a value". Self-contained
 * kernel + launcher, kept in this TU. */
__global__ void ctx_fill_kernel(double* x, double value, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = value;
}

/* device-side row gather: dst[m x p] = X[idx[0..m), ], column-major.
 * X[i + j*n] -> dst[i + j*m]. idx is a device int array of length m. */
__global__ void ctx_gather_rows_kernel(const double* X, int n, int p,
                                       const int* idx, int m, double* dst) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * p;
  if (tid >= total) return;
  const int i = tid % m;
  const int j = tid / m;
  dst[i + static_cast<std::size_t>(j) * m] =
      X[idx[i] + static_cast<std::size_t>(j) * n];
}

struct Buf {
  DeviceBuffer mem;   /* device double buffer, RAII */
  int n;              /* capacity in doubles */
  explicit Buf(int count)
      : mem(static_cast<std::size_t>(count)), n(count) {}
};

struct Ctx {
  cublasHandle_t handle;
  bool handle_ok;
  cusolverDnHandle_t cusolver;   /* best-effort; potrf/potrs only */
  bool cusolver_ok;
  double* d_work;               /* cuSOLVER potrf workspace (doubles) */
  int work_cap;                  /* capacity in doubles */
  int* d_info;                   /* device int, cuSOLVER info */
  int* d_idx;                    /* device int array, gather row indices */
  int idx_cap;                   /* capacity in ints */
  std::vector<Buf*> bufs;        /* owned; freed at rcupp_ctx_destroy */
  Ctx()
      : handle(), handle_ok(false), cusolver(), cusolver_ok(false),
        d_work(NULL), work_cap(0), d_info(NULL), d_idx(NULL), idx_cap(0) {}
};

/* Resolve an opaque buffer handle: must be non-NULL, belong to this context,
 * and hold at least `need` doubles. Returns NULL otherwise (越界校验). */
Buf* resolve(Ctx* c, void* h, std::size_t need) {
  if (c == NULL || h == NULL) return NULL;
  for (std::size_t k = 0; k < c->bufs.size(); ++k) {
    Buf* b = c->bufs[k];
    if (static_cast<void*>(b) == h) {
      if (need > static_cast<std::size_t>(b->n)) return NULL;
      return b;
    }
  }
  return NULL;
}

bool ctx_ok(Ctx* c) { return c != NULL && c->handle_ok; }

/* Map RCUPP_FILL_* to cublasFillMode_t (LOWER=0, UPPER=1). */
cublasFillMode_t fill_mode(int uplo) {
  return uplo == 1 ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
}

/* Lazily (re)size the gather index scratch to hold `need` ints. */
bool ensure_idx(Ctx* c, int need) {
  if (need <= c->idx_cap && c->d_idx != NULL) return true;
  int* p = NULL;
  if (cudaMalloc(reinterpret_cast<void**>(&p),
                 static_cast<std::size_t>(need) * sizeof(int))
      != cudaSuccess) return false;
  if (c->d_idx != NULL) cudaFree(c->d_idx);
  c->d_idx = p;
  c->idx_cap = need;
  return true;
}

/* Lazily (re)size the cuSOLVER workspace to hold `need` doubles. */
bool ensure_work(Ctx* c, int need) {
  if (need <= c->work_cap && c->d_work != NULL) return true;
  double* p = NULL;
  if (cudaMalloc(reinterpret_cast<void**>(&p),
                 static_cast<std::size_t>(need) * sizeof(double))
      != cudaSuccess) return false;
  if (c->d_work != NULL) cudaFree(c->d_work);
  c->d_work = p;
  c->work_cap = need;
  return true;
}

/* Lazily allocate the single device int used for cuSOLVER info. */
bool ensure_info(Ctx* c) {
  if (c->d_info != NULL) return true;
  int* p = NULL;
  if (cudaMalloc(reinterpret_cast<void**>(&p), sizeof(int)) != cudaSuccess)
    return false;
  c->d_info = p;
  return true;
}

}  // namespace

extern "C" int rcupp_ctx_abi(void) {
  return 1;
}

extern "C" void* rcupp_ctx_create(void) {
  Ctx* c = new Ctx();
  if (cublasCreate(&c->handle) != CUBLAS_STATUS_SUCCESS) {
    delete c;
    return NULL;
  }
  c->handle_ok = true;
  /* cuSOLVER handle is best-effort: if it fails, only potrf/potrs are
   * unavailable (they return 1); the other 16 operators still work. */
  if (cusolverDnCreate(&c->cusolver) == CUSOLVER_STATUS_SUCCESS) {
    c->cusolver_ok = true;
  }
  return static_cast<void*>(c);
}

extern "C" void rcupp_ctx_destroy(void* h) {
  if (h == NULL) return;
  Ctx* c = static_cast<Ctx*>(h);
  for (std::size_t k = 0; k < c->bufs.size(); ++k) delete c->bufs[k];
  c->bufs.clear();
  if (c->d_work != NULL) cudaFree(c->d_work);
  if (c->d_info != NULL) cudaFree(c->d_info);
  if (c->d_idx != NULL) cudaFree(c->d_idx);
  if (c->cusolver_ok) cusolverDnDestroy(c->cusolver);
  if (c->handle_ok) cublasDestroy(c->handle);
  delete c;
}

extern "C" void rcupp_ctx_sync(void* h) {
  if (h == NULL) return;
  cudaDeviceSynchronize();
}

extern "C" void* rcupp_buf_alloc(void* h, int n) {
  if (h == NULL || n <= 0) return NULL;
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c)) return NULL;
  Buf* b = new Buf(n);
  if (!b->mem.valid()) {
    delete b;
    return NULL;
  }
  c->bufs.push_back(b);
  return static_cast<void*>(b);
}

extern "C" void rcupp_buf_free(void* h, void* bh) {
  if (h == NULL || bh == NULL) return;
  Ctx* c = static_cast<Ctx*>(h);
  for (std::size_t k = 0; k < c->bufs.size(); ++k) {
    if (static_cast<void*>(c->bufs[k]) == bh) {
      delete c->bufs[k];
      c->bufs.erase(c->bufs.begin() + k);
      return;
    }
  }
}

extern "C" int rcupp_buf_upload(void* h, void* bh,
                                const double* host, int n) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || host == NULL || n < 0) return 1;
  Buf* b = resolve(c, bh, static_cast<std::size_t>(n));
  if (b == NULL) return 1;
  if (n == 0) return 0;
  return cudaMemcpy(b->mem.get(), host, n * sizeof(double),
                    cudaMemcpyHostToDevice) == cudaSuccess ? 0 : 1;
}

extern "C" int rcupp_buf_download(void* h, void* bh,
                                  double* host, int n) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || host == NULL || n < 0) return 1;
  Buf* b = resolve(c, bh, static_cast<std::size_t>(n));
  if (b == NULL) return 1;
  if (n == 0) return 0;
  /* Synchronous cudaMemcpy doubles as the stream sync. */
  return cudaMemcpy(host, b->mem.get(), n * sizeof(double),
                    cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 1;
}

extern "C" int rcupp_buf_fill(void* h, void* bh, double value, int n) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n < 0) return 1;
  Buf* b = resolve(c, bh, static_cast<std::size_t>(n));
  if (b == NULL) return 1;
  if (n == 0) return 0;
  const int block = 256;
  const int grid = (n + block - 1) / block;
  ctx_fill_kernel<<<grid, block>>>(b->mem.get(), value, n);
  return 0;
}

extern "C" int rcupp_d_gemv(void* h, void* Xh, int n, int p,
                            void* vh, void* outh) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0 || p <= 0) return 1;
  Buf* X = resolve(c, Xh, static_cast<std::size_t>(n) * p);
  Buf* v = resolve(c, vh, static_cast<std::size_t>(p));
  Buf* out = resolve(c, outh, static_cast<std::size_t>(n));
  if (X == NULL || v == NULL || out == NULL) return 1;
  op_gemv(c->handle, X->mem.get(), n, p, v->mem.get(), out->mem.get());
  return 0;
}

extern "C" int rcupp_d_gemv_t(void* h, void* Xh, int n, int p,
                              void* rh, void* outh) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0 || p <= 0) return 1;
  Buf* X = resolve(c, Xh, static_cast<std::size_t>(n) * p);
  Buf* r = resolve(c, rh, static_cast<std::size_t>(n));
  Buf* out = resolve(c, outh, static_cast<std::size_t>(p));
  if (X == NULL || r == NULL || out == NULL) return 1;
  op_gemv_t(c->handle, X->mem.get(), n, p, r->mem.get(), out->mem.get());
  return 0;
}

extern "C" int rcupp_d_axpy(void* h, int n, double alpha,
                            void* xh, void* yh) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 1;
  Buf* x = resolve(c, xh, static_cast<std::size_t>(n));
  Buf* y = resolve(c, yh, static_cast<std::size_t>(n));
  if (x == NULL || y == NULL) return 1;
  op_axpy(c->handle, n, alpha, x->mem.get(), y->mem.get());
  return 0;
}

extern "C" int rcupp_d_scal(void* h, int n, double alpha, void* xh) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 1;
  Buf* x = resolve(c, xh, static_cast<std::size_t>(n));
  if (x == NULL) return 1;
  cublasDscal(c->handle, n, &alpha, x->mem.get(), 1);
  return 0;
}

extern "C" int rcupp_d_copy(void* h, int n, void* srch, void* dsth) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 1;
  Buf* src = resolve(c, srch, static_cast<std::size_t>(n));
  Buf* dst = resolve(c, dsth, static_cast<std::size_t>(n));
  if (src == NULL || dst == NULL) return 1;
  cublasDcopy(c->handle, n, src->mem.get(), 1, dst->mem.get(), 1);
  return 0;
}

/* A(n x p, column-major) += alpha * x(n) * y(p)^T. */
extern "C" int rcupp_d_ger(void* h, int n, int p, double alpha,
                           void* xh, void* yh, void* Ah) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0 || p <= 0) return 1;
  Buf* x = resolve(c, xh, static_cast<std::size_t>(n));
  Buf* y = resolve(c, yh, static_cast<std::size_t>(p));
  Buf* A = resolve(c, Ah, static_cast<std::size_t>(n) * p);
  if (x == NULL || y == NULL || A == NULL) return 1;
  /* cuBLAS is column-major: m=n rows, n=p cols, lda=n. */
  cublasDger(c->handle, n, p, &alpha, x->mem.get(), 1, y->mem.get(), 1,
             A->mem.get(), n);
  return 0;
}

extern "C" int rcupp_d_soft_threshold(void* h, void* uh, double t,
                                      void* outh, int n) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 1;
  Buf* u = resolve(c, uh, static_cast<std::size_t>(n));
  Buf* out = resolve(c, outh, static_cast<std::size_t>(n));
  if (u == NULL || out == NULL) return 1;
  op_soft_threshold(u->mem.get(), t, out->mem.get(), n);
  return 0;
}

extern "C" double rcupp_d_dot(void* h, int n, void* ah, void* bh) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 0.0;
  Buf* a = resolve(c, ah, static_cast<std::size_t>(n));
  Buf* b = resolve(c, bh, static_cast<std::size_t>(n));
  if (a == NULL || b == NULL) return 0.0;
  /* cuBLAS v2 reduction writes the scalar through a host pointer (blocking). */
  return op_dot(c->handle, n, a->mem.get(), b->mem.get());
}

extern "C" double rcupp_d_nrm2(void* h, int n, void* xh) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 0.0;
  Buf* x = resolve(c, xh, static_cast<std::size_t>(n));
  if (x == NULL) return 0.0;
  return op_nrm2(c->handle, n, x->mem.get());
}

/* ---------------------------------------------------------------------------
 * B-stage additive parts (ABI stays 1). cuBLAS syrk, cuSOLVER potrf/potrs,
 * and a device-side gather. All run on resident buffers; only info/idx cross
 * the host-device boundary (O(m) ints, never O(n*p)).
 * ------------------------------------------------------------------------- */

extern "C" int rcupp_d_gather_rows(void* h, void* Xh, int n, int p,
                                   const int* idx, int m, void* dsth) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0 || p <= 0 || m <= 0 || idx == NULL) return 1;
  /* Validate host row indices (O(m)) before any device work. */
  for (int k = 0; k < m; ++k) {
    if (idx[k] < 0 || idx[k] >= n) return 1;
  }
  Buf* X = resolve(c, Xh, static_cast<std::size_t>(n) * p);
  Buf* dst = resolve(c, dsth, static_cast<std::size_t>(m) * p);
  if (X == NULL || dst == NULL) return 1;
  if (!ensure_idx(c, m)) return 1;
  if (cudaMemcpy(c->d_idx, idx, static_cast<std::size_t>(m) * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 1;
  const int total = m * p;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  ctx_gather_rows_kernel<<<grid, block>>>(X->mem.get(), n, p,
                                          c->d_idx, m, dst->mem.get());
  return 0;
}

extern "C" int rcupp_d_syrk(void* h, void* Xh, int n, int p,
                            int uplo, double alpha, double beta, void* Ch) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0 || p <= 0) return 1;
  if (uplo != 0 && uplo != 1) return 1;
  Buf* X = resolve(c, Xh, static_cast<std::size_t>(n) * p);
  Buf* C = resolve(c, Ch, static_cast<std::size_t>(p) * p);
  if (X == NULL || C == NULL) return 1;
  /* C[p x p] = alpha * X^T X + beta * C. X is n x p with lda = n, so with
   * CUBLAS_OP_T the cuBLAS "A" is k x n = n x p exactly as stored. ldc = p. */
  cublasStatus_t st = cublasDsyrk(c->handle, fill_mode(uplo), CUBLAS_OP_T,
                                  p, n, &alpha, X->mem.get(), n,
                                  &beta, C->mem.get(), p);
  return st == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

extern "C" int rcupp_d_potrf(void* h, int uplo, int n,
                             void* Ah, int lda, int* info) {
  if (info == NULL) return 1;
  *info = -1;
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || !c->cusolver_ok || n <= 0 || lda < n) return 1;
  if (uplo != 0 && uplo != 1) return 1;
  Buf* A = resolve(c, Ah,
                   static_cast<std::size_t>(n - 1) * lda + n);
  if (A == NULL) return 1;
  int lwork = 0;
  if (cusolverDnDpotrf_bufferSize(c->cusolver, fill_mode(uplo), n,
                                  A->mem.get(), lda, &lwork)
      != CUSOLVER_STATUS_SUCCESS) return 1;
  if (!ensure_work(c, lwork) || !ensure_info(c)) return 1;
  if (cusolverDnDpotrf(c->cusolver, fill_mode(uplo), n,
                       A->mem.get(), lda, c->d_work, lwork, c->d_info)
      != CUSOLVER_STATUS_SUCCESS) return 1;
  if (cudaMemcpy(info, c->d_info, sizeof(int), cudaMemcpyDeviceToHost)
      != cudaSuccess) return 1;
  return 0;
}

extern "C" int rcupp_d_potrs(void* h, int uplo, int n, int nrhs,
                             void* Ah, int lda, void* Bh, int ldb,
                             int* info) {
  if (info == NULL) return 1;
  *info = -1;
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || !c->cusolver_ok || n <= 0 || nrhs <= 0 ||
      lda < n || ldb < n) return 1;
  if (uplo != 0 && uplo != 1) return 1;
  Buf* A = resolve(c, Ah, static_cast<std::size_t>(n - 1) * lda + n);
  Buf* B = resolve(c, Bh, static_cast<std::size_t>(nrhs - 1) * ldb + n);
  if (A == NULL || B == NULL) return 1;
  if (!ensure_info(c)) return 1;
  if (cusolverDnDpotrs(c->cusolver, fill_mode(uplo), n, nrhs,
                       A->mem.get(), lda, B->mem.get(), ldb, c->d_info)
      != CUSOLVER_STATUS_SUCCESS) return 1;
  if (cudaMemcpy(info, c->d_info, sizeof(int), cudaMemcpyDeviceToHost)
      != cudaSuccess) return 1;
  return 0;
}
