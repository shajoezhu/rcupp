#include <cuda_runtime.h>
#include <cublas_v2.h>
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
 * Device code: the only __global__ kernel here (ctx_fill_kernel) is DEFINED
 * and LAUNCHED in this translation unit, so no device symbol crosses .cu
 * files and nvcc needs neither -rdc=true nor a device-link step -- the same
 * rule gpu_ops.cu documents. All other operators call the existing op_*
 * host launchers in gpu_ops.cu or cuBLAS directly.
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

struct Buf {
  DeviceBuffer mem;   /* device double buffer, RAII */
  int n;              /* capacity in doubles */
  explicit Buf(int count)
      : mem(static_cast<std::size_t>(count)), n(count) {}
};

struct Ctx {
  cublasHandle_t handle;
  bool handle_ok;
  std::vector<Buf*> bufs;   /* owned; freed at rcupp_ctx_destroy */
  Ctx() : handle(), handle_ok(false) {}
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
  return static_cast<void*>(c);
}

extern "C" void rcupp_ctx_destroy(void* h) {
  if (h == NULL) return;
  Ctx* c = static_cast<Ctx*>(h);
  for (std::size_t k = 0; k < c->bufs.size(); ++k) delete c->bufs[k];
  c->bufs.clear();
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
