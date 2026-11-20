#include <cmath>
#include <algorithm>
#include <cstddef>
#include <vector>
#include <R_ext/Lapack.h>   /* F77_NAME(dgesvd/dpotrf/dpotrs), FCONE */
#include <R_ext/BLAS.h>     /* F77_NAME(dsyrk/dgemm), FCONE */
#include "rcupp_ctx.h"
#include "rcupp_parts_impl.h"   /* shared level-2 cg_solve / prox_grad_step */

/*
 * CPU-fallback implementation of the PHASE 0 generic device-context C ABI
 * (rcupp_ctx.h). Algorithm-agnostic, resident: a context owns a registry of
 * host buffers; every rcupp_d_* operator runs on those buffers with plain
 * column-major loops. NO CUDA, NO Rcpp, NO RcppArmadillo -- it links cleanly
 * in the no-CUDA build. Downstream composes them into its own solver loop.
 *
 * This object is compiled by g++ ONLY in the no-CUDA build (configure adds
 * it to @EXTRA_OBJECTS@); the CUDA build links rcupp_ctx_cuda.o instead.
 * Both register the SAME 18 symbols, so downstream packages are backend-
 * agnostic (linking-to-rcupp.md § backend-agnostic). Semantics match the F0
 * CPU operators (rcupp_api_cpu.cpp) exactly: X is n x p, column-major (R
 * order), X[i + j*n].
 *
 * "Resident" here just means the data lives in the ctx-owned host vectors
 * between calls -- there is no host<->device distinction on the fallback, but
 * the ownership / lifetime / validation contract is identical to the CUDA
 * backend, so a downstream .cpp compiles and runs unchanged on both.
 */

namespace {

struct Buf {
  std::vector<double> mem;   /* host "device" memory, zero-initialized */
  int n;                     /* capacity in doubles */
  explicit Buf(int count)
      : mem(static_cast<std::size_t>(count), 0.0), n(count) {}
};

struct Ctx {
  bool ok;                  /* mirrors Ctx::handle_ok in the CUDA backend */
  std::vector<Buf*> bufs;   /* owned; freed at rcupp_ctx_destroy */
  Ctx() : ok(true) {}
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

bool ctx_ok(Ctx* c) { return c != NULL && c->ok; }

}  // namespace

extern "C" int rcupp_ctx_abi(void) {
  return 1;
}

extern "C" void* rcupp_ctx_create(void) {
  return static_cast<void*>(new Ctx());
}

extern "C" void rcupp_ctx_destroy(void* h) {
  if (h == NULL) return;
  Ctx* c = static_cast<Ctx*>(h);
  for (std::size_t k = 0; k < c->bufs.size(); ++k) delete c->bufs[k];
  c->bufs.clear();
  delete c;
}

extern "C" void rcupp_ctx_sync(void* h) {
  /* Host loops are synchronous; there is no queued work to wait for. */
  (void)h;
}

extern "C" void* rcupp_buf_alloc(void* h, int n) {
  if (h == NULL || n <= 0) return NULL;
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c)) return NULL;
  Buf* b = new Buf(n);
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
  double* d = b->mem.data();
  for (int i = 0; i < n; ++i) d[i] = host[i];
  return 0;
}

extern "C" int rcupp_buf_download(void* h, void* bh,
                                  double* host, int n) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || host == NULL || n < 0) return 1;
  Buf* b = resolve(c, bh, static_cast<std::size_t>(n));
  if (b == NULL) return 1;
  if (n == 0) return 0;
  const double* d = b->mem.data();
  for (int i = 0; i < n; ++i) host[i] = d[i];
  return 0;
}

extern "C" int rcupp_buf_fill(void* h, void* bh, double value, int n) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n < 0) return 1;
  Buf* b = resolve(c, bh, static_cast<std::size_t>(n));
  if (b == NULL) return 1;
  if (n == 0) return 0;
  double* d = b->mem.data();
  for (int i = 0; i < n; ++i) d[i] = value;
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
  const double* xd = X->mem.data();
  const double* vd = v->mem.data();
  double* od = out->mem.data();
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < p; ++j) s += xd[i + j * n] * vd[j];
    od[i] = s;
  }
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
  const double* xd = X->mem.data();
  const double* rd = r->mem.data();
  double* od = out->mem.data();
  for (int j = 0; j < p; ++j) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += xd[i + j * n] * rd[i];
    od[j] = s;
  }
  return 0;
}

extern "C" int rcupp_d_axpy(void* h, int n, double alpha,
                            void* xh, void* yh) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 1;
  Buf* x = resolve(c, xh, static_cast<std::size_t>(n));
  Buf* y = resolve(c, yh, static_cast<std::size_t>(n));
  if (x == NULL || y == NULL) return 1;
  const double* xd = x->mem.data();
  double* yd = y->mem.data();
  for (int i = 0; i < n; ++i) yd[i] += alpha * xd[i];
  return 0;
}

extern "C" int rcupp_d_scal(void* h, int n, double alpha, void* xh) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 1;
  Buf* x = resolve(c, xh, static_cast<std::size_t>(n));
  if (x == NULL) return 1;
  double* xd = x->mem.data();
  for (int i = 0; i < n; ++i) xd[i] *= alpha;
  return 0;
}

extern "C" int rcupp_d_copy(void* h, int n, void* srch, void* dsth) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 1;
  Buf* src = resolve(c, srch, static_cast<std::size_t>(n));
  Buf* dst = resolve(c, dsth, static_cast<std::size_t>(n));
  if (src == NULL || dst == NULL) return 1;
  const double* sd = src->mem.data();
  double* dd = dst->mem.data();
  for (int i = 0; i < n; ++i) dd[i] = sd[i];
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
  const double* xd = x->mem.data();
  const double* yd = y->mem.data();
  double* ad = A->mem.data();
  for (int j = 0; j < p; ++j) {
    const double ay = alpha * yd[j];
    for (int i = 0; i < n; ++i) ad[i + j * n] += ay * xd[i];
  }
  return 0;
}

extern "C" int rcupp_d_soft_threshold(void* h, void* uh, double t,
                                      void* outh, int n) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 1;
  Buf* u = resolve(c, uh, static_cast<std::size_t>(n));
  Buf* out = resolve(c, outh, static_cast<std::size_t>(n));
  if (u == NULL || out == NULL) return 1;
  const double* ud = u->mem.data();
  double* od = out->mem.data();
  for (int i = 0; i < n; ++i) {
    const double a = std::fabs(ud[i]) - t;
    od[i] = a > 0.0 ? std::copysign(a, ud[i]) : 0.0;
  }
  return 0;
}

extern "C" double rcupp_d_dot(void* h, int n, void* ah, void* bh) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 0.0;
  Buf* a = resolve(c, ah, static_cast<std::size_t>(n));
  Buf* b = resolve(c, bh, static_cast<std::size_t>(n));
  if (a == NULL || b == NULL) return 0.0;
  const double* ad = a->mem.data();
  const double* bd = b->mem.data();
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += ad[i] * bd[i];
  return s;
}

extern "C" double rcupp_d_nrm2(void* h, int n, void* xh) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0) return 0.0;
  Buf* x = resolve(c, xh, static_cast<std::size_t>(n));
  if (x == NULL) return 0.0;
  const double* xd = x->mem.data();
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += xd[i] * xd[i];
  return std::sqrt(s);
}

/* ---------------------------------------------------------------------------
 * B-stage additive parts (ABI stays 1). Host gather loop + R's LAPACK/BLAS.
 * Semantics mirror the CUDA backend exactly: only the uplo triangle of C is
 * written by syrk; potrf/potrs report LAPACK info through the int* out-arg.
 * BLAS_INT / La_INT are both plain int here.
 * ------------------------------------------------------------------------- */

static char uplo_char(int uplo) { return uplo == 1 ? 'U' : 'L'; }

extern "C" int rcupp_d_gather_rows(void* h, void* Xh, int n, int p,
                                   const int* idx, int m, void* dsth) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0 || p <= 0 || m <= 0 || idx == NULL) return 1;
  for (int k = 0; k < m; ++k) {
    if (idx[k] < 0 || idx[k] >= n) return 1;
  }
  Buf* X = resolve(c, Xh, static_cast<std::size_t>(n) * p);
  Buf* dst = resolve(c, dsth, static_cast<std::size_t>(m) * p);
  if (X == NULL || dst == NULL) return 1;
  const double* xd = X->mem.data();
  double* dd = dst->mem.data();
  for (int j = 0; j < p; ++j) {
    for (int i = 0; i < m; ++i) {
      dd[i + static_cast<std::size_t>(j) * m] =
          xd[idx[i] + static_cast<std::size_t>(j) * n];
    }
  }
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
  /* C[p x p] = alpha * X^T X + beta * C. With TRANS='T', BLAS treats A as
   * k x n = n x p (lda = n) and C as N x N = p x p (ldc = p). */
  char up = uplo_char(uplo);
  char tr = 'T';
  int nn = p, kk = n, lda = n, ldc = p;
  F77_NAME(dsyrk)(&up, &tr, &nn, &kk, &alpha, X->mem.data(), &lda,
                  &beta, C->mem.data(), &ldc FCONE FCONE);
  return 0;
}

extern "C" int rcupp_d_potrf(void* h, int uplo, int n,
                             void* Ah, int lda, int* info) {
  if (info == NULL) return 1;
  *info = -1;
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0 || lda < n) return 1;
  if (uplo != 0 && uplo != 1) return 1;
  Buf* A = resolve(c, Ah, static_cast<std::size_t>(n - 1) * lda + n);
  if (A == NULL) return 1;
  char up = uplo_char(uplo);
  int nn = n, ldaa = lda, inf = 0;
  F77_NAME(dpotrf)(&up, &nn, A->mem.data(), &ldaa, &inf FCONE);
  *info = inf;
  return 0;
}

extern "C" int rcupp_d_potrs(void* h, int uplo, int n, int nrhs,
                             void* Ah, int lda, void* Bh, int ldb,
                             int* info) {
  if (info == NULL) return 1;
  *info = -1;
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || n <= 0 || nrhs <= 0 || lda < n || ldb < n) return 1;
  if (uplo != 0 && uplo != 1) return 1;
  Buf* A = resolve(c, Ah, static_cast<std::size_t>(n - 1) * lda + n);
  Buf* B = resolve(c, Bh, static_cast<std::size_t>(nrhs - 1) * ldb + n);
  if (A == NULL || B == NULL) return 1;
  char up = uplo_char(uplo);
  int nn = n, nr = nrhs, ldaa = lda, ldbb = ldb, inf = 0;
  F77_NAME(dpotrs)(&up, &nn, &nr, A->mem.data(), &ldaa,
                   B->mem.data(), &ldbb, &inf FCONE);
  *info = inf;
  return 0;
}

/* ---------------------------------------------------------------------------
 * Reserved / level-2 parts (ABI stays 1). gemm via R's BLAS dgemm, gesvd via
 * R's LAPACK dgesvd, cg_solve / prox_grad_step via the shared
 * rcupp_parts_impl.h that composes the level-1 resident operators. Semantics
 * mirror the CUDA backend exactly (column-major, tall-only gesvd, A read-only).
 * ------------------------------------------------------------------------- */

static char op_char(int trans) { return trans == 1 ? 'T' : 'N'; }

extern "C" int rcupp_d_gemm(void* h, int transa, int transb, int m, int n,
                            int k, double alpha, void* Ah, int lda,
                            void* Bh, int ldb, double beta, void* Ch, int ldc) {
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c) || m <= 0 || n <= 0 || k <= 0) return 1;
  if (transa != 0 && transa != 1) return 1;
  if (transb != 0 && transb != 1) return 1;
  if (lda < (transa == 0 ? m : k)) return 1;
  if (ldb < (transb == 0 ? k : n)) return 1;
  if (ldc < m) return 1;
  const std::size_t a_need = (transa == 0)
      ? static_cast<std::size_t>(k - 1) * lda + m
      : static_cast<std::size_t>(m - 1) * lda + k;
  const std::size_t b_need = (transb == 0)
      ? static_cast<std::size_t>(n - 1) * ldb + k
      : static_cast<std::size_t>(k - 1) * ldb + n;
  const std::size_t c_need = static_cast<std::size_t>(n - 1) * ldc + m;
  Buf* A = resolve(c, Ah, a_need);
  Buf* B = resolve(c, Bh, b_need);
  Buf* C = resolve(c, Ch, c_need);
  if (A == NULL || B == NULL || C == NULL) return 1;
  char ta = op_char(transa), tb = op_char(transb);
  F77_NAME(dgemm)(&ta, &tb, &m, &n, &k, &alpha,
                  A->mem.data(), &lda, B->mem.data(), &ldb,
                  &beta, C->mem.data(), &ldc FCONE FCONE);
  return 0;
}

extern "C" int rcupp_d_gesvd(void* h, void* Ah, int m, int n, int lda,
                             void* Sh, void* Uh, int ldu, void* VTh, int ldvt,
                             int* info) {
  if (info == NULL) return 1;
  *info = -1;
  Ctx* c = static_cast<Ctx*>(h);
  if (!ctx_ok(c)) return 1;
  /* Tall-only to match the CUDA backend's unified contract. */
  if (m < n || n <= 0 || lda < m || ldu < m || ldvt < n) return 1;
  const int k = n;
  const std::size_t a_need  = static_cast<std::size_t>(n - 1) * lda + m;
  const std::size_t u_need  = static_cast<std::size_t>(k - 1) * ldu + m;
  const std::size_t vt_need = static_cast<std::size_t>(n - 1) * ldvt + k;
  Buf* A  = resolve(c, Ah, a_need);
  Buf* S  = resolve(c, Sh, static_cast<std::size_t>(k));
  Buf* U  = resolve(c, Uh, u_need);
  Buf* VT = resolve(c, VTh, vt_need);
  if (A == NULL || S == NULL || U == NULL || VT == NULL) return 1;
  /* dgesvd destroys its input; factor a host copy so A is preserved. */
  std::vector<double> acopy(A->mem.data(), A->mem.data() + a_need);
  /* Generous workspace bound (>= LAPACK's documented minimum for 'S','S');
   * LAPACK happily uses extra workspace, so no query call is needed. */
  const int mn = std::min(m, n), mx = std::max(m, n);
  int lwork = 5 * mx + 5 * mn + 64;
  std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
  char ju = 'S', jv = 'S';
  int inf = 0;
  F77_NAME(dgesvd)(&ju, &jv, &m, &n, acopy.data(), &lda,
                   S->mem.data(), U->mem.data(), &ldu,
                   VT->mem.data(), &ldvt, work.data(), &lwork,
                   &inf FCONE FCONE);
  *info = inf;
  return 0;
}

extern "C" int rcupp_d_cg_solve(void* h, void* X, int n, int p, void* y,
                                double shift, double tol, int max_iter,
                                void* x, int* iters, double* residual,
                                int* converged) {
  return rcupp_cg_solve_impl(h, X, n, p, y, shift, tol, max_iter, x,
                             iters, residual, converged);
}

extern "C" int rcupp_d_prox_grad_step(void* h, void* X, int n, int p, void* y,
                                      void* z, double step, double threshold,
                                      double inv_scale, void* out) {
  return rcupp_prox_grad_step_impl(h, X, n, p, y, z, step, threshold,
                                   inv_scale, out);
}
