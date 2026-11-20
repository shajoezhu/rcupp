#include <Rcpp.h>
#include <R_ext/Rdynload.h>
#include "rcupp_internal.h"
#include "rcupp_api.h"
#include "rcupp_session.h"
#include "rcupp_ctx.h"

/*
 * The ONLY file containing [[Rcpp::export]] attributes. It is compiled by g++
 * in BOTH build modes and is CUDA-free; it just calls the shared C ABI, whose
 * implementation is chosen at link time by configure (see Makevars.in).
 *
 * Do NOT add [[Rcpp::export]] to cpu_stub.cpp, gpu_ops.cu,
 * lasso_fista_cuda.cu, rcupp_api_*.{cu,cpp} or rcupp_session_*.{cu,cpp}:
 * compileAttributes() scans .cpp/.h but not .cu/.cuh, and duplicate attributes
 * would generate duplicate wrappers and break the build.
 */

// [[Rcpp::export]]
Rcpp::List lasso_fista_cpp(Rcpp::NumericMatrix X, Rcpp::NumericVector y,
                           double lambda, int max_iter, double tol) {
  const int n = X.nrow();
  const int p = X.ncol();

  Rcpp::NumericVector beta(p);
  int iters = 0;
  int converged = 0;
  double obj = 0.0;

  const int rc = rcupp_lasso_fista(X.begin(), y.begin(), n, p, lambda,
                                   max_iter, tol,
                                   beta.begin(), &iters, &obj, &converged);
  if (rc != 0) {
    Rcpp::stop("rcupp_lasso_fista failed (code %d)", rc);
  }

  return Rcpp::List::create(
      Rcpp::Named("beta")       = beta,
      Rcpp::Named("iterations") = iters,
      Rcpp::Named("objective")  = obj,
      Rcpp::Named("converged")  = (bool)converged,
      Rcpp::Named("cuda")       = (bool)rcupp_cuda_available());
}

// [[Rcpp::export]]
bool cuda_available_cpp() {
  return (bool)rcupp_cuda_available();
}

/* ---------------------------------------------------------------------------
 * P1 session API: an opaque void* handle (from rcupp_session_create) is stored
 * in a raw EXTPTRSXP with a C finalizer that calls rcupp_session_destroy at GC.
 * The (n, p) shape is cached on the pointer via Rf_setAttrib() in
 * session_set_data_cpp so session_fit_cpp can size beta_out[p]. All C ABI
 * interaction stays behind plain-C declarations in rcupp_session.h; this file
 * remains CUDA-free.
 * ------------------------------------------------------------------------- */

static void rcupp_session_finalizer(SEXP s) {
  void* h = R_ExternalPtrAddr(s);
  if (h != NULL) {
    rcupp_session_destroy(h);
    R_ClearExternalPtr(s);
  }
}

static SEXP rcupp_check_session(SEXP s) {
  if (TYPEOF(s) != EXTPTRSXP) {
    Rcpp::stop("'session' is not a valid rcupp session (external pointer)");
  }
  void* h = R_ExternalPtrAddr(s);
  if (h == NULL) {
    Rcpp::stop("this rcupp session has already been destroyed");
  }
  return s;
}

// [[Rcpp::export]]
SEXP session_create_cpp() {
  void* h = rcupp_session_create();
  if (h == NULL) Rcpp::stop("rcupp: failed to create a session");
  SEXP s = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(s, rcupp_session_finalizer, TRUE);
  Rf_setAttrib(s, R_ClassSymbol, Rf_mkString("rcupp_session"));
  UNPROTECT(1);
  return s;
}

// [[Rcpp::export]]
void session_set_data_cpp(SEXP s, Rcpp::NumericMatrix X, Rcpp::NumericVector y) {
  rcupp_check_session(s);
  void* h = R_ExternalPtrAddr(s);
  const int rc = rcupp_session_set_data(h, X.begin(), y.begin(),
                                        X.nrow(), X.ncol());
  if (rc != 0) Rcpp::stop("rcupp_session_set_data failed (code %d)", rc);
  /* Cache the shape on the pointer for session_fit_cpp (in-place, persistent). */
  Rf_setAttrib(s, Rf_install("rcupp_n"), Rf_ScalarInteger(X.nrow()));
  Rf_setAttrib(s, Rf_install("rcupp_p"), Rf_ScalarInteger(X.ncol()));
}

// [[Rcpp::export]]
Rcpp::List session_fit_cpp(SEXP s, double lambda, int warm_start,
                           int max_iter, double tol) {
  rcupp_check_session(s);
  void* h = R_ExternalPtrAddr(s);
  const int p = Rf_asInteger(Rf_getAttrib(s, Rf_install("rcupp_p")));
  if (p <= 0) {
    Rcpp::stop("session has no data; call session_set_data() first");
  }

  Rcpp::NumericVector beta(p);
  int iters = 0;
  int converged = 0;
  double obj = 0.0;
  const int rc = rcupp_session_fit(h, lambda, warm_start, max_iter, tol,
                                    beta.begin(), &iters, &obj, &converged);
  if (rc != 0) Rcpp::stop("rcupp_session_fit failed (code %d)", rc);

  return Rcpp::List::create(
      Rcpp::Named("beta")       = beta,
      Rcpp::Named("iterations") = iters,
      Rcpp::Named("objective")  = obj,
      Rcpp::Named("converged")  = (bool)converged,
      Rcpp::Named("cuda")       = (bool)rcupp_cuda_available());
}

// [[Rcpp::export]]
void session_destroy_cpp(SEXP s) {
  if (TYPEOF(s) != EXTPTRSXP) return;
  void* h = R_ExternalPtrAddr(s);
  if (h != NULL) {
    rcupp_session_destroy(h);
    R_ClearExternalPtr(s);
  }
}

/* Release the process-global one-shot pool at unload (see .onUnload in zzz.R). */
// [[Rcpp::export]]
void pool_shutdown_cpp() {
  rcupp_pool_shutdown();
}

/*
 * Publish the fixed-operator + session C ABI to downstream packages via
 * R_RegisterCCallable(). [[Rcpp::init]] makes compileAttributes() inject a call
 * to this function into the generated R_init_rcupp(DllInfo*) in
 * RcppExports.cpp, so registration happens at DLL load -- before any
 * R_GetCCallable() lookup from a LinkingTo: rcupp package.
 *
 * NOTE: compileAttributes() is NOT run by any build/install/check step, so
 * after editing the attributes the generated RcppExports.cpp / RcppExports.R
 * MUST be regenerated (Rcpp::compileAttributes()) and kept in the tree.
 */
// [[Rcpp::init]]
void rcupp_init_ccallables(DllInfo* dll) {
  R_RegisterCCallable("rcupp", "rcupp_cuda_available",
                      (DL_FUNC) &rcupp_cuda_available);
  R_RegisterCCallable("rcupp", "rcupp_gemv",
                      (DL_FUNC) &rcupp_gemv);
  R_RegisterCCallable("rcupp", "rcupp_gemv_t",
                      (DL_FUNC) &rcupp_gemv_t);
  R_RegisterCCallable("rcupp", "rcupp_axpy",
                      (DL_FUNC) &rcupp_axpy);
  R_RegisterCCallable("rcupp", "rcupp_scal",
                      (DL_FUNC) &rcupp_scal);
  R_RegisterCCallable("rcupp", "rcupp_dot",
                      (DL_FUNC) &rcupp_dot);
  R_RegisterCCallable("rcupp", "rcupp_nrm2",
                      (DL_FUNC) &rcupp_nrm2);
  R_RegisterCCallable("rcupp", "rcupp_soft_threshold",
                      (DL_FUNC) &rcupp_soft_threshold);
  R_RegisterCCallable("rcupp", "rcupp_session_create",
                      (DL_FUNC) &rcupp_session_create);
  R_RegisterCCallable("rcupp", "rcupp_session_set_data",
                      (DL_FUNC) &rcupp_session_set_data);
  R_RegisterCCallable("rcupp", "rcupp_session_fit",
                      (DL_FUNC) &rcupp_session_fit);
  R_RegisterCCallable("rcupp", "rcupp_session_destroy",
                      (DL_FUNC) &rcupp_session_destroy);

  /* Phase 0: generic device-context C ABI (inst/include/rcupp.h §4).
   * No [[Rcpp::export]] is added for these; they are C++-callable only,
   * resolved by downstream packages via R_GetCCallable(). */
  R_RegisterCCallable("rcupp", "rcupp_ctx_abi",            (DL_FUNC) &rcupp_ctx_abi);
  R_RegisterCCallable("rcupp", "rcupp_ctx_create",         (DL_FUNC) &rcupp_ctx_create);
  R_RegisterCCallable("rcupp", "rcupp_ctx_destroy",        (DL_FUNC) &rcupp_ctx_destroy);
  R_RegisterCCallable("rcupp", "rcupp_ctx_sync",           (DL_FUNC) &rcupp_ctx_sync);
  R_RegisterCCallable("rcupp", "rcupp_buf_alloc",          (DL_FUNC) &rcupp_buf_alloc);
  R_RegisterCCallable("rcupp", "rcupp_buf_free",           (DL_FUNC) &rcupp_buf_free);
  R_RegisterCCallable("rcupp", "rcupp_buf_upload",         (DL_FUNC) &rcupp_buf_upload);
  R_RegisterCCallable("rcupp", "rcupp_buf_download",       (DL_FUNC) &rcupp_buf_download);
  R_RegisterCCallable("rcupp", "rcupp_buf_fill",           (DL_FUNC) &rcupp_buf_fill);
  R_RegisterCCallable("rcupp", "rcupp_d_gemv",             (DL_FUNC) &rcupp_d_gemv);
  R_RegisterCCallable("rcupp", "rcupp_d_gemv_t",           (DL_FUNC) &rcupp_d_gemv_t);
  R_RegisterCCallable("rcupp", "rcupp_d_axpy",             (DL_FUNC) &rcupp_d_axpy);
  R_RegisterCCallable("rcupp", "rcupp_d_scal",             (DL_FUNC) &rcupp_d_scal);
  R_RegisterCCallable("rcupp", "rcupp_d_copy",             (DL_FUNC) &rcupp_d_copy);
  R_RegisterCCallable("rcupp", "rcupp_d_ger",              (DL_FUNC) &rcupp_d_ger);
  R_RegisterCCallable("rcupp", "rcupp_d_soft_threshold",   (DL_FUNC) &rcupp_d_soft_threshold);
  R_RegisterCCallable("rcupp", "rcupp_d_dot",              (DL_FUNC) &rcupp_d_dot);
  R_RegisterCCallable("rcupp", "rcupp_d_nrm2",             (DL_FUNC) &rcupp_d_nrm2);

  /* B-stage additive parts (ABI still 1): device-side row gather + syrk +
   * Cholesky factor/solve. No [[Rcpp::export]]; C++-callable only, resolved
   * downstream via R_GetCCallable(). */
  R_RegisterCCallable("rcupp", "rcupp_d_gather_rows",      (DL_FUNC) &rcupp_d_gather_rows);
  R_RegisterCCallable("rcupp", "rcupp_d_syrk",             (DL_FUNC) &rcupp_d_syrk);
  R_RegisterCCallable("rcupp", "rcupp_d_potrf",            (DL_FUNC) &rcupp_d_potrf);
  R_RegisterCCallable("rcupp", "rcupp_d_potrs",            (DL_FUNC) &rcupp_d_potrs);
}
