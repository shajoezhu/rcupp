# rcupp 0.1.4

* Development version bumped from 0.1.3.

# rcupp 0.1.3

* Make `rcupp_d_prox_grad_step` genuinely fused on CUDA: it now launches ONE
  `ctx_prox_grad_fused_kernel` per call (single block, shared-memory scratch,
  two `__syncthreads()`), folding `X·z`, residual scaling, `Xᵀ·r` and the prox
  into a single kernel instead of composing nine level-1 operators with four
  `cudaMalloc`/`cudaFree` pairs per call. The CPU fallback still uses the
  shared `rcupp_parts_impl.h` composition, bit-identical; `z` and `out` may
  alias. The public signature, registrations, `configure` and `rcupp_ctx_abi()`
  (still 1) are unchanged, and `rcupp` stays algorithm-agnostic (no lasso /
  ridge / λ logic). This delivers on the documented "one launch per step"
  contract and
  lets downstream hot loops such as `cudaFin::rolling_resident()` collapse an
  ISTA iteration to one launch (demo 3 FIN `3/6` -> `6/6`).

* Add a dependency-free diagnostics and profiling API covering GPU device
  memory, method/function timing, a device-versus-wall time split, and R heap
  memory. Six new R functions (`rcupp_device_memory()`, `rcupp_tic()`,
  `rcupp_toc()`, `rcupp_profile()`, `rcupp_bench()`, `rcupp_memory_report()`)
  expose a global `cudaMemGetInfo()` snapshot, a default-stream CUDA-event
  timer, and combined per-expression diagnostic snapshots. The underlying
  primitives are also published through the public C ABI
  (`inst/include/rcupp.h`) so `LinkingTo: rcupp` packages can profile their own
  C++ solver loops; ABI stays 1. Implemented by both the CUDA backend and the
  CPU fallback, which returns `NA` / sentinel values without error. A new
  `profiling` vignette documents the four metrics and their limits. No new R
  dependencies; solver behavior and the operator hot path are unchanged.

* Development version bumped from 0.1.2.

# rcupp 0.1.2

* Add the four originally reserved / level-2 parts to the Phase 0 device-context
  C ABI (ABI still 1): `rcupp_d_gemm` (general GEMM), `rcupp_d_gesvd` (thin SVD
  of a tall matrix), `rcupp_d_cg_solve` (conjugate-gradient solve of the damped
  normal equations), and `rcupp_d_prox_grad_step` (one fused proximal-gradient
  step). `gemm`/`gesvd` use cuBLAS/cuSOLVER (CUDA) or R's BLAS/LAPACK (CPU
  fallback); the two level-2 parts are shared composites built only from the
  level-1 resident operators. No new `[[Rcpp::export]]`, `rcupp_ctx_abi()` stays
  1, and `rcupp` remains algorithm-agnostic (no finished solver).

* Development version bumped from 0.1.1.

# rcupp 0.1.1

* Development version bumped from 0.1.0.

# rcupp 0.1.0

 * Add four additive B-stage parts to the Phase 0 device-context C ABI
   (ABI still 1): `rcupp_d_gather_rows` (device-side row indexing of a
   resident matrix), `rcupp_d_syrk` (symmetric rank-k), and
   `rcupp_d_potrf`/`rcupp_d_potrs` (Cholesky factor/solve). These let a
   downstream package keep one full matrix resident and refit many row
   subsets without re-uploading. CUDA uses cuBLAS/cuSOLVER; the CPU fallback
   uses R's LAPACK/BLAS. No new `[[Rcpp::export]]` symbols; the existing
   F0/P1/Phase-0 operators are unchanged.
 * Add Phase 0 generic device-context C ABI (ABI version 1): an algorithm-agnostic,
   resident operator layer (`rcupp_ctx_*` / `rcupp_buf_*` / `rcupp_d_*`). Downstream
   packages can now compose GPU-resident operators into their own solver loops
   ("Rcpp + rcupp") with no host-device round trip per operator. Implemented by
   both the CUDA backend and the CPU fallback (backend-agnostic). The existing
   F0 one-shot operators and the lasso P1 session are unchanged; no new
   `[[Rcpp::export]]` symbols are added.

# rcupp 0.0.2

 * First CRAN release.
