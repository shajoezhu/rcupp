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
