# rcupp 0.1.1

* Development version bumped from 0.1.0.

# rcupp 0.1.0

 * Add Phase 0 generic device-context C ABI (ABI version 1): an algorithm-agnostic,
   resident operator layer (`rcupp_ctx_*` / `rcupp_buf_*` / `rcupp_d_*`). Downstream
   packages can now compose GPU-resident operators into their own solver loops
   ("Rcpp + rcupp") with no host-device round trip per operator. Implemented by
   both the CUDA backend and the CPU fallback (backend-agnostic). The existing
   F0 one-shot operators and the lasso P1 session are unchanged; no new
   `[[Rcpp::export]]` symbols are added.

# rcupp 0.0.2

 * First CRAN release.
