.onAttach <- function(libname, pkgname) {
  if (!cuda_available()) {
    packageStartupMessage(
      "rcupp: no CUDA backend linked; using the CPU FISTA fallback ",
      "(full functionality, no GPU acceleration)."
    )
  }
}

# Release the process-global one-shot pool's CUDA resources at a controlled
# time, before the DLL is unloaded / the CUDA context is torn down (avoids
# destruction-order crashes at process exit). Guarded because the DLL may
# already be gone in some unload paths.
.onUnload <- function(libpath) {
  try(pool_shutdown_cpp(), silent = TRUE)
}
