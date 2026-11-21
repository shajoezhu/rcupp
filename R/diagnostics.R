# Internal helpers (not exported). Only base R is used: no stats::/utils::
# calls, so DESCRIPTION needs no change.

# Median of a numeric vector, dropping NA; returns NA_real_ if none remain.
# Written locally because stats::median() would require the stats namespace.
.rcupp_median <- function(x) {
  x <- x[!is.na(x)]
  n <- length(x)
  if (n == 0L) return(NA_real_)
  s <- sort(x)
  mid <- n %/% 2L
  if (n %% 2L == 1L) s[mid + 1L] else (s[mid] + s[mid + 1L]) / 2
}

# One clean global device-memory snapshot. The C ABI returns -1.0 sentinels on
# the CPU shim / on failure; those become NA_real_ here, never errors.
.rcupp_device_snapshot <- function() {
  info <- device_mem_info_cpp()
  ok <- isTRUE(info$status == 0L) &&
    !isTRUE(info$free_bytes < 0) && !isTRUE(info$total_bytes < 0)
  free_b <- if (ok) info$free_bytes else NA_real_
  total_b <- if (ok) info$total_bytes else NA_real_
  used_b <- if (ok) total_b - free_b else NA_real_
  cuda <- cuda_available()
  list(
    available = ok,
    cuda = cuda,
    backend = if (isTRUE(cuda)) "cuda" else "cpu-shim",
    free_bytes = free_b,
    total_bytes = total_b,
    used_bytes = used_b,
    free_mb = free_b / 1048576,
    total_mb = total_b / 1048576,
    used_mb = used_b / 1048576
  )
}

#' Global GPU device memory
#'
#' Reports free / total / used device memory from `cudaMemGetInfo()`. This is a
#' whole-device snapshot, not a per-object byte count: other processes and
#' rcupp's own caching allocator affect it.
#'
#' @return A list with `available` (logical), `cuda`, `backend`, and the
#'   `free_*` / `total_*` / `used_*` fields in bytes and MiB. On the CPU
#'   fallback (no CUDA) the numeric fields are `NA_real_` and `available` is
#'   `FALSE`; the call never errors.
#' @export
#' @examples
#' rcupp_device_memory()
rcupp_device_memory <- function() {
  structure(.rcupp_device_snapshot(), class = "rcupp_device_memory")
}

#' Default-stream device timer
#'
#' A manual CUDA-event timer for the legacy default stream (0), which every
#' rcupp operator uses. [rcupp_tic()] records a start event; [rcupp_toc()]
#' records a stop event, synchronizes, and returns device-side elapsed time.
#' This is a **device-vs-wall** split, not per-kernel attribution: it measures
#' GPU-side elapsed (kernels plus synchronous transfers), which you can compare
#' against a host wall clock. There is one process-global event pair, so the
#' timer is not reentrant or nestable; a new `rcupp_tic()` simply re-arms it.
#'
#' @return [rcupp_tic()] returns an invisible logical: `TRUE` when armed,
#'   `FALSE` on the CPU shim or on error. [rcupp_toc()] returns device-side
#'   elapsed milliseconds, or `NA_real_` when CUDA is unavailable / no armed
#'   tic / an error occurred.
#' @name rcupp_device_timer
NULL

#' @rdname rcupp_device_timer
#' @export
#' @examples
#' rcupp_tic()
#' invisible(lasso_fista(matrix(rnorm(60 * 5), 60, 5), rnorm(60), 0.1))
#' rcupp_toc()
rcupp_tic <- function() {
  invisible(isTRUE(device_tic_cpp() == 0L))
}

#' @rdname rcupp_device_timer
#' @export
rcupp_toc <- function() {
  ms <- device_toc_cpp()
  if (isTRUE(ms < 0)) NA_real_ else ms
}

#' Profile one expression (time, device time, and memory)
#'
#' Runs `expr` once and returns a diagnostic snapshot combining four views:
#' host wall/user/sys time, default-stream device time, R heap change, and a
#' whole-device memory delta. This is a **diagnostic snapshot, not a
#' benchmark**; see the details for what each number does and does not mean.
#'
#' @param expr An expression, evaluated in the calling frame. Its value is
#'   returned in `$result` so the work is not wasted.
#'
#' @details The `proc.time()` window brackets `device_tic()` / `device_toc()`,
#'   including the device synchronize in `toc`, so `wall_ms >= device_ms`
#'   always holds. `device_ms` is GPU-side elapsed on the default stream;
#'   `host_overhead_ms = wall_ms - device_ms` is the remainder (R time,
#'   handle / `cudaMalloc` setup, JIT, clock ramp). This is a device-vs-wall
#'   split, **not per-kernel attribution** -- that needs external Nsight
#'   Systems / NVTX instrumentation.
#'
#'   `r_cells_net_delta` is the net change in R heap cells (`gc()` "used"),
#'   and `r_cells_peak` is the peak "max used" after `gc(reset = TRUE)`. These
#'   are retained-heap figures, **not cumulative bytes allocated** (use
#'   `Rprofmem()` for that). `device_*_delta_bytes` are whole-device
#'   `cudaMemGetInfo()` differences and are approximate: transient per-call
#'   buffers are freed before the call returns, and the caching allocator and
#'   other processes add noise.
#'
#'   On WSL/CUDA each operator call costs roughly 5 ms and GPU clocks ramp
#'   lazily, so single-shot device numbers are dominated by overhead. Use
#'   [rcupp_bench()] to see the distribution. On the CPU fallback every
#'   device-derived field is `NA_real_`.
#'
#' @return A list with `expr`, `result`, `cuda`, `backend`; `wall_ms`,
#'   `user_ms`, `sys_ms`, `device_ms`, `host_overhead_ms`; `r_gc_before`,
#'   `r_gc_after`, `r_cells_net_delta`, `r_cells_peak`; and `device_before`,
#'   `device_after`, `device_free_delta_bytes`, `device_used_delta_bytes`.
#' @export
#' @examples
#' x <- 41
#' p <- rcupp_profile(x + 1)
#' p$result
#' p$wall_ms
rcupp_profile <- function(expr) {
  expr_sub <- substitute(expr)
  env <- parent.frame()

  dev_before <- .rcupp_device_snapshot()
  gc_before <- gc(reset = TRUE)
  t0 <- proc.time()
  device_tic_cpp()
  result <- eval(expr_sub, env)
  dev_ms <- device_toc_cpp()
  t1 <- proc.time()
  gc_after <- gc(reset = FALSE)
  dev_after <- .rcupp_device_snapshot()

  dt <- t1 - t0
  wall_ms <- unname(dt["elapsed"]) * 1000
  user_ms <- unname(dt["user.self"]) * 1000
  sys_ms <- unname(dt["sys.self"]) * 1000
  device_ms <- if (isTRUE(dev_ms < 0)) NA_real_ else dev_ms
  host_overhead_ms <- if (is.na(device_ms)) NA_real_ else
    wall_ms - device_ms

  used_before <- sum(gc_before[, "used"])
  used_after <- sum(gc_after[, "used"])
  peak_max <- sum(gc_after[, "max used"])

  free_na <- is.na(dev_before$free_bytes) || is.na(dev_after$free_bytes)
  free_delta <- if (free_na) {
    NA_real_
  } else {
    dev_after$free_bytes - dev_before$free_bytes
  }
  used_na <- is.na(dev_before$used_bytes) || is.na(dev_after$used_bytes)
  used_delta <- if (used_na) {
    NA_real_
  } else {
    dev_after$used_bytes - dev_before$used_bytes
  }

  cuda <- cuda_available()
  structure(list(
    expr = paste(deparse(expr_sub), collapse = " "),
    result = result,
    cuda = cuda,
    backend = if (isTRUE(cuda)) "cuda" else "cpu-shim",
    wall_ms = wall_ms,
    user_ms = user_ms,
    sys_ms = sys_ms,
    device_ms = device_ms,
    host_overhead_ms = host_overhead_ms,
    r_gc_before = gc_before,
    r_gc_after = gc_after,
    r_cells_net_delta = used_after - used_before,
    r_cells_peak = peak_max,
    device_before = dev_before,
    device_after = dev_after,
    device_free_delta_bytes = free_delta,
    device_used_delta_bytes = used_delta
  ), class = "rcupp_profile")
}

#' Repeated timing of one expression
#'
#' Evaluates `expr` `reps` times and reports the distribution of wall time and
#' default-stream device time. Prefer this over a single [rcupp_profile()] when
#' you care about timing: WSL/CUDA clocks ramp lazily and host load swings, so
#' single-shot numbers are noisy. No absolute threshold is implied.
#'
#' @param expr An expression, evaluated in the calling frame each repetition.
#' @param reps Positive integer number of repetitions (default 10).
#'
#' @return A list with `expr`, `reps`, `cuda`, `backend`; per-repetition
#'   `wall_ms` and `device_ms` vectors; and `wall_ms_mean`, `wall_ms_min`,
#'   `wall_ms_max`, `wall_ms_median`, `device_ms_median`. On the CPU fallback
#'   `device_ms` is all `NA_real_` and `device_ms_median` is `NA_real_`.
#' @export
#' @examples
#' b <- rcupp_bench(1 + 1, reps = 3)
#' b$wall_ms
rcupp_bench <- function(expr, reps = 10L) {
  valid <- is.numeric(reps) && length(reps) == 1L && !is.na(reps)
  if (valid) {
    valid <- reps >= 1 && reps == floor(reps)
  }
  if (!valid) {
    stop("'reps' must be a single positive integer", call. = FALSE)
  }
  reps <- as.integer(reps)
  expr_sub <- substitute(expr)
  env <- parent.frame()

  wall <- numeric(reps)
  dev <- rep(NA_real_, reps)
  for (i in seq_len(reps)) {
    t0 <- proc.time()
    device_tic_cpp()
    eval(expr_sub, env)
    d <- device_toc_cpp()
    t1 <- proc.time()
    wall[i] <- unname((t1 - t0)["elapsed"]) * 1000
    if (!isTRUE(d < 0)) dev[i] <- d
  }

  cuda <- cuda_available()
  structure(list(
    expr = paste(deparse(expr_sub), collapse = " "),
    reps = reps,
    cuda = cuda,
    backend = if (isTRUE(cuda)) "cuda" else "cpu-shim",
    wall_ms = wall,
    device_ms = dev,
    wall_ms_mean = mean(wall),
    wall_ms_min = min(wall),
    wall_ms_max = max(wall),
    wall_ms_median = .rcupp_median(wall),
    device_ms_median = .rcupp_median(dev)
  ), class = "rcupp_bench")
}

#' R heap memory report
#'
#' A standalone snapshot of R's heap via `gc()`, plus a device-memory snapshot
#' for context. This is the R-level counterpart to the device-side figures in
#' [rcupp_profile()].
#'
#' @return A list with `cuda`, `backend`, the full `r_gc` matrix,
#'   `r_cells_used`, `r_cells_max_used`, and the `device` snapshot.
#' @export
#' @examples
#' rcupp_memory_report()$r_cells_used
rcupp_memory_report <- function() {
  g <- gc(reset = FALSE)
  cuda <- cuda_available()
  structure(list(
    cuda = cuda,
    backend = if (isTRUE(cuda)) "cuda" else "cpu-shim",
    r_gc = g,
    r_cells_used = sum(g[, "used"]),
    r_cells_max_used = sum(g[, "max used"]),
    device = .rcupp_device_snapshot()
  ), class = "rcupp_memory_report")
}
