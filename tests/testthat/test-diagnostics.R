# Diagnostics / profiling tests. Deliberately backend-agnostic: they must pass
# identically on a CUDA build and on the no-CUDA CPU fallback, so there are no
# skip_* calls and no absolute timing thresholds. Expressions are cheap pure-R
# to stay fast even on WSL/CUDA where each device call has fixed overhead.

test_that("rcupp_device_memory() reports a whole-device snapshot", {
  m <- rcupp_device_memory()
  expect_s3_class(m, "rcupp_device_memory")
  expect_identical(m$cuda, cuda_available())
  expect_true(m$backend %in% c("cuda", "cpu-shim"))
  expect_type(m$available, "logical")
  nms <- c("free_bytes", "total_bytes", "used_bytes")
  expect_true(all(nms %in% names(m)))
  for (nm in nms) {
    v <- m[[nm]]
    expect_true(is.na(v) || (is.numeric(v) && v >= 0))
  }
})

test_that("rcupp_tic()/rcupp_toc() are backend-agnostic", {
  armed <- rcupp_tic()
  expect_type(armed, "logical")
  expect_length(armed, 1L)
  ms <- rcupp_toc()
  expect_true(is.na(ms) || (is.numeric(ms) && ms >= 0))
})

test_that("rcupp_profile() keeps the result and reports four views", {
  x <- 41
  p <- rcupp_profile(x + 1)
  expect_s3_class(p, "rcupp_profile")
  expect_identical(p$result, 42)
  expect_identical(p$cuda, cuda_available())
  expect_true(p$backend %in% c("cuda", "cpu-shim"))
  expect_true(is.numeric(p$wall_ms) && p$wall_ms >= 0)
  expect_true(is.numeric(p$user_ms))
  expect_true(is.numeric(p$sys_ms))
  expect_true(is.na(p$device_ms) || p$device_ms >= 0)
  expect_true(is.na(p$host_overhead_ms) || is.finite(p$host_overhead_ms))
  expect_true(is.numeric(p$r_cells_net_delta))
  expect_true(is.numeric(p$r_cells_peak))
  expect_true(all(c("device_before", "device_after") %in% names(p)))
})

test_that("rcupp_bench() returns per-repetition distributions", {
  b <- rcupp_bench(1 + 1, reps = 3L)
  expect_s3_class(b, "rcupp_bench")
  expect_identical(b$reps, 3L)
  expect_length(b$wall_ms, 3L)
  expect_length(b$device_ms, 3L)
  expect_true(all(is.numeric(b$wall_ms)))
  expect_true(all(is.na(b$device_ms) | b$device_ms >= 0))
  expect_true(is.numeric(b$wall_ms_median))
  expect_true(is.numeric(b$device_ms_median))
})

test_that("rcupp_bench() validates 'reps'", {
  expect_error(rcupp_bench(1 + 1, reps = 0), "positive integer")
  expect_error(rcupp_bench(1 + 1, reps = 2.5), "positive integer")
  expect_error(rcupp_bench(1 + 1, reps = -1), "positive integer")
})

test_that("rcupp_memory_report() reports R heap and device context", {
  r <- rcupp_memory_report()
  expect_s3_class(r, "rcupp_memory_report")
  expect_identical(r$cuda, cuda_available())
  expect_true(r$backend %in% c("cuda", "cpu-shim"))
  expect_true(is.numeric(r$r_cells_used) && r$r_cells_used >= 0)
  expect_true(is.numeric(r$r_cells_max_used))
  expect_true(is.list(r$device))
  expect_true(r$device$backend %in% c("cuda", "cpu-shim"))
})
