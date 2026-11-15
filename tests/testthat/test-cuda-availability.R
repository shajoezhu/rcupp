test_that("cuda_available() returns a single non-NA logical", {
  available <- cuda_available()
  expect_type(available, "logical")
  expect_length(available, 1L)
  expect_false(is.na(available))
})

test_that(".onAttach only announces the CPU fallback when CUDA is absent", {
  with_mocked_bindings(
    cuda_available = function() FALSE,
    expect_message(rcupp:::.onAttach("lib", "rcupp"), "CPU FISTA fallback"),
    .package = "rcupp"
  )
  with_mocked_bindings(
    cuda_available = function() TRUE,
    expect_silent(rcupp:::.onAttach("lib", "rcupp")),
    .package = "rcupp"
  )
})
