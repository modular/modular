##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_kgen_kernel(test_kernels
  SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_kernels.mlir)

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-kgen

  SUITE_NAME
  KGENUnitTests

  TEST_NAME
  KGENTests

  KGENInvokeTest.cpp
  KGENKernelTest.cpp

  LINK_LIBS PRIVATE
  GenericMLSupport
  LLCLRuntime
  MSupportML
  test_kernels
)
