##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_kgen_kernel(test_kernels
  SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_kernels.mlir
)

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-mojo

  SUITE_NAME
  KGENUnitTests

  TEST_NAME
  KGENTests

  KGENKernelTest.cpp

  DEPENDS
  test_kernels-headers

  LINK_LIBS PRIVATE
  AsyncRTRuntime
  AsyncRTRuntimeGlobals
  MSupportML
  MSupportGlobals
  test_kernels
)

add_subdirectory(mojo-lsp-server)
add_subdirectory(mojo-debug)
