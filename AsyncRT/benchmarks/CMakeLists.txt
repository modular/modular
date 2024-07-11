##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  BENCHMARK

  TEST_NAME
  AsyncRTBenchmarks

  QueueBenchmark.cpp

  LINK_LIBS PRIVATE
  AsyncRTRuntime
  AsyncRTRuntimeGlobals
  MSupportGlobals
)
