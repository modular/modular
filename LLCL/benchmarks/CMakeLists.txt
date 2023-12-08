##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  BENCHMARK

  TEST_NAME
  LLCLBenchmarks

  QueueBenchmark.cpp

  LINK_LIBS PRIVATE
  LLCLRuntime
  LLCLRuntimeGlobals
  MSupportGlobals
)
