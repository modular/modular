##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  SUITE_NAME
  SupportUnitTests

  TEST_NAME
  SupportTests

  BlobCacheTest.cpp
  HMACTest.cpp
  MicroBenchmarkTest.cpp
  SIMDTest.cpp
  TimeProfilerTest.cpp

  LINK_LIBS PRIVATE
  MSupport
)
