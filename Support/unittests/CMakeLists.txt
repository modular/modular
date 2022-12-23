##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  SupportUnitTests

  TEST_NAME
  SupportTests

  BCastTest.cpp
  DTypeTest.cpp
  FunctionExtrasTest.cpp
  HMACTest.cpp
  MathExtras.cpp
  MicroBenchmarkTest.cpp
  SmartVariantTest.cpp
  TimeProfilerTest.cpp

  LINK_LIBS PRIVATE
  MSupport
  MSupportML
)
