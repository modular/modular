##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(ADT)

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  SupportUnitTests

  TEST_NAME
  SupportTests

  AlignedAllocTest.cpp
  BCastTest.cpp
  DTypeTest.cpp
  ErrorTest.cpp
  ErrorOrTest.cpp
  FunctionExtrasTest.cpp
  HostTest.cpp
  HMACTest.cpp
  JSONTest.cpp
  MathExtras.cpp
  MicroBenchmarkTest.cpp
  TensorShapeTest.cpp
  TensorSpecTest.cpp
  TimeProfilerTest.cpp
  URITest.cpp

  LINK_LIBS PRIVATE
  MSupport
  MSupportML
  MSupportGlobals
)
