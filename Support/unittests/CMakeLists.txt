##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(ADT)
add_subdirectory(Entitlements)
add_subdirectory(HTTP)
add_subdirectory(MArchTarget)
add_subdirectory(MDialect)

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  SupportUnitTests

  TEST_NAME
  SupportTests

  AlignedAllocTest.cpp
  ASN1Test.cpp
  BCastTest.cpp
  Base64Test.cpp
  ConfigurationTest.cpp
  DTypeTest.cpp
  DeviceSpecsTest.cpp
  ErrorOrTest.cpp
  ErrorTest.cpp
  FileSystemExtrasTest.cpp
  FunctionExtrasTest.cpp
  HMACTest.cpp
  HostTest.cpp
  JSONTest.cpp
  MathExtras.cpp
  MicroBenchmarkTest.cpp
  RandomTest.cpp
  TelemetryTest.cpp
  TensorShapeTest.cpp
  TensorSpecTest.cpp
  TimeProfilerTest.cpp
  URITest.cpp
  VersionTest.cpp

  LINK_LIBS PRIVATE
  MSupport
  MSupportASN1
  MSupportML
  MSupportGlobals
  MTelemetry
  MProfiling
)
