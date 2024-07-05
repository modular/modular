##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(ADT)
add_subdirectory(Compiler)
add_subdirectory(Cryptography)
add_subdirectory(DebugInfoDialect)
add_subdirectory(Entitlements)
add_subdirectory(HTTP)
add_subdirectory(Metering)
add_subdirectory(MArchTarget)
add_subdirectory(ML)

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  SupportUnitTests

  TEST_NAME
  SupportTests

  AlignedAllocTest.cpp
  AssertStreamTest.cpp
  ASN1Test.cpp
  BCastTest.cpp
  Base64Test.cpp
  BufferTest.cpp
  ConfigurationTest.cpp
  ContextTest.cpp
  DTypeTest.cpp
  DeviceSpecsTest.cpp
  ErrorOrTest.cpp
  ErrorTest.cpp
  FileSystemExtrasTest.cpp
  FunctionExtrasTest.cpp
  HostTest.cpp
  IPIntTest.cpp
  IPRationalTest.cpp
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
  MThreading
  MProfiling
)
