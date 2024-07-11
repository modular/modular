##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(Driver/CommonDriver)
add_subdirectory(GraphCompiler/CLOptions)
add_subdirectory(GraphCompiler/Conversion)
add_subdirectory(GraphCompiler/Matchers)
add_subdirectory(GraphCompiler/MODialect)
add_subdirectory(GraphCompiler/RMODialect)
add_subdirectory(API)

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-genericml

  SUITE_NAME
  GenericMLUnitTests

  TEST_NAME
  GenericMLTests

  BufferRefTest.cpp
  ContextTest.cpp
  TensorTest.cpp
  ModelCollectionTest.cpp
  ValueTest.cpp

  LINK_LIBS PRIVATE
  GenericMLSupport
  GenericMLSupportGlobals
  AsyncRTRuntime
  AsyncRTRuntimeGlobals
  MSupportML
  MSupportGlobals
)
