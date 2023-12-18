##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(GraphCompiler/Conversion)
add_subdirectory(GraphCompiler/Matchers)
add_subdirectory(GraphCompiler/MODialect)
add_subdirectory(EngineCLI)

if (TARGET modular-pybind-ext)
  add_subdirectory(API)
endif()

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

  LINK_LIBS PRIVATE
  GenericMLSupport
  GenericMLSupportGlobals
  LLCLRuntime
  LLCLRuntimeGlobals
  MSupportML
  MSupportGlobals
)
