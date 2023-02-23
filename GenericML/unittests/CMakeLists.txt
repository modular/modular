##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(GraphCompiler/Conversion)

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

  LINK_LIBS PRIVATE
  GenericMLSupport
  LLCLRuntime
  MSupportML
)
