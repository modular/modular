##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  SUITE_NAME
  GenericMLUnitTests

  TEST_NAME
  GenericMLTests

  BufferTest.cpp

  LINK_LIBS PRIVATE
  LLCLRuntime
)
