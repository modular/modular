##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  SUITE_NAME
  LLCLUnitTests

  TEST_NAME
  LLCLTests

  BufferRefTest.cpp

  LINK_LIBS PRIVATE
  LLCLRuntime
)
