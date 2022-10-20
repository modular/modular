##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-llcl

  SUITE_NAME
  LLCLUnitTests

  TEST_NAME
  LLCLTests

  RCBufferTest.cpp

  LINK_LIBS PRIVATE
  LLCLRuntime
)
