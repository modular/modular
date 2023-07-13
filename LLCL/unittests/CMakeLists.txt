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

  AsyncValueTest.cpp
  DebuggingAllocatorsTest.cpp
  TypeIDTest.cpp
  RuntimeTest.cpp

  LINK_LIBS PRIVATE
  LLCLRuntime
  LLCLRuntimeGlobals
  MSupportGlobals
)
