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

  AllocatorTest.cpp
  AsyncValueTest.cpp
  DebuggingAllocatorsTest.cpp
  ResourceTest.cpp
  RuntimeTest.cpp
  TimerHeapTest.cpp
  TypeIDTest.cpp

  LINK_LIBS PRIVATE
  LLCLRuntime
  LLCLRuntimeGlobals
  MSupportGlobals
)
