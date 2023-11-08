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
  GenericRCRefTest.cpp
  GenericUniquePtrTest.cpp
  ResourceTest.cpp
  RuntimeTest.cpp
  TypeIDTest.cpp
  ThreadAffinityTest.cpp

  LINK_LIBS PRIVATE
  LLCLRuntime
  LLCLRuntimeGlobals
  MSupportGlobals
)
