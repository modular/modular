##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  SUITE_NAME
  AsyncRTUnitTests

  TEST_NAME
  AsyncRTTests

  AllocatorTest.cpp
  AsyncValueTest.cpp
  DebuggingAllocatorsTest.cpp
  ResourceTest.cpp
  RuntimeTest.cpp
  TimerHeapTest.cpp
  TypeIDTest.cpp

  LINK_LIBS PRIVATE
  AsyncRTRuntime
  AsyncRTRuntimeGlobals
  MSupportGlobals
)
