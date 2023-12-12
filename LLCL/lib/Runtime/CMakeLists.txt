##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

find_package(Threads)

add_modular_library(LLCLRuntime
  STATIC OBJECT # Always generate an obj.${name} target.

  AsyncValue.cpp
  CompactRuntimePtr.cpp
  DebuggingAllocators.cpp
  MallocAllocator.cpp
  Runtime.cpp
  RuntimeCLOptions.cpp
  SingleThreadWorkQueue.cpp
  ThreadPoolWorkQueue.cpp
  TypeID.cpp

  LINK_LIBS PUBLIC
  LLCLSupport
  LLVMSupport
  MSupport
  MProfiling
  Threads::Threads
)

add_subdirectory(Globals)
