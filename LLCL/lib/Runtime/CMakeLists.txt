##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

find_package(Threads)

add_modular_library(LLCLRuntime
  AsyncValue.cpp
  DebuggingAllocators.cpp
  MallocAllocator.cpp
  Runtime.cpp
  SingleThreadWorkQueue.cpp
  ShardedSemaphoreWorkQueue.cpp
  ThreadPoolWorkQueue.cpp

  LINK_LIBS PUBLIC
  LLCLSupport
  LLVMSupport
  MSupport
  Threads::Threads
  )
