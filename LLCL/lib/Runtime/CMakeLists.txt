#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

find_package(Threads)

add_mlir_library(LLCLRuntime
  AllocatorType.cpp
  AsyncValue.cpp
  DebuggingAllocators.cpp
  Location.cpp
  MallocAllocator.cpp
  Runtime.cpp
  SingleThreadWorkQueue.cpp
  ThreadPoolWorkQueue.cpp
  WorkQueue.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  LLVMSupport
  MSupport
  Threads::Threads
  )
