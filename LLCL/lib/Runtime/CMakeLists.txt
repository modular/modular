#===- CMakeLists.txt -------------------------------------------*- cmake -*-===#
#
# This file is Modular Inc proprietary.
#
#===------------------------------------------------------------------------===#

find_package(Threads)

add_mlir_library(LLCLRuntime
  AsyncValue.cpp
  DebuggingAllocators.cpp
  MallocAllocator.cpp
  Runtime.cpp
  SingleThreadWorkQueue.cpp
  ThreadPoolWorkQueue.cpp
  
  ADDITIONAL_HEADER_DIRS

  DEPENDS

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  MSupport
  LLVMSupport
  Threads::Threads
  )
