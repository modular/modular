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
  RuntimeCLOptions.cpp
  SingleThreadWorkQueue.cpp
  ThreadPoolWorkQueue.cpp
  Threading.cpp
  TypeID.cpp

  LINK_LIBS PUBLIC
  LLCLSupport
  LLVMSupport
  MSupport
  Threads::Threads
  )

add_subdirectory(Globals)
