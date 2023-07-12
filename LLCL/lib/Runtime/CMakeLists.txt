##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

find_package(Threads)

add_modular_library(LLCLRuntime
  OBJECT # Always generate an obj.${name} target.

  AsyncValue.cpp
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
  Threads::Threads
  )

add_subdirectory(Globals)
