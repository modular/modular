##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

find_package(Threads)
include(AddTCMalloc)

add_modular_library(LLCLRuntime
  STATIC OBJECT # Always generate an obj.${name} target.

  AsyncValue.cpp
  CompactRuntimePtr.cpp
  DebuggingAllocators.cpp
  MallocAllocator.cpp
  TCMallocAllocator.cpp
  Runtime.cpp
  RuntimeCLOptions.cpp
  SingleThreadWorkQueue.cpp
  ThreadPoolWorkQueue.cpp

  LINK_LIBS PUBLIC
  LLCLSupport
  LLVMSupport
  MSupport
  MProfiling
  Threads::Threads
)

if (TCMALLOC_LIBRARY)
  target_link_libraries(LLCLRuntime PUBLIC libtcmalloc)
  target_link_libraries(obj.LLCLRuntime PUBLIC libtcmalloc)
endif()

add_subdirectory(Globals)
