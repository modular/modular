##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

find_package(Threads)
option(LLCL_WORKER_STATS "Enable collection and print of WorkQueue stats." OFF)
if(LLCL_WORKER_STATS)
  add_compile_definitions(LLCL_WORKER_STATS=1)
endif()
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
  target_include_directories(LLCLRuntime PUBLIC ${TCMALLOC_PATH}/include)
endif()

add_subdirectory(Globals)
