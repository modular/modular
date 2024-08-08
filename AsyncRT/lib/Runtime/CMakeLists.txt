##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

find_package(Threads)
option(ASYNCRT_WORKER_STATS "Enable collection and print of WorkQueue stats." OFF)
option(ASYNCRT_NO_AFFINITY "Disable affinity in ThreadPoolWorkQueue. Also uses hyper-threading when available." OFF)
if(ASYNCRT_WORKER_STATS)
  add_compile_definitions(ASYNCRT_WORKER_STATS=1)
endif()

if(ASYNCRT_NO_AFFINITY)
  add_compile_definitions(ASYNCRT_NO_AFFINITY=1)
endif()

include(AddTCMalloc)

add_modular_library(AsyncRTRuntime
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
  AsyncRTSupport
  LLVMSupport
  MSupport
  MProfiling
  Threads::Threads
)

add_subdirectory(Globals)
