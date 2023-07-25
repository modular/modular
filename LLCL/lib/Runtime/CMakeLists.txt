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
  LLCLTelemetry
  LLVMSupport
  MSupport
  Threads::Threads
  )

if (MODULAR_ENABLE_TELEMETRY)
  # For some reason, LLCLTelemetry's compile definitions don't propagate to LLCLRuntime's
  # cpp files (Runtime.cpp, AsyncValue.cpp, etc.), but transitively they do propagate
  # to any libraries that link with LLCLRuntime.
  # This is a workaround to force these definitions on LLCLRuntime's source files.
  # Note that with the exception of MODULAR_ENABLE_TELEMETRY, these come from
  # OpenTelemetry and so this is subject to change in the future.
  add_compile_definitions(MODULAR_ENABLE_TELEMETRY HAVE_CPP_STDLIB ENABLE_LOGS_PREVIEW)
endif()

add_subdirectory(Globals)
