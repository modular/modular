##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
add_modular_library(MSupport
  OBJECT # Always generate an obj.${name} target.

  AlignedAlloc.cpp
  Base64.cpp
  BuildInfo.cpp
  CommonCLOptions.cpp
  Configuration.cpp
  DeviceSpecs.cpp
  Error.cpp
  ErrorOr.cpp
  FileSystemExtras.cpp
  HMAC.cpp
  Host.cpp
  IPInt.cpp
  JSON.cpp
  MicroBenchmark.cpp
  Process.cpp
  Target.cpp
  TimeProfiler.cpp
  URI.cpp
  Version.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMTarget
  LLVMTargetParser
  MLIRSupport
  MVersion
  )

if (APPLE)
  target_include_directories(MSupport PRIVATE ${CLANG_INCLUDE_DIRS})
  target_link_libraries(MSupport PRIVATE clangBasic)
endif ()

#TODO: Move this to a cmake module.
if (MODULAR_USE_TCMALLOC AND UNIX)

  if (APPLE)
    file(GLOB_RECURSE Tcmalloc_LIBRARY
      /opt/homebrew/Cellar/gperftools/*libtcmalloc_minimal.dylib
    )
  else()
    file(GLOB_RECURSE Tcmalloc_LIBRARY
      /usr/lib/*libtcmalloc_minimal.so*
    )
  endif()

  target_link_libraries(MSupport PUBLIC ${Tcmalloc_LIBRARY})
endif()

string(REPLACE ";" " " LLVM_TARGETS_BUILT "${LLVM_TARGETS_TO_BUILD}")
set(BUILD_INFO_DEFINITIONS
  "MODULAR_KERNELS_BUILD_TYPE=\"${MODULAR_KERNELS_BUILD_TYPE}\""
  "LLVM_TARGETS_BUILT=\"${LLVM_TARGETS_BUILT}\"")
set_source_files_properties(BuildInfo.cpp PROPERTIES
  COMPILE_DEFINITIONS "${BUILD_INFO_DEFINITIONS}")
unset(BUILD_INFO_DEFINITIONS)
unset(LLVM_TARGETS_BUILT)

add_subdirectory(ADT)
add_subdirectory(Benchmark)
add_subdirectory(CrashReporting)
add_subdirectory(Driver)
add_subdirectory(MDialect)
add_subdirectory(MArchTarget)
add_subdirectory(ML)
add_subdirectory(Compiler)
add_subdirectory(DebugInfoDialect)
add_subdirectory(Globals)
add_subdirectory(Telemetry)
