##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupport
  AlignedAlloc.cpp
  BuildInfo.cpp
  CommonCLOptions.cpp
  Error.cpp
  FileSystemExtras.cpp
  HMAC.cpp
  Host.cpp
  MicroBenchmark.cpp
  Target.cpp
  TimeProfiler.cpp
  URI.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMTarget
  LLVMTargetParser
  MLIRSupport
  MVersion
  )

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
add_subdirectory(MDialect)
add_subdirectory(ML)
add_subdirectory(Compiler)
add_subdirectory(Interpreter)
add_subdirectory(DebugInfoDialect)
add_subdirectory(Globals)
