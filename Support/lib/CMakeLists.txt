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
  IPInt.cpp
  JSON.cpp
  MicroBenchmark.cpp
  Process.cpp
  Random.cpp
  ReferenceCounted.cpp
  URI.cpp
  Version.cpp

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
add_subdirectory(ASN1)
add_subdirectory(Benchmark)
add_subdirectory(CrashReporting)
add_subdirectory(Cryptography)
add_subdirectory(Driver)
add_subdirectory(Entitlements)
add_subdirectory(HTTP)
add_subdirectory(MDialect)
add_subdirectory(Threading)
add_subdirectory(MArchTarget)
add_subdirectory(ML)
add_subdirectory(Compiler)
add_subdirectory(DebugInfoDialect)
add_subdirectory(Globals)
add_subdirectory(Telemetry)
add_subdirectory(Profiling)

