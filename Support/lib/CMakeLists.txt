##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupport
  OBJECT # Always generate an obj.${name} target.

  AlignedAlloc.cpp
  Base64.cpp
  Buffer.cpp
  BuildInfo.cpp
  CPUCache.cpp
  CommonCLOptions.cpp
  Configuration.cpp
  DeviceSpecs.cpp
  DynamicLibrary.cpp
  Error.cpp
  ErrorOr.cpp
  FileSystemExtras.cpp
  HMAC.cpp
  IPInt.cpp
  JSON.cpp
  MicroBenchmark.cpp
  PlatformLibNames.cpp
  Process.cpp
  Progress.cpp
  Random.cpp
  ReferenceCounted.cpp
  Stats.cpp
  URI.cpp
  Version.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMTarget
  LLVMTargetParser
  MLIRSupport
  MVersion
  )

set_property(
  SOURCE PlatformLibNames.cpp
  APPEND
  PROPERTY COMPILE_DEFINITIONS
  SHARED_LIBRARY_PREFIX=\"${CMAKE_SHARED_LIBRARY_PREFIX}\"
  SHARED_LIBRARY_SUFFIX=\"${CMAKE_SHARED_LIBRARY_SUFFIX}\"
  STATIC_LIBRARY_PREFIX=\"${CMAKE_STATIC_LIBRARY_PREFIX}\"
  STATIC_LIBRARY_SUFFIX=\"${CMAKE_STATIC_LIBRARY_SUFFIX}\")

string(REPLACE ";" " " LLVM_TARGETS_BUILT "${LLVM_TARGETS_TO_BUILD}")
set(BUILD_INFO_DEFINITIONS
  "MODULAR_KERNELS_BUILD_TYPE=\"${MODULAR_KERNELS_BUILD_TYPE}\""
  "LLVM_TARGETS_BUILT=\"${LLVM_TARGETS_BUILT}\"")
set_source_files_properties(BuildInfo.cpp PROPERTIES
  COMPILE_DEFINITIONS "${BUILD_INFO_DEFINITIONS}")
unset(BUILD_INFO_DEFINITIONS)
unset(LLVM_TARGETS_BUILT)

add_subdirectory(Filesystem)
add_subdirectory(ADT)
add_subdirectory(ASN1)
add_subdirectory(CrashReporting)
add_subdirectory(Cryptography)
add_subdirectory(Driver)
add_subdirectory(HTTP)
add_subdirectory(MDialect)
add_subdirectory(Threading)
add_subdirectory(MArchTarget)
add_subdirectory(ML)
add_subdirectory(Compiler)
add_subdirectory(DebugInfoDialect)
add_subdirectory(Globals)
add_subdirectory(CUDA)
add_subdirectory(Profiling)

# Entitlements is a composite, it requires ASN1, Cryptography, and HTTP.
add_subdirectory(Entitlements)

# Telemetry requires Entitlements.
add_subdirectory(Telemetry)

add_subdirectory(Frameworks)
