##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MCache
  BlobCache.cpp
  CachedTransform.cpp
  CacheTelemetryContext.cpp

  LINK_LIBS PUBLIC
  AsyncRTCompilerSupport
  AsyncRTRuntime
  AsyncRTSupport
  AsyncRTCompilerSupport
  LLVMSupport
  MCompilerSupport
  MLIRBytecodeWriter
  MSupport
  MSupportEntitlements
  MSupportFilesystem
  MTelemetry
  MVersion
  )

target_compile_definitions(MCache PRIVATE
  "MODULAR_DERIVED_DIR=\"${MODULAR_DERIVED_DIR}\"")
