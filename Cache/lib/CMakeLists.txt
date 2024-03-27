##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MCache
  BlobCache.cpp
  CachedTransform.cpp

  Backends/HTTPCacheBackend.cpp

  LINK_LIBS PUBLIC
  LLCLCompilerSupport
  LLCLRuntime
  LLCLSupport
  LLVMSupport
  MCompilerSupport
  MLIRBytecodeWriter
  MSupport
  MSupportFilesystem
  MVersion
  )

target_compile_definitions(MCache PRIVATE
  "MODULAR_DERIVED_DIR=\"${MODULAR_DERIVED_DIR}\"")
