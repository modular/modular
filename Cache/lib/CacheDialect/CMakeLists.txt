##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(CacheDialect
  CacheDialect.cpp
  CacheOps.cpp
  CachedTransform.cpp

  DEPENDS
  MLIRCacheIncGen

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  LLCLRuntime
  LLCLCompilerSupport
  MCache
  MCompilerSupport
  MDialect
  MSupport
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRIR
  )
