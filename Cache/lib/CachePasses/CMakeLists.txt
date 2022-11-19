##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(CachePasses
  CachePasses.cpp

  DEPENDS
  MLIRCachePassesIncGen

  LINK_LIBS PUBLIC
  CacheDialect
  LLCLRuntime
  LLCLSupport
  MLIRIR
  )
