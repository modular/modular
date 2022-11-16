##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(CacheDialect
  CacheAttrs.cpp
  CacheDialect.cpp
  CacheOps.cpp

  DEPENDS
  MLIRCacheIncGen

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  MLIRIR
  MCache
  MSupport
  )
