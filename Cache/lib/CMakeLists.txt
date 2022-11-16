##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MCache
  BlobCache.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  MSupport

  DEPENDS
  MConfig
  )

add_subdirectory(CacheDialect)
add_subdirectory(CachePasses)
