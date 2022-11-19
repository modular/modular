##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MCache
  BlobCache.cpp
  Buffer.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  LLCLRuntime
  LLCLSupport
  MSupport

  DEPENDS
  MConfig
  )

add_subdirectory(CacheDialect)
add_subdirectory(CachePasses)
