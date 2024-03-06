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
  LLVMSupport
  LLCLRuntime
  LLCLSupport
  MSupport
  MSupportFilesystem
  MVersion
  )

target_compile_definitions(MCache PRIVATE
  "MODULAR_DERIVED_DIR=\"${MODULAR_DERIVED_DIR}\"")

add_subdirectory(CacheDialect)
add_subdirectory(CachePasses)
