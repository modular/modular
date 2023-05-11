##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-cache

  SUITE_NAME
  CacheUnitTests

  TEST_NAME
  CacheTests

  BlobCacheTest.cpp
  BufferTest.cpp
  CachedTransformTest.cpp
  KeyTest.cpp

  LINK_LIBS PRIVATE
  CacheDialect
  LLCLRuntime
  LLCLRuntimeGlobals
  LLCLSupport
  MSupport
  MSupportGlobals
  MCache

  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
)

target_compile_definitions(CacheTests
  PRIVATE -DCACHE_TEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/CacheTestsDir)
