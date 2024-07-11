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
  CachedTransformTest.cpp
  KeyTest.cpp

  LINK_LIBS PRIVATE
  AsyncRTRuntime
  AsyncRTRuntimeGlobals
  AsyncRTSupport
  MSupport
  MSupportGlobals
  MCache

  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
)
