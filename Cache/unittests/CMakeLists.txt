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

  LINK_LIBS PRIVATE
  MSupport
  MCache
)
