##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  SUITE_NAME
  SupportUnitTests

  TEST_NAME
  SupportTests

  HMACTest.cpp BlobCacheTest.cpp

  LINK_LIBS PRIVATE
  MSupport
)
