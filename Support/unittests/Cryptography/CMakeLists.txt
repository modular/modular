##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  SupportCryptographyUnitTests

  TEST_NAME
  SupportCryptographyTests

  KeypairTest.cpp
  SymmetricTest.cpp

  LINK_LIBS PRIVATE
  MSupportCryptography
)
