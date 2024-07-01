##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  SupportMLUnitTests

  TEST_NAME
  SupportMLTests

  FillTest.cpp
  Permutation.cpp

  LINK_LIBS PRIVATE
  MSupportML
)
