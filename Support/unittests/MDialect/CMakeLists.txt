##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  SupportMDialectUnitTests

  TEST_NAME
  SupportMDialectTests

  MAttrsTest.cpp

  LINK_LIBS PRIVATE
  MDialect
)
