##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  SupportEntitlementUnitTests

  TEST_NAME
  SupportEntitlementTests

  EntitlementsTest.cpp

  LINK_LIBS PRIVATE
  MSupportEntitlements
)
