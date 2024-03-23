##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  BillingUnitTests

  TEST_NAME
  BillingTests

  BillingTest.cpp

  LINK_LIBS PRIVATE
  MBilling
  MSupport
)
