##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

ignore_aws_compile_warnings()

add_modular_unittest(
  SUITE_NAME
  AWSBillingUnitTests

  TEST_NAME
  AWSBillingTests

  BillingTest.cpp

  LINK_LIBS PRIVATE
  MAWSBilling
  MSupport
)
