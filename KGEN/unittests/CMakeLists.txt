##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  SUITE_NAME
  KGENUnitTests

  TEST_NAME
  KGENTests

  KGENInvokeTest.cpp

  LINK_LIBS PRIVATE
  GenericMLSupport
  LLCLRuntime
  MSupportML
)
