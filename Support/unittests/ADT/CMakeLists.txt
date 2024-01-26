##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  SupportADTUnitTests

  TEST_NAME
  SupportADTTests

  SmartVariantTest.cpp
  GenericRCRefTest.cpp
  GenericUniquePtrTest.cpp

  LINK_LIBS PRIVATE
  MSupport
  MSupportGlobals
)
