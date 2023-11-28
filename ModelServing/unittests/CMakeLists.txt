##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  SUITE_NAME
  UnitCalculatorUnitTests

  TEST_NAME
  UnitCalculatorTests

  UnitCalculatorTest.cpp

  LINK_LIBS PRIVATE
  MCUCalculator
  MThreading
)

if(MODULAR_WITH_AWS_BILLING)
  add_subdirectory(AWS)
endif()
