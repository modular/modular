##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  MeteringUnitTests

  TEST_NAME
  MeteringTests

  MeteringContextLogTest.cpp
  MeteringContextTest.cpp

  LINK_LIBS PRIVATE
  MMetering
  MSupport
)

set_tests_properties(MeteringTests PROPERTIES TIMEOUT 20)
