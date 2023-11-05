##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  DebugInfoDialectUnitTests

  TEST_NAME
  DebugInfoDialectTest

  DebugInfoAttrsTest.cpp

  LINK_LIBS PRIVATE
  DebugInfoDialect

  MSupport
  MSupportGlobals

  MLIRIR
  MLIRParser
)
