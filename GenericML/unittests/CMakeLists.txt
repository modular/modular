##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-genericml

  SUITE_NAME
  GenericMLUnitTests

  TEST_NAME
  GenericMLTests

  MOPContextTest.cpp
  TensorTest.cpp

  LINK_LIBS PRIVATE
  GenericMLSupport
  LLCLRuntime
  MOP
  MSupportML
)
