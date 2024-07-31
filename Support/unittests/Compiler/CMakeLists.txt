##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  SupportCompilerUnitTests

  TEST_NAME
  SupportCompilerTests

  DomainAwareReplacerTest.cpp
  MLIRDenseAttrTest.cpp

  LINK_LIBS PRIVATE
  MCompilerSupport
)
