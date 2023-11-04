##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-mojo

  SUITE_NAME
  KGENDialectUnitTests

  TEST_NAME
  KGENDialectTests

  KGENAttrsTest.cpp

  LINK_COMPONENTS
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  LINK_LIBS PRIVATE
  KGENDialect

  MDialect
  MSupportGlobals

  MLIRIR
  MLIRParser
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
)
