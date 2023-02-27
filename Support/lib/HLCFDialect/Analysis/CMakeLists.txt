##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(HLCFAnalysis
  ControlFlowTree.cpp
  DataFlow.cpp

  DEPENDS
  MLIRHLCFIncGen

  LINK_LIBS PUBLIC
  InterpreterInterface
  MDialect
  MLIRIR
  MLIRPass
)
