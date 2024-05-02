##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(HLCFAnalysis
  CFG.cpp
  ControlFlowTree.cpp

  LINK_LIBS PUBLIC
  HLCFDialect
  MDialect
  MLIRIR
  MLIRPass
)
