##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(HLCFAnalysis
  ControlFlowTree.cpp
  DataFlow.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
