##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(Analysis)

add_modular_dialect_library(HLCFDialect
  HLCFDialect.cpp
  HLCFInterfaces.cpp
  HLCFOps.cpp

  DEPENDS
  MLIRHLCFIncGen

  LINK_LIBS PUBLIC
  HLCFAnalysis
  InterpreterInterface

  MLIRIR
  MLIRSideEffectInterfaces
)
