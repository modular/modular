##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(Analysis)

add_modular_dialect_library(HLCFDialect
  HLCFAttrs.cpp
  HLCFDialect.cpp
  HLCFInterfaces.cpp
  HLCFOps.cpp
  HLCFOpsCanonicalizers.cpp
  HLCFUtils.cpp

  DEPENDS
  MLIRHLCFIncGen
  MLIRKGENIncGen

  LINK_LIBS PUBLIC
  HLCFAnalysis
  InterpreterInterface

  MLIRIR
  MLIRSideEffectInterfaces
)
