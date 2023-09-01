##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(HLCFDialect
  HLCFAttrs.cpp
  HLCFDialect.cpp
  HLCFInterfaces.cpp
  HLCFOps.cpp
  HLCFOpsCanonicalizers.cpp
  HLCFUtils.cpp

  DEPENDS
  MLIRDebugInfoIncGen
  MLIRHLCFIncGen
  MLIRKGENIncGen

  LINK_LIBS PUBLIC
  InterpreterInterface

  MLIRIR
  MLIRSideEffectInterfaces
)

add_subdirectory(Analysis)
