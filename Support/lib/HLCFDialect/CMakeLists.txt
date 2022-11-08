##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(HLCFDilaect
  HLCFDialect.cpp
  HLCFOps.cpp

  DEPENDS
  MLIRHLCFIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
