##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(ZAPDialect
  ZAPDialect.cpp
  ZAPOps.cpp

  DEPENDS
  MLIRZAPIncGen

  LINK_LIBS PUBLIC
  MetaDialect
  MLIRSideEffectInterfaces

  MLIRIR
)
