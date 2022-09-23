##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(ZAPDialect
  OpaqueObjectImpl.cpp
  ZAPDialect.cpp
  ZAPOps.cpp
  ZAPTypes.cpp

  DEPENDS
  MLIRZAPIncGen

  LINK_LIBS PUBLIC
  MetaDialect
  MLIRSideEffectInterfaces

  MLIRIR
)
