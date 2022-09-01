##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(IndexDialect
  IndexAttrs.cpp
  IndexDialect.cpp
  IndexOps.cpp

  DEPENDS
  MLIRIndexIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCastInterfaces
  MLIRSideEffectInterfaces
)
