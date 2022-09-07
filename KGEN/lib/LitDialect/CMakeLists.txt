##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(LitDialect
  LitDialect.cpp
  LitOps.cpp

  DEPENDS
  MLIRLitIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
