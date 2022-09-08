##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(ZapDialect
  ZapDialect.cpp
  ZapOps.cpp

  DEPENDS
  MLIRZapIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
