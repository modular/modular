##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(CODialect
  CODialect.cpp
  COOps.cpp
  COTypes.cpp
  COUtils.cpp

  DEPENDS
  MLIRCOIncGen
  MLIRCODialectBytecodeIncGen

  LINK_LIBS PUBLIC
  KGENDialect
)
