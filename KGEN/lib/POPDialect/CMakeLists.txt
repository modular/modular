##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(POPDialect
  POPAttrs.cpp
  POPDialect.cpp
  POPOps.cpp
  POPOpsFolders.cpp
  POPTypes.cpp

  DEPENDS
  MLIRPOPIncGen

  LINK_LIBS PUBLIC
  InterpreterInterface
  KGENDialect
  CODialect
  MCompilerSupport
  MDialect
)
