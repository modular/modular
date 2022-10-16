##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(ParseLit
  LitExprNodes.cpp
  LitLexer.cpp
  ParseLit.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  LITDialect
  MLIRSCFDialect
  IndexDialect
  KGENDialect
  POPDialect

  MLIRTranslateLib
)
