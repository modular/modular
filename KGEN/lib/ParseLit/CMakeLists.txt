##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(ParseLit
  LitDecls.cpp
  LitExprNodes.cpp
  LitExprParser.cpp
  LitLexer.cpp
  LitParserBase.cpp
  LitStmtParser.cpp
  ParseLit.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  LITDialect
  KGENDialect
  POPDialect

  MLIRIndexDialect
  MLIRTranslateLib
  MLIRSCFDialect
)
