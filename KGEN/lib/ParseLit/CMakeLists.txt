##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(ParseLit
  ASTType.cpp
  IRValues.cpp
  LitDecls.cpp
  LitExprEmitter.cpp
  LitExprNodes.cpp
  LitExprParser.cpp
  LitLexer.cpp
  LitParserBase.cpp
  LitSharedState.cpp
  LitStmtParser.cpp
  ParseLit.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  HLCFDialect
  LITDialect
  KGENDialect
  POPDialect
  ZAPDialect

  MLIRIndexDialect
  MLIRTranslateLib
  MLIRSCFDialect
)
