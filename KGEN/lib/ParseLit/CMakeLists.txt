##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(ParseLit
  ASTType.cpp
  IRValues.cpp
  LitDecls.cpp
  LitDiags.cpp
  LitExprCalls.cpp
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
  DebugInfoDialect
  HLCFDialect
  LITDialect
  KGENDialect
  POPDialect

  MLIRIndexDialect
  MLIRTranslateLib
  MLIRSCFDialect
)
