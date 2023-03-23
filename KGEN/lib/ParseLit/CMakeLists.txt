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
  LitDocString.cpp
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
  KGENVersion
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  CacheDialect
  DebugInfoDialect
  HLCFDialect
  LITDialect
  LLCLRuntime
  KGENDialect
  POPDialect

  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRIndexDialect
  MLIRTranslateLib
)
