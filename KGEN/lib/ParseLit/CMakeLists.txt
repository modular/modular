##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(ParseLit
  ASTType.cpp
  Diags.cpp
  IRValues.cpp
  LitDecls.cpp
  DocString.cpp
  LitExprCalls.cpp
  LitExprEmitter.cpp
  LitExprNodes.cpp
  Lexer.cpp
  ParserBase.cpp
  ParserDriver.cpp
  ParserExprs.cpp
  ParserStmts.cpp
  SharedState.cpp

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
