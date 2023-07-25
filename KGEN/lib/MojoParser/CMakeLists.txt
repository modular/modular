##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(MojoParser
  ASTDeclView.cpp
  ASTDeclRef.cpp
  ASTType.cpp
  CallEmission.cpp
  CodeComplete.cpp
  DeclResolver.cpp
  Diags.cpp
  DocString.cpp
  ExprEmitter.cpp
  IRValues.cpp
  ExprNodes.cpp
  Lexer.cpp
  ParserBase.cpp
  ParserDriver.cpp
  ParserDriverREPL.cpp
  ParserExprs.cpp
  ParserStmts.cpp
  SharedState.cpp
  ClosureEmitter.cpp

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
