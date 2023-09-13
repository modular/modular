##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoParser
  ASTDeclView.cpp
  ASTDeclRef.cpp
  ASTType.cpp
  CallEmission.cpp
  ClosureEmitter.cpp
  CodeComplete.cpp
  DeclResolver.cpp
  Diags.cpp
  DocString.cpp
  ExprEmitter.cpp
  ExprNodes.cpp
  IRValues.cpp
  Lexer.cpp
  ParserBase.cpp
  ParserDriver.cpp
  ParserDriverREPL.cpp
  ParserExprs.cpp
  ParserStmts.cpp
  SharedState.cpp
  StructEmitter.cpp
  UncheckedCallEmission.cpp
  Utils.cpp

  DEPENDS
  KGENVersion
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  InitAllKGENDialects
  LLCLRuntime
  MTelemetry

  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRIndexDialect
  MLIRTranslateLib
)
