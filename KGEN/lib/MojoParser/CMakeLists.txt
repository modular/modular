##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoParser
  ASTDecl.cpp
  ASTType.cpp
  CallEmission.cpp
  ClosureEmitter.cpp
  DebugInfo.cpp
  DeclResolution.cpp
  DeclResolver.cpp
  DocString.cpp
  EntryPoint.cpp
  ExprEmitter.cpp
  ExprNodes.cpp
  IRValues.cpp
  Lexer.cpp
  MojoUtils.cpp
  OverloadFitness.cpp
  ParsedArgument.cpp
  ParserBase.cpp
  ParserExprs.cpp
  ParserParamEvaluator.cpp
  ParserStmts.cpp
  SharedState.cpp
  StructEmitter.cpp
  UncheckedCallEmission.cpp

  DEPENDS
  KGENVersion

  LINK_LIBS PUBLIC
  MCompilerSupport
  KGENSupport
  KGENToolCommon
  LLCLRuntime
  MTelemetry
  MLIRTransformUtils

  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRIndexDialect
)
