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
  ParamBindings.cpp
  ParameterInference.cpp
  ParserBase.cpp
  ParserExprs.cpp
  ParserParamEvaluator.cpp
  ParserStmts.cpp
  SharedState.cpp
  Signatures.cpp
  StructEmitter.cpp
  Traits.cpp
  UncheckedCallEmission.cpp

  DEPENDS
  KGENVersion

  LINK_LIBS PUBLIC
  MSupportFilesystem
  MCompilerSupport
  KGENSupport
  KGENInitAllDialects
  KGENCompilationOptions
  MTelemetry
  AsyncRTRuntime

  MLIRTransformUtils
  MLIRBytecodeReader
  MLIRBytecodeWriter
)
