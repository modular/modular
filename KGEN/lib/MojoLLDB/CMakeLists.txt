##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoLLDB SHARED
  Plugin.cpp

  ExpressionParser/MojoExpressionVariable.cpp
  REPL/MojoREPL.cpp
  TypeSystem/MojoTypeSystem.cpp

  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  LINK_LIBS
  KGENCompiler
  KGENEmitObject
  LLCLRuntimeGlobals
  ParseLit

  MLIRBuiltinToLLVMIRTranslation

  liblldb
)
