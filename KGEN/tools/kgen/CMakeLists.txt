##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen
  kgen.cpp

  DEPENDS
  KGENCompilerRTShared

  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  LINK_LIBS PRIVATE
  KGENToolCommon
  KGENCompiler
  KGENEmitObject
  KGENExecutionEngine
  LLCLCompilerSupport
  LLCLRuntimeGlobals
  MArchTarget
  MSupportGlobals
  MSupportInit
  MVersion
  MojoParser

  MLIRBytecodeWriter
  MLIRParser
  )
