##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo-prime-package-cache
  mojo-prime-package-cache.cpp

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
  MojoToolsCommon

  MLIRBytecodeWriter
  MLIRParser
)
