##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen
  kgen.cpp

  DEPENDS
  orc-rt-cas
  compiler-rt-cas

  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  LINK_LIBS PRIVATE
  KGENCLOptions
  KGENCompiler
  KGENDialect
  KGENEmitObject
  KGENExecutionEngine
  LITDialect
  LLCLRuntimeGlobals
  MSupportGlobals
  POPDialect
  MVersion

  MLIRBuiltinToLLVMIRTranslation
  MLIRBytecodeWriter
  MLIRIndexDialect
  MLIRParser
  )
