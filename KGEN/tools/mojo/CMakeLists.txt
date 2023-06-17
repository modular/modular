##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo
  mojo.cpp

  LINK_COMPONENTS
  native
  nativecodegen
  OrcTargetProcess

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
