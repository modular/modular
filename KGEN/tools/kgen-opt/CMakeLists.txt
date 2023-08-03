##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-opt
  kgen-opt.cpp

  LINK_COMPONENTS
  native
  nativecodegen

  LINK_LIBS PRIVATE
  DebugInfoToLLVM
  LITDialect
  KGENCompiler
  KGENDialect
  KGENElaborator
  POPDialect
  KGENToLLVM
  KGENTransforms
  LowerLIT

  LLCLRuntimeGlobals
  MSupportGlobals

  MLIRBuiltinToLLVMIRTranslation
  MLIRIR
  MLIRIndexDialect
  MLIRIndexToLLVM
  MLIRLLVMDialect
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms
  )
