##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-elaborate-opt
  kgen-elaborate-opt.cpp

  LINK_COMPONENTS
  native
  nativecodegen

  LINK_LIBS PRIVATE
  DebugInfoToLLVM
  LITDialect
  KGENDialect
  KGENEmitObject
  KGENCompilerRT
  KGENCompiler
  POPDialect
  KGENToLLVM
  KGENElaborator
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
  MLIRReconcileUnrealizedCasts
  MLIRSupport
  MLIRTransforms
  )
