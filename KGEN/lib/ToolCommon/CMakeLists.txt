##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENToolCommon
  CLOptions.cpp
  CompilationOptions.cpp
  InitAllDialects.cpp
  PassRegistry.cpp

  LINK_LIBS PUBLIC
  DebugInfoDialect
  HLCFDialect
  KGENDialect
  LITDialect
  POPDialect
  CacheDialect
  LowerLIT
  KGENToLLVM
  KGENElaborator
  KGENTransforms
  DebugInfoToLLVM
  MOGGPreElabPasses

  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  )
