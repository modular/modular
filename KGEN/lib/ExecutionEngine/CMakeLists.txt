##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENExecutionEngine
  ExecutionEngine.cpp

  LINK_COMPONENTS
  OrcJIT
  JITLink
  MC
  native
  nativecodegen

  LINK_LIBS PUBLIC
  LowerZAPToPOP
  KGENDialect
  KGENToLLVM
  KGENEmitObject

  MCache

  MLIRLLVMToLLVMIRTranslation
  MLIRIndexToLLVM
  MLIRTargetLLVMIRExport
  )
