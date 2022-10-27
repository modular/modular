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
  nativecodegen

  LINK_LIBS PUBLIC
  LowerZAPToPOP
  KGENDialect
  KGENToLLVM
  KGENEmitObject

  MLIRLLVMToLLVMIRTranslation
  MLIRIndexToLLVM
  MLIRTargetLLVMIRExport
  )
