##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENExecutionEngine
  ExecutionEngine.cpp

  LINK_COMPONENTS
  OrcJIT
  OrcShared
  OrcTargetProcess
  JITLink
  MC

  LINK_LIBS PUBLIC
  KGENDialect
  KGENToLLVM
  KGENEmitObject

  MCache

  MLIRLLVMToLLVMIRTranslation
  MLIRIndexToLLVM
  MLIRTargetLLVMIRExport
  )
