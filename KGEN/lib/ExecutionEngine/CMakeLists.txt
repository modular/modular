#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_modular_library(KGENExecutionEngine
  ExecutionEngine.cpp

  LINK_COMPONENTS
  OrcJIT
  JITLink
  MC
  nativecodegen

  LINK_LIBS PUBLIC
  KGENDialect
  KGENToLLVM

  MLIRArithmeticToLLVM
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  )
