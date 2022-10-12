##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENEmitObject
  BytecodeUtils.cpp
  LowerToLLVM.cpp
  LowerToObject.cpp
  StandaloneObject.cpp

  LINK_COMPONENTS
  OrcJIT
  JITLink
  MC
  nativecodegen

  LINK_LIBS PUBLIC
  IndexToLLVM
  KGENDialect
  KGENToLLVM
  LowerZAPToPOP

  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRBytecodeWriter
  MLIRBytecodeReader
  )
