##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENEmitObject
  LowerToLLVM.cpp
  LowerToObject.cpp
  StandaloneObject.cpp
  ProduceFunctionDecls.cpp

  LINK_COMPONENTS
  OrcJIT
  JITLink
  MC
  native
  nativecodegen

  LINK_LIBS PUBLIC
  KGENDialect
  KGENToLLVM
  LowerZAPToPOP

  MCache

  MLIRIndexToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRBytecodeWriter
  MLIRBytecodeReader
  )
