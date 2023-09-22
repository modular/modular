##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENEmitObject
  LowerToLLVM.cpp
  LowerToObject.cpp
  LLVMPassesPipeline.cpp
  StandaloneObject.cpp
  ProduceFunctionDecls.cpp

  DEPENDS
  KGENVersion

  LINK_COMPONENTS
  OrcJIT
  JITLink
  MC

  LINK_LIBS PUBLIC
  KGENToLLVM
  KGENExecutionEngine
  KGENToolCommon
  KGENPipeline

  MCache
  MCompilerSupport
  MTelemetry

  MLIRIndexToLLVM
  MLIRTargetLLVMIRExport
  MLIRBytecodeWriter
  MLIRBytecodeReader
  )
