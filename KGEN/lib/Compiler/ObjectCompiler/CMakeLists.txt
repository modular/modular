##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENObjectCompiler
  KGENToLLVMPipeline.cpp
  LowerToLLVM.cpp
  LowerToObject.cpp
  LLVMIRUtils.cpp
  LLVMPassesPipeline.cpp
  StandaloneObject.cpp
  ProduceFunctionDecls.cpp

  DEPENDS
  KGENVersion

  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsInfos
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
  MSupport
  LLCLCompilerSupport

  MLIRReconcileUnrealizedCasts
  MLIRIndexToLLVM
  MLIRTargetLLVMIRExport
  MLIRBytecodeWriter
  MLIRBytecodeReader
  )
