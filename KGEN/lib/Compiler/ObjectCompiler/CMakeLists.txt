##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENObjectCompiler
  CXXHeaderGen.cpp
  KGENToLLVMPipeline.cpp
  LLVMIRUtils.cpp
  LLVMPassesPipeline.cpp
  ObjectCompiler.cpp

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
