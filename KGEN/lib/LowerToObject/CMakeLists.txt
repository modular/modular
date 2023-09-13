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
  CacheDialect
  DebugInfoToLLVM
  KGENDialect
  KGENToLLVM
  KGENTransforms
  LowerLIT

  MCache
  MCompilerSupport
  MTelemetry

  MLIRIndexToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRBytecodeWriter
  MLIRBytecodeReader
  )
