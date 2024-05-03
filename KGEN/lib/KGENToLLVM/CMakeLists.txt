##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENToLLVM
  DebugInfoTypeConverter.cpp
  KGENToLLVMPipeline.cpp
  LLVMLoweringUtils.cpp
  LowerControlFlow.cpp
  LowerKGENClosuresToLLVM.cpp
  LowerKGENCoroutines.cpp
  LowerKGENToLLVM.cpp
  LowerPOPToLLVM.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  CODialect
  DebugInfoToLLVM
  DebugInfoTransforms
  HLCFAnalysis
  KGENDialect
  KGENSupport
  MCompilerSupport
  POPDialect

  MLIRIR
  MLIRIndexToLLVM
  MLIRLLVMCommonConversion
)
