##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENToLLVM
  KGENToLLVMPipeline.cpp
  LLVMLoweringUtils.cpp
  LowerCallingConvention.cpp
  LowerControlFlow.cpp
  LowerKGENCoroutines.cpp
  LowerKGENToLLVM.cpp
  LowerPOPToLLVM.cpp
  LowerKGENClosuresToLLVM.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  DebugInfoToLLVM
  DebugInfoTransforms
  HLCFAnalysis
  KGENDialect
  MCompilerSupport
  POPDialect

  MLIRIR
  MLIRIndexToLLVM
  MLIRLLVMCommonConversion
)
