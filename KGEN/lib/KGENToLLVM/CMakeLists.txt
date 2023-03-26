##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENToLLVM
  HLCFToLLVM.cpp
  KGENToLLVMPipeline.cpp
  LLVMLoweringUtils.cpp
  LowerControlFlow.cpp
  LowerKGENCoroutines.cpp
  LowerKGENToLLVM.cpp
  LowerKGENToPOP.cpp
  LowerPOPToLLVM.cpp
  LowerPOPClosuresToLLVM.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  DebugInfoToLLVM
  DebugInfoTransforms
  KGENDialect
  POPDialect
  MCompilerSupport

  MLIRIR
  MLIRIndexToLLVM
  MLIRLLVMCommonConversion
  )
