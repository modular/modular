##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENToLLVM
  KGENToLLVMPipeline.cpp
  LLVMLoweringUtils.cpp
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
  KGENDialect
  POPDialect
  MCompilerSupport

  MLIRIR
  MLIRIndexToLLVM
  MLIRLLVMCommonConversion
  )
