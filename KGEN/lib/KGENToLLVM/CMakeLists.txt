##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENToLLVM
  KGENToLLVMPipeline.cpp
  LLVMLoweringUtils.cpp
  LowerKGENCoroutines.cpp
  LowerKGENToLLVM.cpp
  LowerKGENToPOP.cpp
  LowerPOPToLLVM.cpp
  LowerPOPClosuresToLLVM.cpp
  LowerSCFToLLVM.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  DebugInfoToLLVM
  DebugInfoTransforms
  HLCFToLLVM
  KGENDialect
  POPDialect
  MCompilerSupport

  MLIRIR
  MLIRIndexToLLVM
  MLIRLLVMCommonConversion
  )
