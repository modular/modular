##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENToLLVM
  KGENToLLVMPipeline.cpp
  LowerKGENToLLVM.cpp
  LowerKGENToPOP.cpp
  LowerPOPToLLVM.cpp
  LowerSCFToLLVM.cpp
  LLVMLoweringUtils.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  HLCFToLLVM
  KGENDialect
  POPDialect
  MCompilerSupport

  MLIRIR
  MLIRIndexToLLVM
  MLIRLLVMCommonConversion
  MLIRSCFDialect
  )
