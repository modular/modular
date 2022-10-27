##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENToLLVM
  KGENToLLVMPipeline.cpp
  LowerKGENToLLVM.cpp
  LowerPOPToLLVM.cpp
  LowerSCFToLLVM.cpp
  LLVMLoweringUtils.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  POPDialect
  MCompilerSupport

  MLIRIR
  MLIRIndexToLLVM
  MLIRLLVMCommonConversion
  MLIRSCFDialect
  )
