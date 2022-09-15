##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENToLLVM
  ConvertKGENToLLVM.cpp
  ConvertPOPToLLVM.cpp
  ConvertSCFToLLVM.cpp
  LLVMLoweringUtils.cpp
  LowerKGENToLLVM.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  MetaDialect
  POPDialect
  IndexToLLVM
  MCompilerSupport

  MLIRIR
  MLIRLLVMCommonConversion
  MLIRSCFDialect
  )
