##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENToLLVM
  ConvertKGENToLLVM.cpp
  ConvertPOPToLLVM.cpp
  LLVMLoweringUtils.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  MetaDialect
  POPDialect
  MLIRIR
  MLIRLLVMCommonConversion
  )
