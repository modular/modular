##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(HLCFToLLVM
  HLCFToLLVM.cpp

  DEPENDS
  HLCFToLLVMIncGen

  LINK_LIBS PUBLIC
  HLCFDialect

  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
)
