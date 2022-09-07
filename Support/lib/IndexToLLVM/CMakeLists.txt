##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(IndexToLLVM
  IndexToLLVM.cpp

  DEPENDS
  IndexToLLVMIncGen

  LINK_LIBS PUBLIC
  IndexDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
)
