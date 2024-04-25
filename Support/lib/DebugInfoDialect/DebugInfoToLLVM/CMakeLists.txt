##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(DebugInfoToLLVM
  DebugInfoToLLVM.cpp
  Target/TargetAdapter.cpp

  DEPENDS
  DebugInfoToLLVMIncGen

  LINK_LIBS PUBLIC
  DebugInfoTransforms
  DebugInfoDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
)
