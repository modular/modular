##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-translate
  kgen-translate.cpp

  LINK_LIBS PRIVATE
  MLIRTranslateLib
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
)
