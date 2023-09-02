##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-translate
  kgen-translate.cpp

  LINK_LIBS PRIVATE
  KGENCLOptions
  LLCLRuntimeGlobals
  MSupportGlobals
  MojoParser
  MArchTarget
  MLIRBuiltinToLLVMIRTranslation
  MLIRTranslateLib
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
)
