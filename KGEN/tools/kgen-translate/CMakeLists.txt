##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-translate
  kgen-translate.cpp

  LINK_LIBS PRIVATE
  KGENTranslationRegistry
  MojoParser
  LLCLCompilerSupport
  LLCLRuntimeGlobals
  MSupportGlobals
  MLLCLInit
  MSupport
  MArchTarget
  MProfiling

  MLIRTranslateLib
  MLIRTargetLLVMIRExport
)
