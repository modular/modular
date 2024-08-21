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
  AsyncRTCompilerSupport
  AsyncRTRuntimeGlobals
  MSupportGlobals
  Init
  MSupport
  MArchTarget
  MProfiling

  MLIRTranslateLib
  MLIRTargetLLVMIRExport
)
