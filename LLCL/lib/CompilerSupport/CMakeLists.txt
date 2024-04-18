##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(LLCLCompilerSupport
  AsyncSideEffectMap.cpp
  Context.cpp
  LLVMThreadPool.cpp
  MLIRLocationDecoder.cpp

  LINK_LIBS PUBLIC
  LLCLSupport
  LLCLRuntime
  LLCLRuntimeGlobals
  MLIRIR
)
