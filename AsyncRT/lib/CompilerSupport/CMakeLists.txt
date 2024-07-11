##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(AsyncRTCompilerSupport
  AsyncSideEffectMap.cpp
  Context.cpp
  LLVMThreadPool.cpp
  MLIRLocationDecoder.cpp

  LINK_LIBS PUBLIC
  AsyncRTSupport
  AsyncRTRuntime
  AsyncRTRuntimeGlobals
  MLIRIR
)
