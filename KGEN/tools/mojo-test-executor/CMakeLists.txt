##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo-test-executor
  main.cpp

  LINK_LIBS PUBLIC
  MLIRLspServerSupportLib
  KGENCompilerRT
  KGENSupport
  AsyncRTCompilerSupport
  AsyncRTRuntimeGlobals
  MSupportGlobals
  MAsyncRTInit
  MojoJupyter
  MojoParser
  MojoTesting
)
