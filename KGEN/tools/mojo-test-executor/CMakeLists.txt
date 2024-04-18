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
  LLCLCompilerSupport
  LLCLRuntimeGlobals
  MSupportGlobals
  MSupportInit
  MojoJupyter
  MojoParser
  MojoTesting
)
