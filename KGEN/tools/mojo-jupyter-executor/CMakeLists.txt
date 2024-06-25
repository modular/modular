##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo-jupyter-executor
  main.cpp

  DEPENDS
  mojo-sdk

  LINK_LIBS PUBLIC
  KGENCompilerRT
  KGENSupport
  LLCLRuntimeGlobals
  MSupportGlobals
  MLLCLInit
  MojoJupyter
)
