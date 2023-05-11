##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo-jupyter-executor
  main.cpp

  LINK_LIBS PUBLIC
  KGENCompilerRT
  LLCLRuntimeGlobals
  MSupportGlobals
  MojoJupyter
)
