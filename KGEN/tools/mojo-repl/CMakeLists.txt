##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo-repl
  main.cpp

  LINK_LIBS PUBLIC
  KGENCompilerRT
  LLCLRuntimeGlobals
)
