##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoLLDB SHARED
  Plugin.cpp

  REPL/MojoREPL.cpp

  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  LINK_LIBS
  LLCLRuntimeGlobals

  liblldb
)
