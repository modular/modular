##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-reduce
  kgen-reduce.cpp

  LINK_LIBS PRIVATE
  KGENToolCommon
  MSupportGlobals
  MLIRIR
  LLCLRuntimeGlobals
)
