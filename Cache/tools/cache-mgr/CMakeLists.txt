##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(cache-mgr
  cache-mgr.cpp

  LINK_LIBS PRIVATE
  MCache
  MSupport
  MSupportGlobals
  MLIRSupport
  LLCLRuntimeGlobals
  )
