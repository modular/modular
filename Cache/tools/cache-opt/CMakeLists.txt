##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(cache-opt
  cache-opt.cpp

  LINK_LIBS PRIVATE
  CacheDialect
  CachePasses
  LLCLRuntimeGlobals
  MCache
  MSupport
  MSupportGlobals
  MLIRFuncDialect
  MLIRArithDialect
  MLIROptLib
  MLIRTransforms
  )
