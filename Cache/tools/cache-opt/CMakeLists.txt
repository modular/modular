##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(cache-opt
  cache-opt.cpp

  LINK_LIBS PRIVATE
  CacheDialect
  LLCLRuntimeGlobals
  MCache
  MSupport
  MSupportGlobals
  MSupportInit
  MDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIROptLib
  MLIRTransforms
  )
