##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(support-dialect-opt
  support-dialect-opt.cpp

  LINK_LIBS PRIVATE
  DebugInfoDialect
  DebugInfoToLLVM
  MDialect
  MSupport
  MProfiling
  MSupportGlobals
  MLIRFuncDialect
  MLIRIndexDialect
  MLIROptLib
  MLIRTransforms
  )
