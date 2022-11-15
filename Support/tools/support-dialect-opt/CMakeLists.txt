##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(support-dialect-opt
  support-dialect-opt.cpp

  LINK_LIBS PRIVATE
  HLCFDialect
  HLCFToLLVM
  MDialect
  MLIRFuncDialect
  MLIROptLib
  MLIRTransforms
  )
