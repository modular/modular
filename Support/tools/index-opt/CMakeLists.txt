##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(index-opt
  index-opt.cpp

  LINK_LIBS PRIVATE
  IndexDialect
  IndexToLLVM
  MLIRFuncDialect
  MLIROptLib
  MLIRTransforms
)
