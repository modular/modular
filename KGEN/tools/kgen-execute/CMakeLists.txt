##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-execute
  kgen-execute.cpp

  LINK_LIBS PRIVATE
  KGENCLOptions
  MLIRParser
  KGENExecutionEngine
  )
