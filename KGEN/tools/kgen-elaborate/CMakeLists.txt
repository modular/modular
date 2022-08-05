#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_modular_tool(kgen-elaborate
  kgen-elaborate.cpp

  LINK_LIBS PRIVATE
  KGENKernelElaborator
  MLIROptLib
  MLIRParser
  MSupport
)
