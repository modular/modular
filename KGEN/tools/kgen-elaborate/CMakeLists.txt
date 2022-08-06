#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_modular_tool(kgen-elaborate
  kgen-elaborate.cpp

  LINK_LIBS PRIVATE
  # Dialects we want available to the parser.
  HLKGENDialect
  MLIRArithmeticDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  POPDialect

  # Tool depenencies
  KGENKernelElaborator
  MLIROptLib
  MLIRParser
  MSupport
)
