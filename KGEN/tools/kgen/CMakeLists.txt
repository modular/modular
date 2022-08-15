#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_modular_tool(kgen
  kgen.cpp

  LINK_LIBS PRIVATE
  KGENCLOptions
  HLKGENDialect
  KGENKernelElaborator
  KGENDialect
  KGENToLLVM
  LowerHLKGEN
  MetaDialect
  POPDialect

  MLIRParser
  MLIRSCFDialect
  MLIRArithmeticToLLVM
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
)
