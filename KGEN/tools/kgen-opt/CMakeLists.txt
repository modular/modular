#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_modular_tool(kgen-opt
  kgen-opt.cpp
  LINK_LIBS PRIVATE
  HLKGENDialect
  KGENDialect
  MetaDialect
  POPDialect
  GRTDialect
  KGENToLLVM

  MLIRArithmeticDialect
  MLIRIR
  MLIRLLVMDialect
  MLIROptLib
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  )
