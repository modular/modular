#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

set(LLVM_LINK_COMPONENTS
  Support
  )

add_modular_tool(kgen-opt
  kgen-opt.cpp
  LINK_LIBS PRIVATE
  KGENDialect
  MetaDialect
  GRTDialect

  MLIRIR
  MLIRArithmeticDialect
  MLIRLLVMDialect
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms
  )
