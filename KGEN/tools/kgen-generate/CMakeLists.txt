#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

set(LLVM_LINK_COMPONENTS
  Support
)

add_modular_tool(kgen-generate
  GenerateKernels.cpp
  kgen-generate.cpp

  LINK_LIBS PRIVATE
  KGENDialect
  MetaDialect
  MLIRArithmetic
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MSupport
)
