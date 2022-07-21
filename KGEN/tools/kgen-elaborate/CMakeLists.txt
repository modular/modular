#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

set(LLVM_LINK_COMPONENTS
  Support
)

add_modular_tool(kgen-elaborate
  BetterCloning.cpp
  KernelElaborator.cpp
  kgen-elaborate.cpp

  LINK_LIBS PRIVATE
  KGENDialect
  MetaDialect
  MLIRArithmeticDialect
  MLIRLLVMDialect
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MSupport
)
