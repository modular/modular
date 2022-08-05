#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_modular_library(KGENKernelElaborator
  BetterCloning.cpp
  KernelElaborator.cpp

  LINK_LIBS PUBLIC
  HLKGENDialect
  KGENDialect
  MetaDialect
  MLIRIR
  MLIRArithmeticDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  MSupport
  POPDialect
  )
