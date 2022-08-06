#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_modular_library(KGENKernelElaborator
  BetterCloning.cpp
  KernelElaborator.cpp

  LINK_LIBS PUBLIC
  KGENDialect
  MLIRIR
  MLIRSupport
  MSupport
  )
