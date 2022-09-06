##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENElaborator
  BetterCloning.cpp
  Elaborator.cpp

  LINK_LIBS PUBLIC
  KGENDialect
  MLIRIR
  MLIRSupport
  MSupport
  MSupportML
  )
