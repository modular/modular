##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENElaborator
  BetterCloning.cpp
  Elaborator.cpp
  SelectFastestFunction.cpp

  LINK_LIBS PUBLIC
  KGENDialect
  KGENExecutionEngine
  MLIRIR
  MLIRSupport
  MSupport
  MSupportML
  )
