##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENElaborator
  BetterCloning.cpp
  Elaborator.cpp
  IncludeResolution.cpp
  SelectFastestFunction.cpp

  LINK_LIBS PUBLIC
  DebugInfoDialect
  KGENDialect
  KGENExecutionEngine
  MLIRIR
  MLIRSupport
  MSupport
  MSupportML
  )
