##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENElaborator
  Elaborator.cpp
  ElaboratorV2.cpp
  IncludeResolution.cpp
  SelectFastestFunction.cpp
  SymbolicExpressions.cpp

  LINK_LIBS PUBLIC
  DebugInfoDialect
  CacheDialect
  KGENDialect
  KGENExecutionEngine
  InterpreterInterface
  MLIRIR
  MLIRSupport
  MSupport
  MSupportML
  )
