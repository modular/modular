##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENElaborator
  Elaborator.cpp
  ElaboratorV2.cpp
  IncludeResolution.cpp
  IREvaluator.cpp
  SelectFastestFunction.cpp

  LINK_LIBS PUBLIC
  DebugInfoDialect
  CacheDialect
  KGENCompilerRT
  KGENDialect
  KGENExecutionEngine
  InterpreterInterface
  MLIRIR
  MLIRSupport
  MSupport
  MSupportML
  )
