##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENElaborator
  Elaborator.cpp
  IREvaluator.cpp
  SelectFastestFunction.cpp

  LINK_LIBS PUBLIC
  DebugInfoDialect
  CacheDialect
  KGENCLOptions
  KGENCompilerRT
  KGENDialect
  KGENExecutionEngine
  InterpreterInterface
  MLIRIR
  MLIRSupport
  MSupport
  MSupportML
  )
