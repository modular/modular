##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENElaborator
  Elaborator.cpp
  IREvaluator.cpp
  SelectFastestFunction.cpp

  DEPENDS
  KGENConfig

  LINK_LIBS PUBLIC
  DebugInfoDialect
  CacheDialect
  KGENCLOptions
  KGENCompilerRT
  KGENDialect
  KGENExecutionEngine
  LITDialect
  InterpreterInterface
  MLIRIR
  MLIRSupport
  MSupport
  MSupportML
  )
