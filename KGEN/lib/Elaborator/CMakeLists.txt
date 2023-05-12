##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENElaborator
  Elaborator.cpp
  IREvaluator.cpp

  DEPENDS
  KGENVersion
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  DebugInfoDialect
  CacheDialect
  KGENCLOptions
  KGENDialect
  LITDialect
  InterpreterInterface
  MLIRIR
  MLIRSupport
  MSupport
  MSupportML
  )
