##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENElaborator
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
  MArchTarget
  MLIRIR
  MLIRSupport
  MSupport
  MSupportML
  )
