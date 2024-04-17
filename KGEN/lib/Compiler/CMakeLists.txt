##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENCompiler
  KGENCompiler.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  DebugInfoToLLVM
  DebugInfoTransforms

  KGENElaborator
  KGENEmitObject
  KGENTransforms
  LowerLIT
  MSupport

  MDialect
  KGENDialect
  LITDialect
  POPDialect

  MLIRIR
)
