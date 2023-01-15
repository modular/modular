##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENCompiler
  KGENCompiler.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  DebugInfoToLLVM
  DebugInfoTransforms

  KGENElaborator
  KGENTransforms
  ParseLit
  LowerLIT
  LowerZAPToPOP

  KGENDialect
  LITDialect
  ZAPDialect
  POPDialect

  MLIRIR
  )
