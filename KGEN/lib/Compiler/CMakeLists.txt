##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENCompiler
  KGENCompiler.cpp
  EmitFuncHeader.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  DebugInfoToLLVM
  DebugInfoTransforms

  KGENCompilerRT
  KGENElaborator
  KGENEmitObject
  KGENTransforms
  MojoParser
  LowerLIT

  KGENDialect
  LITDialect
  POPDialect

  MLIRIR
  )
