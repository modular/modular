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
  KGENTransforms
  MojoParser
  LowerLIT

  KGENDialect
  LITDialect
  POPDialect

  MLIRIR
  )
