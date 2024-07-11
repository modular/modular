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
  KGENObjectCompiler
  KGENPipeline
  KGENTransforms
  LowerLIT
  MSupport
  AsyncRTCompilerSupport

  KGENDialect
  LITDialect
  POPDialect

  MLIRIR
)
