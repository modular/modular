##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENPipeline
  Pipeline.cpp

  DEPENDS
  MLIRKGENPasses
  MLIRKGENMOGGPreElabPasses

  LINK_LIBS PUBLIC
  DebugInfoToLLVM
  DebugInfoTransforms
  KGENElaborator
  KGENToLLVM
  KGENTransforms
  MOGGPreElabPasses
  LowerLIT
)
