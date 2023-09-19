##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENPipeline
  Pipeline.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  DebugInfoToLLVM
  DebugInfoTransforms
  KGENElaborator
  KGENTransforms
  LowerLIT
)
