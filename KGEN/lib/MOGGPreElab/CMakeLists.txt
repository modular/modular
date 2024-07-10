##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MOGGPreElabPasses
  AutoParameterize.cpp
  Helpers.cpp
  MOGGAnnotate.cpp
  OutlineMOGGFuncs.cpp
  MOGGPreElabPipeline.cpp
  SliceMOGGFuncs.cpp
  UserLibraryChecker.cpp
  MOGGIdentifyKernels.cpp

  DEPENDS
  MLIRKGENMOGGPreElabPasses

  LINK_LIBS PUBLIC
  HLCFAnalysis
  KGENDialect
  POPDialect
  LITDialect
  MLIRTransformUtils
)
