##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MOGGPreElabPasses
  Helpers.cpp
  OutlineMOGGFuncs.cpp
  MOGGPreElabPipeline.cpp
  RemoveDecorators.cpp
  SliceMOGGFuncs.cpp
  UserLibraryChecker.cpp

  DEPENDS
  MLIRKGENMOGGPreElabPasses

  LINK_LIBS PUBLIC
  HLCFAnalysis
  KGENDialect
  POPDialect
  LITDialect
  MLIRTransformUtils
)
