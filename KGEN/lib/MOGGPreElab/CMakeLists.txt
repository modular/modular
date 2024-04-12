##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MOGGPreElabPasses
  OutlineMOGGFuncs.cpp
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
