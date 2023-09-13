##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(LowerLIT
  CheckLifetimes.cpp
  CheckRecursiveStructs.cpp
  ConstraintReduction.cpp
  ConstraintSet.cpp
  LowerClosures.cpp
  LowerLIT.cpp
  LowerPreElaboratedLIT.cpp
  LowerSemanticCF.cpp
  LowerStructs.cpp
  StripParserMetadata.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  LITDialect
  MLIRIR
  )
