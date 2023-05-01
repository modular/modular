##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(LowerLIT
  CheckLifetimes.cpp
  ConstraintReduction.cpp
  ConstraintSet.cpp
  LowerClosures.cpp
  LowerLIT.cpp
  LowerStructs.cpp
  LowerSemanticCF.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  LITDialect
  MLIRIR
  )
