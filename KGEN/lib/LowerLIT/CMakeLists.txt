##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(LowerLIT
  ConstraintReduction.cpp
  ConstraintSet.cpp
  LowerAsyncFunctions.cpp
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
