##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(LowerLIT
  ConstraintReduction.cpp
  ConstraintSet.cpp
  LowerLIT.cpp
  LowerStructs.cpp
  LowerTerminators.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  LITDialect
  MLIRIR
  )
