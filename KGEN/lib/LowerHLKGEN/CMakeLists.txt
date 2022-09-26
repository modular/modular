##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(LowerHLKGEN
  ConstraintSet.cpp
  LowerHLKGEN.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  HLKGENDialect
  MLIRIR
  )
