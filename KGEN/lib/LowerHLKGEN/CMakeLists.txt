#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_modular_conversion_library(LowerHLKGEN
  LowerHLKGEN.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  HLKGENDialect
  MetaDialect
  MLIRIR
  )
