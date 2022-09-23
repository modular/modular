##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(LowerZAPToPOP
  LowerZAPToPOP.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  POPDialect
  ZAPDialect

  MLIRTransforms
)
