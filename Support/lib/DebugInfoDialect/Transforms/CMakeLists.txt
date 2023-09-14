##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_library(DebugInfoTransforms
  Conversion.cpp
  DebugInfoStrip.cpp
  SyntheticDebugInfoBuilder.cpp

  DEPENDS
  DebugInfoTransformPassesIncGen

  LINK_LIBS PUBLIC
  DebugInfoDialect
  MLIRTransformUtils
  MLIRParser
)
