##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_library(DebugInfoTransforms
  Conversion.cpp

  LINK_LIBS PUBLIC
  DebugInfoDialect
)
