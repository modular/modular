##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(MCompilerSupport
  MLIRDType.cpp

  LINK_LIBS PUBLIC
  MSupportML
  MLIRIR
)
