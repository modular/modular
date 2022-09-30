##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(MCompilerSupport
  DenseArray.cpp
  MLIRDType.cpp

  LINK_LIBS PUBLIC
  MSupportML
  MLIRIR
)
