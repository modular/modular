##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(MCompilerSupport
  MLIRDenseAttrStorage.cpp
  MLIRDType.cpp

  LINK_LIBS PUBLIC
  MSupportML
  MLIRIR
)
