##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(MCompilerSupport
  MLIRDType.cpp
  OperationUtils.cpp
  VerifyUtils.cpp

  LINK_LIBS PUBLIC
  MSupportML
  MLIRIR
)
