##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupportML
  BCast.cpp
  CompiledFrameworkLabel.cpp
  DType.cpp
  MatMulBCast.cpp
  TensorShape.cpp
  TensorSpec.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  )
