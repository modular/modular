##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupportML
  BroadcastShape.cpp
  CompiledFrameworkLabel.cpp
  DType.cpp
  Fill.cpp
  MatMulBroadcast.cpp
  RangeUtils.cpp
  TensorShape.cpp
  TensorSpec.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  MSupport
)
