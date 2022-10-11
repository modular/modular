##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupportML
  DType.cpp
  TensorShape.cpp
  TensorSpec.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  )
