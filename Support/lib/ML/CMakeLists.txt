##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupportML
  CompiledFrameworkLabel.cpp
  DType.cpp
  TensorShape.cpp
  TensorSpec.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  )
