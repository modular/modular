##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(MSupportML
  DType.cpp
  TensorShape.cpp
  TensorSpec.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  LLVMSupport
  )
