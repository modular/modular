##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_compile_definitions(
  MODULAR_KERNELS_BUILD_TYPE="${MODULAR_KERNELS_BUILD_TYPE}")

add_modular_dialect_library(MDialect
  MAttrInterfaces.cpp
  MAttrs.cpp
  MDialect.cpp
  MTypeInterfaces.cpp
  MTypes.cpp

  DEPENDS
  MLIRMIncGen

  LINK_LIBS PUBLIC
  LLVMTargetParser
  MLIRIR
)
