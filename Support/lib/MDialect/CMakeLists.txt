##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(MDialect
  MAttrInterfaces.cpp
  MAttrs.cpp
  MDialect.cpp
  MTypeInterfaces.cpp
  MTypes.cpp

  DEPENDS
  MVersion
  MLIRMIncGen

  LINK_LIBS PUBLIC
  LLVMTargetParser
  MLIRIR
  MVersion
)
