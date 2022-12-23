##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(MDialect
  MAttrs.cpp
  MDialect.cpp
  MTypeInterfaces.cpp
  MTypes.cpp

  DEPENDS
  MLIRMIncGen

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  MLIRIR
)
