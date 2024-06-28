##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(CustomDialect
  CustomDialect.cpp

  DEPENDS
  MLIRCustomIncGen
)
