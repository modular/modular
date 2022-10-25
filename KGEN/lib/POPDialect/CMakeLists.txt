##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(POPDialect
  POPAttrs.cpp
  POPDialect.cpp
  POPOps.cpp
  POPTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MODULAR_MAIN_INCLUDE_DIR}/KGEN/PSLDialect

  DEPENDS
  MLIRPOPIncGen

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  KGENDialect
  MCompilerSupport
  MDialect
)
