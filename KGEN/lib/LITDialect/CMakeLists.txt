##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(LITDialect
  LITAttrs.cpp
  LITDialect.cpp
  LITOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MODULAR_MAIN_INCLUDE_DIR}/LIT/LITDialect

  DEPENDS
  KGENDialect
  POPDialect
  MLIRKGENIncGen
  MLIRLITIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MSupport
  MSupportML
  MLIRIR
)
