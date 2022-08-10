#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_modular_dialect_library(MetaDialect
  MetaDialect.cpp
  MetaOps.cpp
  MetaTypeConverter.cpp

  ADDITIONAL_HEADER_DIRS
  ${MODULAR_MAIN_INCLUDE_DIR}/KGEN/MetaDialect

  DEPENDS
  MLIRMetaIncGen

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  KGENDialect
)
