##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(HLKGENDialect
  HLKGENDialect.cpp
  HLKGENOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MODULAR_MAIN_INCLUDE_DIR}/HLKGEN/HLKGENDialect

  DEPENDS
  KGENDialect
  MLIRKGENIncGen
  MLIRHLKGENIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MSupport
  MSupportML
  MLIRIR
)
