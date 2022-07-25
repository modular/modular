#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

add_modular_dialect_library(KGENDialect
  KGENAttrs.cpp
  KGENDialect.cpp
  KGENOps.cpp
  KGENParameters.cpp
  ParameterEvaluator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MODULAR_MAIN_INCLUDE_DIR}/KGEN/KGENDialect

  DEPENDS
  MLIRKGENIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MSupport
  MLIRIR
)
