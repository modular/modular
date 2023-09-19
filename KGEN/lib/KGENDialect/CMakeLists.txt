##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(KGENDialect
  KGENAttrInterfaces.cpp
  KGENAttrs.cpp
  KGENDialect.cpp
  KGENDialectBytecode.cpp
  KGENInterfaces.cpp
  KGENOps.cpp
  KGENParameters.cpp
  KGENTypeInterfaces.cpp
  KGENTypes.cpp
  KGENUtils.cpp
  ParameterEvaluator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MODULAR_MAIN_INCLUDE_DIR}/KGEN/KGENDialect

  DEPENDS
  MLIRKGENIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  DebugInfoDialect
  HLCFDialect
  InterpreterInterface
  MCompilerSupport
  MSupport
  MSupportML
  MLIRIR
  MLIRFunctionInterfaces
)
