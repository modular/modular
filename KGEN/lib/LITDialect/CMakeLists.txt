##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(LITDialect
  LifetimeTrackable.cpp
  LITAttrs.cpp
  LITDialect.cpp
  LITInterfaces.cpp
  LITOps.cpp
  LITTypes.cpp
  LITUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MODULAR_MAIN_INCLUDE_DIR}/LIT/LITDialect

  DEPENDS
  MLIRDebugInfoIncGen
  MLIRKGENIncGen
  MLIRLITIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  DebugInfoDialect
  HLCFDialect
  KGENDialect
  POPDialect
  CODialect
  MSupport
  MSupportML
  MLIRIR
)
