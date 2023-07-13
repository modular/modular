##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(LITDialect
  LITAttrs.cpp
  LITDialect.cpp
  LITOps.cpp
  LifetimeTrackable.cpp

  ADDITIONAL_HEADER_DIRS
  ${MODULAR_MAIN_INCLUDE_DIR}/LIT/LITDialect

  DEPENDS
  MLIRKGENIncGen
  MLIRLITIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  DebugInfoDialect
  HLCFDialect
  KGENDialect
  POPDialect
  MSupport
  MSupportML
  MLIRIR
)
