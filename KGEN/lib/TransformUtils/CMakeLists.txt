##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENTransformUtils
  ControlFlowUtils.cpp
  InliningUtils.cpp
  Walkers.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MSupport
  MCompilerSupport
  HLCFDialect
  KGENDialect
)
