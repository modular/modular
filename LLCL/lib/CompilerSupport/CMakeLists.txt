##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(LLCLCompilerSupport
  MLIRLocationDecoder.cpp

  LINK_LIBS PUBLIC
  LLCLSupport
  MLIRIR
)
