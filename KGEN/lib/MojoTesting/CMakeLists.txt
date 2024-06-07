##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoTesting
  Test.cpp

  LINK_LIBS PUBLIC
  MojoParser
  MojoTooling
  MSupport
  KGENSupport
  LLCLRuntimeGlobals
  MSupportGlobals
  MLIRLspServerSupportLib
)
