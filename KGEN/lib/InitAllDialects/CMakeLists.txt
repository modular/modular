##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(InitAllKGENDialects
  InitAllDialects.cpp

  LINK_LIBS PUBLIC
  DebugInfoDialect
  HLCFDialect
  KGENDialect
  LITDialect
  POPDialect
  CacheDialect

  MLIRIndexDialect
  MLIRLLVMDialect
)
