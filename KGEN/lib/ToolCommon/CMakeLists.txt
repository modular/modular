##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENToolCommon
  CLOptions.cpp
  CompilationOptions.cpp
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
