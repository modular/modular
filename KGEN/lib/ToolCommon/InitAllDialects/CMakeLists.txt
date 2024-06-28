##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENInitAllDialects
  InitAllDialects.cpp

  LINK_LIBS PUBLIC
  DebugInfoDialect
  HLCFDialect
  KGENDialect
  LITDialect
  POPDialect
  CODialect
  CustomDialect

  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRNVVMDialect
  )
