##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENCLOptions
  CLOptions.cpp

  LINK_LIBS PUBLIC
  KGENExecutionEngine
  MLIRLLVMDialect
  MSupport

  DISALLOWED_DEPENDS
  )
