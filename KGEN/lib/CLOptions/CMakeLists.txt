##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENCLOptions
  CLOptions.cpp

  DEPENDS
  MLIRDebugInfoIncGen
  InterpreterInterfaceIncGen
  MLIRKGENIncGen
  MLIRMIncGen
  MLIRPOPIncGen

  LINK_LIBS PUBLIC
  KGENExecutionEngine
  MLIRLLVMDialect
  MSupport

  DISALLOWED_DEPENDS
  )
