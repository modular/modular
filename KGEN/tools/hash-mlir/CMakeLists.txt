##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(hash-mlir
  hash-mlir.cpp

  LINK_LIBS PRIVATE
  KGENToolCommon
  MSupportGlobals
  MLIRIR
  AsyncRTRuntimeGlobals
  MLIRBytecodeWriter
)
