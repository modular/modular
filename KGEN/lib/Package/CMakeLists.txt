##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(KGENPackage
  Package.cpp

  LINK_LIBS PUBLIC
  KGENCompiler
  KGENDialect
  KGENExecutionEngine
  KGENEmitObject
  LLCLRuntime
  LLCLRuntimeGlobals
  MCache
  MDialect
  MSupportGlobals

  MLIRIR
  MLIRPass
  MLIRBytecodeWriter
  )
