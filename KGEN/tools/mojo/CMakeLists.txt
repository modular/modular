##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo
  mojo.cpp

  LINK_COMPONENTS
  native
  nativecodegen

  LINK_LIBS PRIVATE
  KGENCLOptions
  KGENCompiler
  KGENCompilerRT
  KGENDialect
  LITDialect
  LLCLRuntimeGlobals
  POPDialect

  MLIRBytecodeWriter
  MLIRIndexDialect
  MLIRParser
)
