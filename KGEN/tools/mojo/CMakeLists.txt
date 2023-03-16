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
  POPDialect

  MLIRBytecodeWriter
  MLIRIndexDialect
  MLIRParser
)
