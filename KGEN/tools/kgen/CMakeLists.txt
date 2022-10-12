##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen
  EmitFuncHeader.cpp
  EmitFuncObject.cpp
  kgen.cpp

  LINK_LIBS PRIVATE
  KGENCLOptions
  LITDialect
  IndexDialect
  KGENCompilerRT
  KGENDialect
  KGENElaborator
  KGENTransforms
  ZAPDialect
  LowerLIT
  LowerZAPToPOP
  POPDialect

  MLIRBytecodeWriter
  MLIRParser
)
